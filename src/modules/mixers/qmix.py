import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class QMixer(nn.Module):
    def __init__(self, args):
        super(QMixer, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        if self.args.env == "SSD":  # SSD环境
            self.C = args.state_shape[0]
            self.H = args.state_shape[1]
            self.W = args.state_shape[2]
            self.conv = nn.Sequential(
            nn.Conv2d(in_channels=self.C, out_channels=args.out_channels, \
                        kernel_size=args.kernel_size, stride=(1, 1), padding='valid'),
            nn.Tanh(),
            nn.Flatten()
            )
            self.state_dim = int(args.out_channels * (self.H - 2) * (self.W - 2))
        else:
            self.state_dim = int(np.prod(args.state_shape))

        self.embed_dim = args.mixing_embed_dim

        if getattr(args, "hypernet_layers", 1) == 1:
            self.hyper_w_1 = nn.Linear(self.state_dim, self.embed_dim * self.n_agents)
            self.hyper_w_final = nn.Linear(self.state_dim, self.embed_dim)
        elif getattr(args, "hypernet_layers", 1) == 2:
            hypernet_embed = self.args.hypernet_embed
            self.hyper_w_1 = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                           nn.ReLU(),
                                           nn.Linear(hypernet_embed, self.embed_dim * self.n_agents))
            self.hyper_w_final = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                           nn.ReLU(),
                                           nn.Linear(hypernet_embed, self.embed_dim))
        elif getattr(args, "hypernet_layers", 1) > 2:
            raise Exception("Sorry >2 hypernet layers is not implemented!")
        else:
            raise Exception("Error setting number of hypernet layers.")

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                               nn.ReLU(),
                               nn.Linear(self.embed_dim, 1))

    def forward(self, agent_qs, states):
        # agent_qs为(bs * max_lex * n_agent)
        # 输入就是动作对应的Q值，以及状态，那么问题来了，
        # 它对应的Q值中，其他动作的Q值是什么呢
        bs = agent_qs.size(0)
        if self.args.env == "SSD":
            states = states / 255.0
            states = states .reshape(-1, self.C, self.H, self.W)
            states = self.conv(states)
        else:
            states = states.reshape(-1, self.state_dim)
        # 此时的动作已经选好了
        agent_qs = agent_qs.view(-1, 1, self.n_agents)
        # First layer
        w1 = th.abs(self.hyper_w_1(states)) #加绝对值
        b1 = self.hyper_b_1(states)
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        # 智能体的价值们，与w1相乘，再＋b1，此时已经得到了Q_tot，只不过此时是32维的
        hidden = F.elu(th.bmm(agent_qs, w1) + b1)
        # Second layer
        w_final = th.abs(self.hyper_w_final(states))
        w_final = w_final.view(-1, self.embed_dim, 1)
        # State-dependent bias
        v = self.V(states).view(-1, 1, 1)
        # Compute final output
        y = th.bmm(hidden, w_final) + v
        # Reshape and return
        q_tot = y.view(bs, -1, 1)
        return q_tot




class QMixer_Noabs(nn.Module):
    def __init__(self, args):
        super(QMixer_Noabs, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        if self.args.env == "SSD":  # SSD环境
            self.C = args.state_shape[0]
            self.H = args.state_shape[1]
            self.W = args.state_shape[2]
            self.conv = nn.Sequential(
            nn.Conv2d(in_channels=self.C, out_channels=args.out_channels, \
                        kernel_size=args.kernel_size, stride=(1, 1), padding='valid'),
            nn.Tanh(),
            nn.Flatten()
            )
            self.state_dim = int(args.out_channels * (self.H - 2) * (self.W - 2))
        else:
            self.state_dim = int(np.prod(args.state_shape))

        self.embed_dim = args.mixing_embed_dim

        if getattr(args, "hypernet_layers", 1) == 1:
            self.hyper_w_1 = nn.Linear(self.state_dim, self.embed_dim * self.n_agents)
            self.hyper_w_final = nn.Linear(self.state_dim, self.embed_dim)
        elif getattr(args, "hypernet_layers", 1) == 2:
            hypernet_embed = self.args.hypernet_embed
            self.hyper_w_1 = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                           nn.ReLU(),
                                           nn.Linear(hypernet_embed, self.embed_dim * self.n_agents))
            self.hyper_w_final = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                           nn.ReLU(),
                                           nn.Linear(hypernet_embed, self.embed_dim))
        elif getattr(args, "hypernet_layers", 1) > 2:
            raise Exception("Sorry >2 hypernet layers is not implemented!")
        else:
            raise Exception("Error setting number of hypernet layers.")

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                               nn.ReLU(),
                               nn.Linear(self.embed_dim, 1))

    def forward(self, agent_qs, states):
        # agent_qs为(bs * max_lex * n_agent)
        bs = agent_qs.size(0)
        if self.args.env == "SSD":
            states = states / 255.0
            states = states .reshape(-1, self.C, self.H, self.W)
            states = self.conv(states)
        else:
            states = states.reshape(-1, self.state_dim)
        # 此时的动作已经选好了
        agent_qs = agent_qs.view(-1, 1, self.n_agents)
        # First layer
        w1 = self.hyper_w_1(states) #加绝对值
        b1 = self.hyper_b_1(states)
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        # 智能体的价值们，与w1相乘，再＋b1，此时已经得到了Q_tot，只不过此时是32维的
        hidden = F.elu(th.bmm(agent_qs, w1) + b1)
        # Second layer
        w_final = self.hyper_w_final(states)
        w_final = w_final.view(-1, self.embed_dim, 1)
        # State-dependent bias
        v = self.V(states).view(-1, 1, 1)
        # Compute final output
        y = th.bmm(hidden, w_final) + v
        # Reshape and return
        q_tot = y.view(bs, -1, 1)
        return q_tot
