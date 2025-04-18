import torch as th
import torch.nn as nn
import torch.nn.functional as F


class MADDPGCritic(nn.Module):
    def __init__(self, scheme, args):
        super(MADDPGCritic, self).__init__()
        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.input_shape = self._get_input_shape(scheme) + self.n_actions * self.n_agents
        self.output_type = "q"

        # Set up network layers
        self.fc1 = nn.Linear(self.input_shape, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc3 = nn.Linear(args.rnn_hidden_dim, 1)

    def forward(self, inputs, actions, hidden_state=None):
        if actions is not None:
            inputs = th.cat([inputs.reshape(-1, self.input_shape - self.n_actions * self.n_agents),
                             actions.contiguous().view(-1, self.n_actions * self.n_agents)], dim=-1)
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q, hidden_state

    def _get_input_shape(self, scheme):
        # The centralized critic takes the state input, not observation
        input_shape = scheme["state"]["vshape"]
        return input_shape
    


class SSD_MADDPGCritic(nn.Module):
    def __init__(self, scheme, args):
        super(SSD_MADDPGCritic, self).__init__()

        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents

        self.C = scheme["state"]["vshape"][0]

        self.state_H = scheme["state"]["vshape"][1]
        self.state_W = scheme["state"]["vshape"][2]

        self.obs_H = scheme["obs"]["vshape"][1]
        self.obs_W = scheme["obs"]["vshape"][2]
        

        self.conv_state = nn.Sequential(
        nn.Conv2d(in_channels=self.C, out_channels=args.out_channels, \
                    kernel_size=args.kernel_size, stride=(1, 1), padding='valid'),
        nn.ReLU(), 
        nn.Flatten(), 
        nn.Linear(args.out_channels * (self.state_H - 2) * (self.state_W - 2), args.rnn_hidden_dim), 
        nn.ReLU()
        )
        
        self.conv_obs = nn.Sequential(
        nn.Conv2d(in_channels=self.C, out_channels=args.out_channels, \
                    kernel_size=args.kernel_size, stride=(1, 1), padding='valid'),
        nn.ReLU(), 
        nn.Flatten(), 
        nn.Linear(args.out_channels * (self.obs_H - 2) * (self.obs_W - 2), args.rnn_hidden_dim), 
        nn.ReLU()
        )

        input_shape = self._get_input_shape(scheme)
        self.output_type = "q"

        # Set up network layers
        self.fc1 = nn.Linear(input_shape, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, obs, state, actions):
        inputs = self._build_inputs(obs, state, actions,)
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q


    # def forward(self, inputs, actions, hidden_state=None):
    #     if actions is not None:
    #         inputs = th.cat([inputs.reshape(-1, self.input_shape - self.n_actions * self.n_agents),
    #                          actions.contiguous().view(-1, self.n_actions * self.n_agents)], dim=-1)
    #     x = F.relu(self.fc1(inputs))
    #     x = F.relu(self.fc2(x))
    #     q = self.fc3(x)
    #     return q, hidden_state


    def _build_inputs(self, obs, state, actions):
        
        inputs = []
        # state
        state = state/255.0
        state = self.conv_state(state.reshape(-1, self.C, self.state_H, self.state_W))
        inputs.append(state.reshape(-1, self.args.rnn_hidden_dim))

        # observation
        obs = obs/255.0
        obs = self.conv_obs(obs.reshape(-1, self.C, self.obs_H, self.obs_W))
        inputs.append(obs.reshape(-1, self.args.rnn_hidden_dim * self.n_agents))

        # actions (masked out by agent)
        actions = actions.contiguous().view(-1, self.n_actions * self.n_agents)
        # batch["actions_onehot"][:, ts].view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1)
        inputs.append(actions)

        # last actions 上一个动作 [bs, max_t, n_agents, n_actions]

        # 生成并扩展一个单位矩阵，用于表示每个智能体的身份信息，并将其广播到整个批次和所有时间步 
        # [bs, max_t, n_agents, n_agents]

        # 将所有输入张量连接在一起，形成一个形状为 [bs, max_t, n_agents, -1] 的张量
        inputs = th.cat(inputs, dim=-1)
        return inputs

    def _get_input_shape(self, scheme):
        # state
        input_shape = self.args.rnn_hidden_dim
        # observation
        input_shape += self.args.rnn_hidden_dim * self.n_agents
        # actions and last actions
        input_shape += scheme["actions_onehot"]["vshape"][0] * self.n_agents
        # # agent id
        # input_shape += self.n_agents
        return input_shape