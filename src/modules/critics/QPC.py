import torch as th
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np

class QPC_CentralVCritic(nn.Module):
    def __init__(self, scheme, args):
        super(QPC_CentralVCritic, self).__init__()

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

    def forward(self, batch, t=None):
        inputs = self._build_inputs(batch, t=t)
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q

    def _build_inputs(self, batch, t=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t+1)
        inputs = []
        # state
        state = batch["state"][:, ts]/255.0
        state = self.conv_state(state.reshape(-1, self.C, self.state_H, self.state_W))
        inputs.append(state.reshape(bs, max_t, self.args.rnn_hidden_dim).unsqueeze(2).repeat(1, 1, self.n_agents, 1))

        # observations
        obs = batch["obs"][:, ts]/255.0
        obs = self.conv_obs(obs.reshape(-1, self.C, self.obs_H, self.obs_W))
        inputs.append(obs.reshape(bs, max_t, self.n_agents, self.args.rnn_hidden_dim))

        # last actions
        if t == 0:
            inputs.append(th.zeros_like(batch["actions_onehot"][:, 0:1]).view(bs, max_t, 1, -1))
        elif isinstance(t, int):
            inputs.append(batch["actions_onehot"][:, slice(t-1, t)].view(bs, max_t, 1, -1))
        else:
            last_actions = th.cat([th.zeros_like(batch["actions_onehot"][:, 0:1]), batch["actions_onehot"][:, :-1]], dim=1)
            last_actions = last_actions.view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1)
            inputs.append(last_actions)

        inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).unsqueeze(0).expand(bs, max_t, -1, -1))

        # 将所有输入张量连接在一起，形成一个形状为 [bs, max_t, n_agents, -1] 的张量
        inputs = th.cat([x.reshape(bs, max_t, self.n_agents, -1) for x in inputs], dim=-1)
        return inputs

    def _get_input_shape(self, scheme):
        # state
        input_shape = self.args.rnn_hidden_dim
        # observation
        input_shape += self.args.rnn_hidden_dim
        # actions and last actions
        input_shape += scheme["actions_onehot"]["vshape"][0] * self.n_agents
        # # agent id
        input_shape += self.n_agents
        return input_shape


class SSD_QPCCritic(nn.Module):
    def __init__(self, scheme, args):
        super(SSD_QPCCritic, self).__init__()

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

        input_shape_ind = self._get_input_shape_ind(scheme)
        input_shape_dep = self._get_input_shape_dep(scheme)
        self.output_type = "q"

        # Set up network layers
        self.fc_ind1 = nn.Linear(input_shape_ind, 128)
        self.fc_ind2 = nn.Linear(128, 1)

        self.fc_dep1 = nn.Linear(input_shape_dep, 128)
        self.fc_dep2 = nn.Linear(128, 1)

        self.fc_Q_agent = nn.Linear(2, 1)
        self.fc_Q_tot = nn.Linear(self.n_agents, 1)
        self.fc_Q_tot_test = nn.Linear(1, 1)

    def forward(self, batch, actions, t=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        # array = batch["state"][0,0]
        # array = array.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        # image = Image.fromarray(array)
        # image.save('state.png')

        # array2 = batch["eliminated_state"][0,0]
        # array2 = array2.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        # image2 = Image.fromarray(array2)
        # image2.save('eliminated_state.png')

        inputs_ind = self._build_inputs_ind(batch, actions, t=t)
        inputs_ind = F.relu(self.fc_ind1(inputs_ind))
        inputs_ind = self.fc_ind2(inputs_ind)

        inputs_dep = self._build_inputs_dep(batch, actions, t=t)
        inputs_dep.requires_grad_(True)
        inputs_dep = F.relu(self.fc_dep1(inputs_dep))
        inputs_dep = self.fc_dep2(inputs_dep)

        Q_agent_cat = th.cat([inputs_ind, inputs_dep], dim=-1)  # cat不会导致梯度问题
        # 因为假设这些是单纯的加和所以不需要
        Q_agent = self.fc_Q_agent(Q_agent_cat)
        Q_agent = Q_agent.squeeze()
        Q_tot = self.fc_Q_tot(Q_agent)
        # 计算 Q_tot 对 Q_agent[:,:,0] 的一阶梯度

        Cooperation = []
        # 计算 Q_tot 对 Q_agent 的一阶梯度（所有智能体）
        # 这里Q_tot对于Q_agent的梯度就是一个常数，即 self.fc_Q_tot的W，与inputs_dep无关，
        # Q_agent可以对inputs_dep求梯度，但是Q_tot对Q_agent的梯度是一个常数
        # 因为本就是线性的关系，这个W的产生与inputs_dep就没有什么关系，所以不能求二阶导数
        # 要是想有inputs_dep使用二阶梯度，那么就必须是用超网络的
        # grad_Q_agent_all = th.autograd.grad(
        #     outputs=Q_tot,
        #     inputs=Q_agent,
        #     grad_outputs=th.ones_like(Q_tot),
        #     create_graph=True,
        #     retain_graph=True  # 保留计算图供后续使用
        # )[0]
        # # 提取第 i 个智能体的梯度
        # grad_Q_agent = grad_Q_agent_all[:, :, i]

        # 利用一阶梯度对 inputs_dep 求二阶偏导，求不了二阶导
        grad_Q_agent_all = th.autograd.grad(
            outputs=Q_tot,
            inputs=inputs_dep,
            grad_outputs=th.ones_like(Q_tot),
            retain_graph=True  # 根据需求决定是否保留计算图
        )[0]
        # Cooperation = Cooperation.view(bs, max_t, -1)
        return Q_tot, Q_agent, grad_Q_agent_all   
        # return grad_Q_agent_all 

    def _build_inputs_ind(self, batch, actions, t=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t+1)
        
        inputs = []
        # eliminated_state
        eliminated_state = batch["eliminated_state"][:, ts]/255.0
        eliminated_state = self.conv_state(eliminated_state.reshape(-1, self.C, self.state_H, self.state_W))
        inputs.append(eliminated_state.reshape(bs, max_t, self.args.rnn_hidden_dim).unsqueeze(2).repeat(1, 1, self.n_agents, 1))

        # observation
        obs = batch["obs"][:, ts]/255.0
        obs = self.conv_obs(obs.reshape(-1, self.C, self.obs_H, self.obs_W))
        inputs.append(obs.reshape(bs, max_t, self.n_agents, self.args.rnn_hidden_dim))

        # actions 只要自己的动作
        # actions = batch["actions_onehot"][:, ts]
        if actions.device != obs.device:
            actions = actions.to(obs.device)
        inputs.append(actions)

        # last actions 只要自己的上一个动作
        if t == 0:
            inputs.append(th.zeros_like(batch["actions_onehot"][:, 0:1]))
        elif isinstance(t, int):
            inputs.append(batch["actions_onehot"][:, slice(t-1, t)])
        else:
            last_actions = th.cat([th.zeros_like(batch["actions_onehot"][:, 0:1]), batch["actions_onehot"][:, :-1]], dim=1)
            inputs.append(last_actions)

        inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).unsqueeze(0).expand(bs, max_t, -1, -1))

        inputs = th.cat([x.reshape(bs, max_t, self.n_agents, -1) for x in inputs], dim=-1)
        return inputs


    def _build_inputs_dep(self, batch, actions, t=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t+1)
        
        inputs = []
        # eliminated_state
        eliminated_state = batch["eliminated_state"][:, ts]/255.0
        eliminated_state = self.conv_state(eliminated_state.reshape(-1, self.C, self.state_H, self.state_W))
        inputs.append(eliminated_state.reshape(bs, max_t, self.args.rnn_hidden_dim).unsqueeze(2).repeat(1, 1, self.n_agents, 1))

        # observation 得到所有的智能体观测，
        # 1.保留所有智能体的观测
        # 2.只保留自己观测到的智能体的观测和动作
        # 3.只删除自己的观测
        obs = batch["obs"][:, ts]/255.0
        obs = self.conv_obs(obs.reshape(-1, self.C, self.obs_H, self.obs_W))
        inputs.append(obs.view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1))


        # 不需要知道所有智能体的动作，只要自己的动作就可以了
        if actions.device != obs.device:
            actions = actions.to(obs.device)
        # actions (masked out by agent),每个agent都有其他所有的agent的动作，测自己动作的依赖效果，是否有合作
        # actions = actions.view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1)
        inputs.append(actions)

        # last actions 上一个动作 [bs, max_t, n_agents, n_actions]再repeat，可以用所有人的上一个动作
        # 用以评价依赖奖励
        # 需要吗，可以有，也可以不用
        if t == 0:
            inputs.append(th.zeros_like(batch["actions_onehot"][:, 0:1]).view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1))
        elif isinstance(t, int):
            inputs.append(batch["actions_onehot"][:, slice(t-1, t)].view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1))
        else:
            last_actions = th.cat([th.zeros_like(batch["actions_onehot"][:, 0:1]), batch["actions_onehot"][:, :-1]], dim=1)
            last_actions = last_actions.view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1)
            inputs.append(last_actions)

        # 生成并扩展一个单位矩阵，用于表示每个智能体的身份信息，并将其广播到整个批次和所有时间步 
        # [bs, max_t, n_agents, n_agents]
        inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).unsqueeze(0).expand(bs, max_t, -1, -1))

        # 将所有输入张量连接在一起，形成一个形状为 [bs, max_t, n_agents, -1] 的张量
        inputs = th.cat([x.reshape(bs, max_t, self.n_agents, -1) for x in inputs], dim=-1)
        return inputs

    def _get_input_shape_ind(self, scheme):
        # state
        input_shape = self.args.rnn_hidden_dim
        # observation
        input_shape += self.args.rnn_hidden_dim
        # actions and last actions
        input_shape += scheme["actions_onehot"]["vshape"][0] * 2
        # # agent id
        input_shape += self.n_agents
        return input_shape

    def _get_input_shape_dep(self, scheme):
        # state
        input_shape = self.args.rnn_hidden_dim
        # observation
        input_shape += self.args.rnn_hidden_dim * self.n_agents
        # actions and last actions
        input_shape += scheme["actions_onehot"]["vshape"][0] * self.n_agents * 2
        # # agent id
        input_shape += self.n_agents
        return input_shape
    


class SSD_QPCCritic(nn.Module):
    def __init__(self, scheme, args):
        super(SSD_QPCCritic, self).__init__()

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

        self.conv_state2 = nn.Sequential(
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

        self.conv_obs2 = nn.Sequential(
        nn.Conv2d(in_channels=self.C, out_channels=args.out_channels, \
                    kernel_size=args.kernel_size, stride=(1, 1), padding='valid'),
        nn.ReLU(), 
        nn.Flatten(), 
        nn.Linear(args.out_channels * (self.obs_H - 2) * (self.obs_W - 2), args.rnn_hidden_dim), 
        nn.ReLU()
        )


        input_shape_ind = self._get_input_shape_ind(scheme)
        input_shape_dep = self._get_input_shape_dep(scheme)
        self.output_type = "q"

        # Set up network layers
        self.fc_ind1 = nn.Linear(input_shape_ind, 128)
        self.fc_ind2 = nn.Linear(128, self.n_actions)

        self.fc_dep1 = nn.Linear(input_shape_dep, 128)
        self.fc_dep2 = nn.Linear(128, self.n_actions)

        self.fc_Q_agent1 = nn.Linear(2, 128)
        self.fc_Q_agent2 = nn.Linear(128, 1)
        self.fc_Q_tot = nn.Linear(self.n_agents, 1)
        self.fc_Q_tot_test = nn.Linear(1, 1)

    def forward(self, batch, actions, grad=False, t=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t+1)
        actions = actions.unsqueeze(-1)
        if actions.device != batch["obs"].device:
            actions = actions.to(batch["obs"].device)
        # inputs.append(actions)
        # array = batch["state"][0,0]
        # array = array.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        # image = Image.fromarray(array)
        # image.save('state.png')

        # array2 = batch["eliminated_state"][0,0]
        # array2 = array2.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        # image2 = Image.fromarray(array2)
        # image2.save('eliminated_state.png')

        inputs_ind = self._build_inputs_ind(batch, t=t)
        inputs_ind = F.relu(self.fc_ind1(inputs_ind))
        inputs_ind = self.fc_ind2(inputs_ind)
        # inputs_ind_taken = th.gather(inputs_ind, dim=3, index=actions)

        inputs_dep = self._build_inputs_dep(batch, t=t)
        inputs_dep.requires_grad_(True)
        inputs_dep = F.relu(self.fc_dep1(inputs_dep))
        inputs_dep = self.fc_dep2(inputs_dep)

        Q_agent_cat = th.cat([inputs_ind.unsqueeze(-1), inputs_dep.unsqueeze(-1)], dim=-1)  # cat不会导致梯度问题
        # 因为假设这些是单纯的加和所以不需要
        Q_agent = self.fc_Q_agent1(Q_agent_cat)
        Q_agent = self.fc_Q_agent2(Q_agent)
        Q_agent = Q_agent.squeeze(-1)
        Q_agent_taken = th.gather(Q_agent, dim=3, index=actions[:,ts]).squeeze(-1)


        Q_tot = self.fc_Q_tot(Q_agent_taken)
        # 计算 Q_tot 对 Q_agent[:,:,0] 的一阶梯度

        Cooperation = []
        # 计算 Q_tot 对 Q_agent 的一阶梯度（所有智能体）
        # 这里Q_tot对于Q_agent的梯度就是一个常数，即 self.fc_Q_tot的W，与inputs_dep无关，
        # Q_agent可以对inputs_dep求梯度，但是Q_tot对Q_agent的梯度是一个常数
        # 因为本就是线性的关系，这个W的产生与inputs_dep就没有什么关系，所以不能求二阶导数
        # 要是想有inputs_dep使用二阶梯度，那么就必须是用超网络的
        # grad_Q_agent_all = th.autograd.grad(
        #     outputs=Q_tot,
        #     inputs=Q_agent,
        #     grad_outputs=th.ones_like(Q_tot),
        #     create_graph=True,
        #     retain_graph=True  # 保留计算图供后续使用
        # )[0]
        # # 提取第 i 个智能体的梯度
        # grad_Q_agent = grad_Q_agent_all[:, :, i]

        # 利用一阶梯度对 inputs_dep 求二阶偏导，求不了二阶导
        if grad:
            grad_Q_agent_all = th.autograd.grad(
                outputs=Q_tot,
                inputs=inputs_dep,
                grad_outputs=th.ones_like(Q_tot),
                retain_graph=True  # 根据需求决定是否保留计算图
            )[0]
            # Cooperation = Cooperation.view(bs, max_t, -1)
            return Q_tot, Q_agent, grad_Q_agent_all   
            # return grad_Q_agent_all 
        else:
            return Q_tot, Q_agent, None   


    def _build_inputs_ind(self, batch, t=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t+1)
        
        inputs = []
        # eliminated_state
        eliminated_state = batch["eliminated_state"][:, ts]/255.0
        eliminated_state = self.conv_state(eliminated_state.reshape(-1, self.C, self.state_H, self.state_W))
        inputs.append(eliminated_state.reshape(bs, max_t, self.args.rnn_hidden_dim).unsqueeze(2).repeat(1, 1, self.n_agents, 1))

        # observation
        obs = batch["obs"][:, ts]/255.0
        obs = self.conv_obs(obs.reshape(-1, self.C, self.obs_H, self.obs_W))
        inputs.append(obs.reshape(bs, max_t, self.n_agents, self.args.rnn_hidden_dim))

        # actions 只要自己的动作
        # actions = batch["actions_onehot"][:, ts]
        # if actions.device != obs.device:
        #     actions = actions.to(obs.device)
        # inputs.append(actions)

        # last actions 只要自己的上一个动作
        if t == 0:
            inputs.append(th.zeros_like(batch["actions_onehot"][:, 0:1]))
        elif isinstance(t, int):
            inputs.append(batch["actions_onehot"][:, slice(t-1, t)])
        else:
            last_actions = th.cat([th.zeros_like(batch["actions_onehot"][:, 0:1]), batch["actions_onehot"][:, :-1]], dim=1)
            inputs.append(last_actions)

        inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).unsqueeze(0).expand(bs, max_t, -1, -1))

        inputs = th.cat([x.reshape(bs, max_t, self.n_agents, -1) for x in inputs], dim=-1)
        return inputs

    def _get_input_shape_ind(self, scheme):
        # state
        input_shape = self.args.rnn_hidden_dim
        # observation
        input_shape += self.args.rnn_hidden_dim
        # last actions
        input_shape += scheme["actions_onehot"]["vshape"][0]
        # # agent id
        input_shape += self.n_agents
        return input_shape

    def _build_inputs_dep(self, batch, t=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t+1)
        
        inputs = []
        # eliminated_state
        eliminated_state = batch["eliminated_state"][:, ts]/255.0
        eliminated_state = self.conv_state2(eliminated_state.reshape(-1, self.C, self.state_H, self.state_W))
        inputs.append(eliminated_state.reshape(bs, max_t, self.args.rnn_hidden_dim).unsqueeze(2).repeat(1, 1, self.n_agents, 1))

        # observation 得到所有的智能体观测，
        # 1.保留所有智能体的观测
        # 2.只保留自己观测到的智能体的观测和动作
        # 3.只删除自己的观测
        obs = batch["obs"][:, ts]/255.0
        obs = self.conv_obs2(obs.reshape(-1, self.C, self.obs_H, self.obs_W))
        inputs.append(obs.view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1))


        # # 不需要知道所有智能体的动作，只要自己的动作就可以了
        # if actions.device != obs.device:
        #     actions = actions.to(obs.device)
        # # actions (masked out by agent),每个agent都有其他所有的agent的动作，测自己动作的依赖效果，是否有合作
        # # actions = actions.view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1)
        # inputs.append(actions)

        # last actions 上一个动作 [bs, max_t, n_agents, n_actions]再repeat，可以用所有人的上一个动作
        # 用以评价依赖奖励
        # 需要吗，可以有，也可以不用
        if t == 0:
            inputs.append(th.zeros_like(batch["actions_onehot"][:, 0:1]).view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1))
        elif isinstance(t, int):
            inputs.append(batch["actions_onehot"][:, slice(t-1, t)].view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1))
        else:
            last_actions = th.cat([th.zeros_like(batch["actions_onehot"][:, 0:1]), batch["actions_onehot"][:, :-1]], dim=1)
            last_actions = last_actions.view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1)
            inputs.append(last_actions)

        # 生成并扩展一个单位矩阵，用于表示每个智能体的身份信息，并将其广播到整个批次和所有时间步 
        # [bs, max_t, n_agents, n_agents]
        inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).unsqueeze(0).expand(bs, max_t, -1, -1))

        # 将所有输入张量连接在一起，形成一个形状为 [bs, max_t, n_agents, -1] 的张量
        inputs = th.cat([x.reshape(bs, max_t, self.n_agents, -1) for x in inputs], dim=-1)
        return inputs


    def _get_input_shape_dep(self, scheme):
        # state
        input_shape = self.args.rnn_hidden_dim
        # observation
        input_shape += self.args.rnn_hidden_dim * self.n_agents
        # actions and last actions
        input_shape += scheme["actions_onehot"]["vshape"][0] * self.n_agents
        # # agent id
        input_shape += self.n_agents
        return input_shape
    


class SSD_QPC_chaowangluo_Critic(nn.Module):
    def __init__(self, scheme, args):
        super(SSD_QPC_chaowangluo_Critic, self).__init__()

        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents

        self.C = scheme["state"]["vshape"][0]

        self.state_H = scheme["state"]["vshape"][1]
        self.state_W = scheme["state"]["vshape"][2]

        self.obs_H = scheme["obs"]["vshape"][1]
        self.obs_W = scheme["obs"]["vshape"][2]
        

        self.conv_state_1 = nn.Sequential(
        nn.Conv2d(in_channels=self.C, out_channels=args.out_channels, \
                    kernel_size=args.kernel_size, stride=(1, 1), padding='valid'),
        nn.ReLU(), 
        nn.Flatten(), 
        nn.Linear(args.out_channels * (self.state_H - 2) * (self.state_W - 2), args.rnn_hidden_dim), 
        nn.ReLU()
        )
        self.conv_obs_1 = nn.Sequential(
        nn.Conv2d(in_channels=self.C, out_channels=args.out_channels, \
                    kernel_size=args.kernel_size, stride=(1, 1), padding='valid'),
        nn.ReLU(), 
        nn.Flatten(), 
        nn.Linear(args.out_channels * (self.obs_H - 2) * (self.obs_W - 2), args.rnn_hidden_dim), 
        nn.ReLU()
        )


        self.conv_state_2 = nn.Sequential(
        nn.Conv2d(in_channels=self.C, out_channels=args.out_channels, \
                    kernel_size=args.kernel_size, stride=(1, 1), padding='valid'),
        nn.ReLU(), 
        nn.Flatten(), 
        nn.Linear(args.out_channels * (self.state_H - 2) * (self.state_W - 2), args.rnn_hidden_dim), 
        nn.ReLU()
        )
        self.conv_obs_2 = nn.Sequential(
        nn.Conv2d(in_channels=self.C, out_channels=args.out_channels, \
                    kernel_size=args.kernel_size, stride=(1, 1), padding='valid'),
        nn.ReLU(), 
        nn.Flatten(), 
        nn.Linear(args.out_channels * (self.obs_H - 2) * (self.obs_W - 2), args.rnn_hidden_dim), 
        nn.ReLU()
        )


        input_shape_ind = self._get_input_shape_ind(scheme)
        input_shape_dep = self._get_input_shape_dep(scheme)
        self.output_type = "q"

        # Set up network layers
        self.fc_ind1 = nn.Linear(input_shape_ind, 128)
        self.fc_ind2 = nn.Linear(128, self.n_actions)

        self.fc_dep1 = nn.Linear(input_shape_dep, 128)
        self.fc_dep2 = nn.Linear(128, self.n_actions)

        self.hyper_w_1 = nn.Sequential(
        nn.Conv2d(in_channels=self.C, out_channels=args.out_channels, \
                    kernel_size=args.kernel_size, stride=(1, 1), padding='valid'),
        nn.ReLU(), 
        nn.Flatten(), 
        nn.Linear(args.out_channels * (self.obs_H - 2) * (self.obs_W - 2), args.rnn_hidden_dim * 2), 
        # nn.ReLU()
                                        )
        self.hyper_b_1 = nn.Sequential(
        nn.Conv2d(in_channels=self.C, out_channels=args.out_channels, \
                    kernel_size=args.kernel_size, stride=(1, 1), padding='valid'),
        nn.ReLU(), 
        nn.Flatten(), 
        nn.Linear(args.out_channels * (self.obs_H - 2) * (self.obs_W - 2), args.rnn_hidden_dim * self.n_actions), 
        # nn.ReLU()                      
        )

        self.hyper_w_2 = nn.Sequential(        
        nn.Conv2d(in_channels=self.C, out_channels=args.out_channels, \
                    kernel_size=args.kernel_size, stride=(1, 1), padding='valid'),
        nn.ReLU(), 
        nn.Flatten(), 
        nn.Linear(args.out_channels * (self.obs_H - 2) * (self.obs_W - 2), args.rnn_hidden_dim), 
        # nn.ReLU()                   
        )       
        
        # State dependent bias for hidden layer

        self.hyper_b_2 = nn.Sequential(
        nn.Conv2d(in_channels=self.C, out_channels=args.out_channels, \
                    kernel_size=args.kernel_size, stride=(1, 1), padding='valid'),
        nn.ReLU(), 
        nn.Flatten(), 
        nn.Linear(args.out_channels * (self.obs_H - 2) * (self.obs_W - 2), self.n_actions), 
        # nn.ReLU()                      
        )

        self.fc_Q_tot = nn.Linear(self.n_agents, 1)
        self.fc_Q_tot_test = nn.Linear(1, 1)

    def forward(self, batch, actions, grad=False, t=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t+1)

        actions = actions.unsqueeze(-1)
        if actions.device != batch["obs"].device:
            actions = actions.to(batch["obs"].device)
        # inputs.append(actions)
        # array = batch["state"][0,0]
        # array = array.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        # image = Image.fromarray(array)
        # image.save('state.png')

        # array2 = batch["eliminated_state"][0,800]
        # array2 = array2.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        # image2 = Image.fromarray(array2)
        # image2.save('eliminated_state.png')

        inputs_ind = self._build_inputs_ind(batch, t=t)
        inputs_ind = F.relu(self.fc_ind1(inputs_ind))
        inputs_ind = self.fc_ind2(inputs_ind)
        # inputs_ind_taken = th.gather(inputs_ind, dim=3, index=actions)

        inputs_dep = self._build_inputs_dep(batch, t=t)
        inputs_dep.requires_grad_(True)
        inputs_dep = F.relu(self.fc_dep1(inputs_dep))
        inputs_dep = self.fc_dep2(inputs_dep)

        Q_agent_cat = th.cat([inputs_ind.unsqueeze(-1), inputs_dep.unsqueeze(-1)], dim=-1)  # cat不会导致梯度问题
        # 因为假设这些是单纯的加和所以不需要

        w_1, b_1, w_2, b_2 = self._build_hyper(batch, t=t)  
        Q_agent = F.elu(th.bmm(Q_agent_cat.reshape(-1, self.n_actions,2), w_1) + b_1)
        Q_agent = (th.bmm(Q_agent, w_2) + b_2).view(bs, max_t, self.n_agents,self.n_actions)

        # Q_agent = self.fc_Q_agent1(Q_agent_cat)
        # Q_agent = self.fc_Q_agent2(Q_agent)
        # Q_agent = Q_agent.squeeze()

        # Q_agent_taken = th.gather(Q_agent, dim=3, index=actions).squeeze(-1)
        Q_agent_taken = th.gather(Q_agent, dim=3, index=actions[:,ts]).squeeze(-1)


        Q_tot = self.fc_Q_tot(Q_agent_taken)
        # 计算 Q_tot 对 Q_agent[:,:,0] 的一阶梯度

        Cooperation = []
        # 计算 Q_tot 对 Q_agent 的一阶梯度（所有智能体）
        # 这里Q_tot对于Q_agent的梯度就是一个常数，即 self.fc_Q_tot的W，与inputs_dep无关，
        # Q_agent可以对inputs_dep求梯度，但是Q_tot对Q_agent的梯度是一个常数
        # 因为本就是线性的关系，这个W的产生与inputs_dep就没有什么关系，所以不能求二阶导数
        # 要是想有inputs_dep使用二阶梯度，那么就必须是用超网络的
        # grad_Q_agent_all = th.autograd.grad(
        #     outputs=Q_tot,
        #     inputs=Q_agent,
        #     grad_outputs=th.ones_like(Q_tot),
        #     create_graph=True,
        #     retain_graph=True  # 保留计算图供后续使用
        # )[0]
        # # 提取第 i 个智能体的梯度
        # grad_Q_agent = grad_Q_agent_all[:, :, i]

        # 利用一阶梯度对 inputs_dep 求二阶偏导，求不了二阶导
        if grad:
            grad_Q_agent_all = th.autograd.grad(
                outputs=Q_tot,
                inputs=inputs_dep,
                grad_outputs=th.ones_like(Q_tot),
                retain_graph=True  # 根据需求决定是否保留计算图
            )[0]
            # Cooperation = Cooperation.view(bs, max_t, -1)
            return Q_tot, Q_agent, grad_Q_agent_all   
            # return grad_Q_agent_all 
        else:
            return Q_tot, Q_agent, None   


    def _build_inputs_ind(self, batch, t=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t+1)
        
        inputs = []
        # eliminated_state
        eliminated_state = batch["eliminated_state"][:, ts]/255.0
        eliminated_state = self.conv_state_1(eliminated_state.reshape(-1, self.C, self.state_H, self.state_W))
        inputs.append(eliminated_state.reshape(bs, max_t, self.args.rnn_hidden_dim).unsqueeze(2).repeat(1, 1, self.n_agents, 1))

        # observation
        obs = batch["obs"][:, ts]/255.0
        obs = self.conv_obs_1(obs.reshape(-1, self.C, self.obs_H, self.obs_W))
        inputs.append(obs.reshape(bs, max_t, self.n_agents, self.args.rnn_hidden_dim))

        # actions 只要自己的动作
        # actions = batch["actions_onehot"][:, ts]
        # if actions.device != obs.device:
        #     actions = actions.to(obs.device)
        # inputs.append(actions)

        # last actions 只要自己的上一个动作
        if t == 0:
            inputs.append(th.zeros_like(batch["actions_onehot"][:, 0:1]))
        elif isinstance(t, int):
            inputs.append(batch["actions_onehot"][:, slice(t-1, t)])
        else:
            last_actions = th.cat([th.zeros_like(batch["actions_onehot"][:, 0:1]), batch["actions_onehot"][:, :-1]], dim=1)
            inputs.append(last_actions)

        inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).unsqueeze(0).expand(bs, max_t, -1, -1))

        inputs = th.cat([x.reshape(bs, max_t, self.n_agents, -1) for x in inputs], dim=-1)
        return inputs

    def _get_input_shape_ind(self, scheme):
        # state
        input_shape = self.args.rnn_hidden_dim
        # observation
        input_shape += self.args.rnn_hidden_dim
        # last actions
        input_shape += scheme["actions_onehot"]["vshape"][0]
        # # agent id
        input_shape += self.n_agents
        return input_shape

    def _build_inputs_dep(self, batch, t=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t+1)
        
        inputs = []
        # eliminated_state
        eliminated_state = batch["eliminated_state"][:, ts]/255.0
        eliminated_state = self.conv_state_2(eliminated_state.reshape(-1, self.C, self.state_H, self.state_W))
        inputs.append(eliminated_state.reshape(bs, max_t, self.args.rnn_hidden_dim).unsqueeze(2).repeat(1, 1, self.n_agents, 1))

        # observation 得到所有的智能体观测，
        # 1.保留所有智能体的观测
        # 2.只保留自己观测到的智能体的观测和动作
        # 3.只删除自己的观测
        obs = batch["obs"][:, ts]/255.0
        obs = self.conv_obs_2(obs.reshape(-1, self.C, self.obs_H, self.obs_W))
        inputs.append(obs.view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1))


        # # 不需要知道所有智能体的动作，只要自己的动作就可以了
        # if actions.device != obs.device:
        #     actions = actions.to(obs.device)
        # # actions (masked out by agent),每个agent都有其他所有的agent的动作，测自己动作的依赖效果，是否有合作
        # # actions = actions.view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1)
        # inputs.append(actions)

        # last actions 上一个动作 [bs, max_t, n_agents, n_actions]再repeat，可以用所有人的上一个动作
        # 用以评价依赖奖励
        # 需要吗，可以有，也可以不用
        if t == 0:
            inputs.append(th.zeros_like(batch["actions_onehot"][:, 0:1]).view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1))
        elif isinstance(t, int):
            inputs.append(batch["actions_onehot"][:, slice(t-1, t)].view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1))
        else:
            last_actions = th.cat([th.zeros_like(batch["actions_onehot"][:, 0:1]), batch["actions_onehot"][:, :-1]], dim=1)
            last_actions = last_actions.view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1)
            inputs.append(last_actions)

        # 生成并扩展一个单位矩阵，用于表示每个智能体的身份信息，并将其广播到整个批次和所有时间步 
        # [bs, max_t, n_agents, n_agents]
        inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).unsqueeze(0).expand(bs, max_t, -1, -1))

        # 将所有输入张量连接在一起，形成一个形状为 [bs, max_t, n_agents, -1] 的张量
        inputs = th.cat([x.reshape(bs, max_t, self.n_agents, -1) for x in inputs], dim=-1)
        return inputs


    def _get_input_shape_dep(self, scheme):
        # state
        input_shape = self.args.rnn_hidden_dim
        # observation
        input_shape += self.args.rnn_hidden_dim * self.n_agents
        # actions and last actions
        input_shape += scheme["actions_onehot"]["vshape"][0] * self.n_agents
        # # agent id
        input_shape += self.n_agents
        return input_shape
    

    def _build_hyper(self, batch, t=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t+1)

        # obs = /
        # obs = self.conv_obs_2(batch["obs"][:, ts].reshape(-1, self.C, self.obs_H, self.obs_W)/255.0)
        # inputs.append(obs.view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1))


        w_1 = self.hyper_w_1(batch["obs"][:, ts].reshape(-1, self.C, self.obs_H, self.obs_W)/255.0).view(-1, 2, self.args.rnn_hidden_dim)
        b_1 = self.hyper_b_1(batch["obs"][:, ts].reshape(-1, self.C, self.obs_H, self.obs_W)/255.0).view(-1, self.n_actions, self.args.rnn_hidden_dim)

        w_2 = self.hyper_w_2(batch["obs"][:, ts].reshape(-1, self.C, self.obs_H, self.obs_W)/255.0).view(-1, self.args.rnn_hidden_dim).unsqueeze(-1)
        b_2 = self.hyper_b_2(batch["obs"][:, ts].reshape(-1, self.C, self.obs_H, self.obs_W)/255.0).view(bs*max_t*self.n_agents, self.n_actions, 1)

        return w_1, b_1, w_2, b_2
    


class SSD_QPC_chaowangluo_IDegree_Critic(nn.Module):
    def __init__(self, scheme, args):
        super(SSD_QPC_chaowangluo_IDegree_Critic, self).__init__()

        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents

        self.C = scheme["state"]["vshape"][0]

        self.state_H = scheme["state"]["vshape"][1]
        self.state_W = scheme["state"]["vshape"][2]

        self.obs_H = scheme["obs"]["vshape"][1]
        self.obs_W = scheme["obs"]["vshape"][2]
        

        self.conv_state_1 = nn.Sequential(
        nn.Conv2d(in_channels=self.C, out_channels=args.out_channels, \
                    kernel_size=args.kernel_size, stride=(1, 1), padding='valid'),
        nn.ReLU(), 
        nn.Flatten(), 
        nn.Linear(args.out_channels * (self.state_H - 2) * (self.state_W - 2), args.rnn_hidden_dim), 
        nn.ReLU()
        )
        self.conv_obs_1 = nn.Sequential(
        nn.Conv2d(in_channels=self.C, out_channels=args.out_channels, \
                    kernel_size=args.kernel_size, stride=(1, 1), padding='valid'),
        nn.ReLU(), 
        nn.Flatten(), 
        nn.Linear(args.out_channels * (self.obs_H - 2) * (self.obs_W - 2), args.rnn_hidden_dim), 
        nn.ReLU()
        )


        self.conv_state_2 = nn.Sequential(
        nn.Conv2d(in_channels=self.C, out_channels=args.out_channels, \
                    kernel_size=args.kernel_size, stride=(1, 1), padding='valid'),
        nn.ReLU(), 
        nn.Flatten(), 
        nn.Linear(args.out_channels * (self.state_H - 2) * (self.state_W - 2), args.rnn_hidden_dim), 
        nn.ReLU()
        )
        self.conv_obs_2 = nn.Sequential(
        nn.Conv2d(in_channels=self.C, out_channels=args.out_channels, \
                    kernel_size=args.kernel_size, stride=(1, 1), padding='valid'),
        nn.ReLU(), 
        nn.Flatten(), 
        nn.Linear(args.out_channels * (self.obs_H - 2) * (self.obs_W - 2), args.rnn_hidden_dim), 
        nn.ReLU()
        )


        input_shape_ind = self._get_input_shape_ind(scheme)
        input_shape_dep = self._get_input_shape_dep(scheme)
        self.output_type = "q"

        # Set up network layers
        self.fc_ind1 = nn.Linear(input_shape_ind, 128)
        self.fc_ind2 = nn.Linear(128, self.n_actions)

        self.fc_dep1 = nn.Linear(input_shape_dep, 128)
        self.fc_dep2 = nn.Linear(128, self.n_actions)

        self.hyper_w_1 = nn.Sequential(
        nn.Conv2d(in_channels=self.C, out_channels=args.out_channels, \
                    kernel_size=args.kernel_size, stride=(1, 1), padding='valid'),
        nn.ReLU(), 
        nn.Flatten(), 
        nn.Linear(args.out_channels * (self.obs_H - 2) * (self.obs_W - 2), args.rnn_hidden_dim * 2), 
        # nn.ReLU()
                                        )
        self.hyper_b_1 = nn.Sequential(
        nn.Conv2d(in_channels=self.C, out_channels=args.out_channels, \
                    kernel_size=args.kernel_size, stride=(1, 1), padding='valid'),
        nn.ReLU(), 
        nn.Flatten(), 
        nn.Linear(args.out_channels * (self.obs_H - 2) * (self.obs_W - 2), args.rnn_hidden_dim * self.n_actions), 
        # nn.ReLU()                      
        )

        self.hyper_w_2 = nn.Sequential(        
        nn.Conv2d(in_channels=self.C, out_channels=args.out_channels, \
                    kernel_size=args.kernel_size, stride=(1, 1), padding='valid'),
        nn.ReLU(), 
        nn.Flatten(), 
        nn.Linear(args.out_channels * (self.obs_H - 2) * (self.obs_W - 2), args.rnn_hidden_dim), 
        # nn.ReLU()                   
        )       
        
        # State dependent bias for hidden layer

        self.hyper_b_2 = nn.Sequential(
        nn.Conv2d(in_channels=self.C, out_channels=args.out_channels, \
                    kernel_size=args.kernel_size, stride=(1, 1), padding='valid'),
        nn.ReLU(), 
        nn.Flatten(), 
        nn.Linear(args.out_channels * (self.obs_H - 2) * (self.obs_W - 2), self.n_actions), 
        # nn.ReLU()                      
        )

        self.fc_Q_tot = nn.Linear(self.n_agents, 1)
        self.fc_Q_tot_test = nn.Linear(1, 1)

    def forward(self, batch, actions, grad=False, t=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t+1)

        actions = actions.unsqueeze(-1)
        if actions.device != batch["obs"].device:
            actions = actions.to(batch["obs"].device)
        # inputs.append(actions)
        # array = batch["state"][0,0]
        # array = array.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        # image = Image.fromarray(array)
        # image.save('state.png')

        # array2 = batch["eliminated_state"][0,800]
        # array2 = array2.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        # image2 = Image.fromarray(array2)
        # image2.save('eliminated_state.png')

        inputs_ind = self._build_inputs_ind(batch, t=t)
        inputs_ind = F.relu(self.fc_ind1(inputs_ind))
        inputs_ind = self.fc_ind2(inputs_ind)
        # inputs_ind_taken = th.gather(inputs_ind, dim=3, index=actions)

        inputs_dep = self._build_inputs_dep(batch, t=t)
        inputs_dep.requires_grad_(True)
        inputs_dep = F.relu(self.fc_dep1(inputs_dep))
        inputs_dep = self.fc_dep2(inputs_dep)

        Q_agent_cat = th.cat([inputs_ind.unsqueeze(-1), inputs_dep.unsqueeze(-1)], dim=-1)  # cat不会导致梯度问题
        # 因为假设这些是单纯的加和所以不需要

        w_1, b_1, w_2, b_2 = self._build_hyper(batch, t=t)  
        Q_agent = F.elu(th.bmm(Q_agent_cat.reshape(-1, self.n_actions,2), w_1) + b_1)
        Q_agent = (th.bmm(Q_agent, w_2) + b_2).view(bs, max_t, self.n_agents,self.n_actions)

        # Q_agent = self.fc_Q_agent1(Q_agent_cat)
        # Q_agent = self.fc_Q_agent2(Q_agent)
        # Q_agent = Q_agent.squeeze()

        # Q_agent_taken = th.gather(Q_agent, dim=3, index=actions).squeeze(-1)
        Q_agent_taken = th.gather(Q_agent, dim=3, index=actions[:,ts]).squeeze(-1)


        Q_tot = self.fc_Q_tot(Q_agent_taken)
        # 计算 Q_tot 对 Q_agent[:,:,0] 的一阶梯度

        Cooperation = []
        # 计算 Q_tot 对 Q_agent 的一阶梯度（所有智能体）
        # 这里Q_tot对于Q_agent的梯度就是一个常数，即 self.fc_Q_tot的W，与inputs_dep无关，
        # Q_agent可以对inputs_dep求梯度，但是Q_tot对Q_agent的梯度是一个常数
        # 因为本就是线性的关系，这个W的产生与inputs_dep就没有什么关系，所以不能求二阶导数
        # 要是想有inputs_dep使用二阶梯度，那么就必须是用超网络的
        # grad_Q_agent_all = th.autograd.grad(
        #     outputs=Q_tot,
        #     inputs=Q_agent,
        #     grad_outputs=th.ones_like(Q_tot),
        #     create_graph=True,
        #     retain_graph=True  # 保留计算图供后续使用
        # )[0]
        # # 提取第 i 个智能体的梯度
        # grad_Q_agent = grad_Q_agent_all[:, :, i]

        # 利用一阶梯度对 inputs_dep 求二阶偏导，求不了二阶导
        if grad:
            grad_Q_agent_all = th.autograd.grad(
                outputs=Q_tot,
                inputs=Q_agent_taken,
                grad_outputs=th.ones_like(Q_tot),
                retain_graph=True  # 根据需求决定是否保留计算图
            )[0]
            # Cooperation = Cooperation.view(bs, max_t, -1)
            return Q_tot, Q_agent, grad_Q_agent_all   
            # return grad_Q_agent_all 
        else:
            return Q_tot, Q_agent, None   


    def _build_inputs_ind(self, batch, t=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t+1)
        
        inputs = []
        # eliminated_state
        eliminated_state = batch["eliminated_state"][:, ts]/255.0
        eliminated_state = self.conv_state_1(eliminated_state.reshape(-1, self.C, self.state_H, self.state_W))
        inputs.append(eliminated_state.reshape(bs, max_t, self.args.rnn_hidden_dim).unsqueeze(2).repeat(1, 1, self.n_agents, 1))

        # observation
        obs = batch["obs"][:, ts]/255.0
        obs = self.conv_obs_1(obs.reshape(-1, self.C, self.obs_H, self.obs_W))
        inputs.append(obs.reshape(bs, max_t, self.n_agents, self.args.rnn_hidden_dim))

        # actions 只要自己的动作
        # actions = batch["actions_onehot"][:, ts]
        # if actions.device != obs.device:
        #     actions = actions.to(obs.device)
        # inputs.append(actions)

        # last actions 只要自己的上一个动作
        if t == 0:
            inputs.append(th.zeros_like(batch["actions_onehot"][:, 0:1]))
        elif isinstance(t, int):
            inputs.append(batch["actions_onehot"][:, slice(t-1, t)])
        else:
            last_actions = th.cat([th.zeros_like(batch["actions_onehot"][:, 0:1]), batch["actions_onehot"][:, :-1]], dim=1)
            inputs.append(last_actions)

        inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).unsqueeze(0).expand(bs, max_t, -1, -1))

        inputs = th.cat([x.reshape(bs, max_t, self.n_agents, -1) for x in inputs], dim=-1)
        return inputs

    def _get_input_shape_ind(self, scheme):
        # state
        input_shape = self.args.rnn_hidden_dim
        # observation
        input_shape += self.args.rnn_hidden_dim
        # last actions
        input_shape += scheme["actions_onehot"]["vshape"][0]
        # # agent id
        input_shape += self.n_agents
        return input_shape

    def _build_inputs_dep(self, batch, t=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t+1)
        
        inputs = []
        # eliminated_state
        eliminated_state = batch["eliminated_state"][:, ts]/255.0
        eliminated_state = self.conv_state_2(eliminated_state.reshape(-1, self.C, self.state_H, self.state_W))
        inputs.append(eliminated_state.reshape(bs, max_t, self.args.rnn_hidden_dim).unsqueeze(2).repeat(1, 1, self.n_agents, 1))

        # observation 得到所有的智能体观测，
        # 1.保留所有智能体的观测
        # 2.只保留自己观测到的智能体的观测和动作
        # 3.只删除自己的观测
        obs = batch["obs"][:, ts]/255.0
        obs = self.conv_obs_2(obs.reshape(-1, self.C, self.obs_H, self.obs_W))
        inputs.append(obs.view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1))


        # # 不需要知道所有智能体的动作，只要自己的动作就可以了
        # if actions.device != obs.device:
        #     actions = actions.to(obs.device)
        # # actions (masked out by agent),每个agent都有其他所有的agent的动作，测自己动作的依赖效果，是否有合作
        # # actions = actions.view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1)
        # inputs.append(actions)

        # last actions 上一个动作 [bs, max_t, n_agents, n_actions]再repeat，可以用所有人的上一个动作
        # 用以评价依赖奖励
        # 需要吗，可以有，也可以不用
        if t == 0:
            inputs.append(th.zeros_like(batch["actions_onehot"][:, 0:1]).view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1))
        elif isinstance(t, int):
            inputs.append(batch["actions_onehot"][:, slice(t-1, t)].view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1))
        else:
            last_actions = th.cat([th.zeros_like(batch["actions_onehot"][:, 0:1]), batch["actions_onehot"][:, :-1]], dim=1)
            last_actions = last_actions.view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1)
            inputs.append(last_actions)

        # 生成并扩展一个单位矩阵，用于表示每个智能体的身份信息，并将其广播到整个批次和所有时间步 
        # [bs, max_t, n_agents, n_agents]
        inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).unsqueeze(0).expand(bs, max_t, -1, -1))

        # 将所有输入张量连接在一起，形成一个形状为 [bs, max_t, n_agents, -1] 的张量
        inputs = th.cat([x.reshape(bs, max_t, self.n_agents, -1) for x in inputs], dim=-1)
        return inputs


    def _get_input_shape_dep(self, scheme):
        # state
        input_shape = self.args.rnn_hidden_dim
        # observation
        input_shape += self.args.rnn_hidden_dim * self.n_agents
        # actions and last actions
        input_shape += scheme["actions_onehot"]["vshape"][0] * self.n_agents
        # # agent id
        input_shape += self.n_agents
        return input_shape
    

    def _build_hyper(self, batch, t=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t+1)

        # obs = /
        # obs = self.conv_obs_2(batch["obs"][:, ts].reshape(-1, self.C, self.obs_H, self.obs_W)/255.0)
        # inputs.append(obs.view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1))


        w_1 = self.hyper_w_1(batch["obs"][:, ts].reshape(-1, self.C, self.obs_H, self.obs_W)/255.0).view(-1, 2, self.args.rnn_hidden_dim)
        b_1 = self.hyper_b_1(batch["obs"][:, ts].reshape(-1, self.C, self.obs_H, self.obs_W)/255.0).view(-1, self.n_actions, self.args.rnn_hidden_dim)

        w_2 = self.hyper_w_2(batch["obs"][:, ts].reshape(-1, self.C, self.obs_H, self.obs_W)/255.0).view(-1, self.args.rnn_hidden_dim).unsqueeze(-1)
        b_2 = self.hyper_b_2(batch["obs"][:, ts].reshape(-1, self.C, self.obs_H, self.obs_W)/255.0).view(bs*max_t*self.n_agents, self.n_actions, 1)

        return w_1, b_1, w_2, b_2



class MAT_QPC_chaowangluo_Critic(nn.Module):
    def __init__(self, scheme, args):
        super(MAT_QPC_chaowangluo_Critic, self).__init__()

        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents

        input_shape_ind = self._get_input_shape_ind(scheme)
        input_shape_dep = self._get_input_shape_dep(scheme)
        self.output_type = "q"

        # Set up network layers
        self.fc_ind1 = nn.Linear(input_shape_ind, 64)
        self.fc_ind2 = nn.Linear(64, self.n_actions)

        self.fc_dep1 = nn.Linear(input_shape_dep, 64)
        self.fc_dep2 = nn.Linear(64, self.n_actions)

        self.hyper_w_1 = nn.Sequential(
        nn.Linear(input_shape_ind, args.rnn_hidden_dim), 
        nn.ReLU(), 
        nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim * 2), 
                                        )
        self.hyper_b_1 = nn.Sequential(
        nn.Linear(input_shape_ind, args.rnn_hidden_dim), 
        nn.ReLU(),                      
        nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim * self.n_actions), 
        )

        self.hyper_w_2 = nn.Sequential(
        nn.Linear(input_shape_ind, args.rnn_hidden_dim), 
        nn.ReLU(),                   
        nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim), )       
        
        # State dependent bias for hidden layer

        self.hyper_b_2 = nn.Sequential(
        nn.Linear(input_shape_ind, args.rnn_hidden_dim), 
        nn.ReLU(),
        nn.Linear(args.rnn_hidden_dim, self.n_actions), 
                              )

        self.fc_Q_tot = nn.Linear(self.n_agents, 1)
        self.fc_Q_tot_test = nn.Linear(1, 1)

    def forward(self, batch, actions, grad=False, t=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t+1)

        actions = actions.unsqueeze(-1)
        if actions.device != batch["obs"].device:
            actions = actions.to(batch["obs"].device)

        inputs_ind_basic = self._build_inputs_ind(batch, t=t)
        inputs_ind = F.relu(self.fc_ind1(inputs_ind_basic))
        inputs_ind = self.fc_ind2(inputs_ind)
        # inputs_ind_taken = th.gather(inputs_ind, dim=3, index=actions)

        inputs_dep = self._build_inputs_dep(batch, t=t)
        inputs_dep.requires_grad_(True)
        inputs_dep = F.relu(self.fc_dep1(inputs_dep))
        inputs_dep = self.fc_dep2(inputs_dep)

        Q_agent_cat = th.cat([inputs_ind.unsqueeze(-1), inputs_dep.unsqueeze(-1)], dim=-1)  # cat不会导致梯度问题
        # 因为假设这些是单纯的加和所以不需要

        w_1, b_1, w_2, b_2 = self._build_hyper(batch, inputs_ind_basic, t=t)  
        Q_agent = F.elu(th.bmm(Q_agent_cat.reshape(-1, self.n_actions,2), w_1) + b_1)
        # Q_agent = (th.bmm(Q_agent, w_2) + b_2).view(bs, max_t, self.n_agents,self.n_actions)
        Q_agent = (th.bmm(Q_agent, w_2) + b_2).view(bs, max_t, self.n_agents,self.n_actions)

        Q_agent_taken = th.gather(Q_agent, dim=3, index=actions[:,ts]).squeeze(-1)


        Q_tot = self.fc_Q_tot(Q_agent_taken)

        if grad:
            grad_Q_agent_all = th.autograd.grad(
                outputs=Q_tot,
                inputs=inputs_dep,
                grad_outputs=th.ones_like(Q_tot),
                retain_graph=True  # 根据需求决定是否保留计算图
            )[0]
            # Cooperation = Cooperation.view(bs, max_t, -1)
            return Q_tot, Q_agent, grad_Q_agent_all   
            # return grad_Q_agent_all 
        else:
            return Q_tot, Q_agent, None   


    def _build_inputs_ind(self, batch, t=None):
        ts = slice(None) if t is None else slice(t, t+1)
        inputs = th.cat([batch["obs"][:, ts], batch["actions_onehot"][:, ts]], dim=-1)
        return inputs

    def _get_input_shape_ind(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        input_shape += scheme["actions_onehot"]["vshape"][0]
        return input_shape

    def _build_inputs_dep(self, batch, t=None):
        ts = slice(None) if t is None else slice(t, t+1)
        bs, max_t, _ = batch["state"][:, ts].shape
        state = batch["state"][:, ts].reshape(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1)
        actions = batch["actions_onehot"][:, ts]
        inputs = th.cat([state, actions], dim=-1)
        return inputs
        # state = np.array([
                        #     [[1, 0], [1, 0]],  # 环境1：智能体0和1的初始状态
                        #     [[0, 1], [0, 1]],  # 环境2：智能体0和1的执行后状态
                        #     ...                 # 共64个环境
                        # ])


    def _get_input_shape_dep(self, scheme):
        input_shape = scheme["state"]["vshape"]
        input_shape += scheme["actions_onehot"]["vshape"][0]
        return input_shape
    

    def _build_hyper(self, batch, inputs, t=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t+1)

        w_1 = self.hyper_w_1(inputs).view(-1, 2, self.args.rnn_hidden_dim)
        b_1 = self.hyper_b_1(inputs).view(-1, self.n_actions, self.args.rnn_hidden_dim)

        w_2 = self.hyper_w_2(inputs).view(-1, self.args.rnn_hidden_dim).unsqueeze(-1)
        b_2 = self.hyper_b_2(inputs).view(bs*max_t*self.n_agents, self.n_actions, 1)
        # b_2 = self.hyper_b_2(inputs).view(bs*self.n_agents, self.n_actions, 1)

        return w_1, b_1, w_2, b_2