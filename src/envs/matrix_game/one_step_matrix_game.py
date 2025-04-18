from envs.multiagentenv import MultiAgentEnv
from utils.dict2namedtuple import convert
import numpy as np
import torch as th

# this non-monotonic matrix can be solved by qmix
# payoff_values = [[12, -0.1, -0.1],
#                     [-0.1, 0, 0],
#                     [-0.1, 0, 0]]

# payoff_values = [[12, -12, -12],
#                     [-12, 0, 0],
#                     [-12, 0, 0]]

# payoff_values = [[12, 0, 10],
#                     [0, 0, 10],
#                     [10, 10, 10]]

PAYOFFS = {
    "prisoners_dilemma": [[(10,10), (0,15)], [(15,0), (5,5)]], 
    "stag_hunt": [[(15,15), (0,10)], [(10,0), (5,5)]], 
    "chicken_game": [[(10,10), (5,15)], [(15,5), (0,0)]]
}

class OneStepMatrixGame(MultiAgentEnv):
    def __init__(self, batch_size=None, **kwargs):
        # Define the agents
        self.args = kwargs
        self.n_agents = self.args["num_agents"]
        self.map_name = self.args["map_name"]
        # Define the internal state
        self.steps = 0
        self.payoff_values = PAYOFFS[self.map_name]
        self.n_actions = len(self.payoff_values[0])
        self.episode_limit = 1


    def reset(self):
        """ Returns initial observations and states"""
        self.steps = 0
        return self.get_obs(), self.get_state()

    def step(self, actions):
        """ Returns reward, terminated, info """
        single_rewards = self.payoff_values[actions[0]][actions[1]]
        total_reward = sum(single_rewards)
        self.steps = 1
        terminated = True

        info = {}
        return single_rewards, total_reward, terminated, info

    def get_obs(self):
        """返回包含智能体ID和时间步的观测"""
        obs = []
        for agent_id in range(self.n_agents):
            # 基础观测：智能体ID + 时间步标志 + 预留位
            agent_obs = np.zeros(3)  # [agent_id, is_initial_step, is_terminated, reserved]
            agent_obs[0] = agent_id  # 智能体ID (0或1)
            agent_obs[1 + self.steps] = 1  # 时间步标志
            obs.append(agent_obs)
        return obs
        # """ Returns all agent observations in a list """
        # # for i in range(self.n_agents):
        # #     if self.steps == 0:
        # #         return [self.get_obs_agent(i) for i in range(self.n_agents)]
        # one_hot_step = np.zeros(2)
        # one_hot_step[self.steps] = 1
        # return [np.copy(one_hot_step) for _ in range(self.n_agents)]

    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        return self.get_obs()[agent_id]

    def get_obs_size(self):
        """ Returns the shape of the observation """
        return len(self.get_obs_agent(0))

    def get_state(self):
        one_hot_step = np.zeros(2)
        one_hot_step[self.steps] = 1
        return one_hot_step

    def get_state_size(self):
        """ Returns the shape of the state"""
        return 2

    def get_avail_actions(self):
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return avail_actions

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        return np.ones(self.n_actions)

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        return self.n_actions

    def get_stats(self):
        return None

    def render(self):
        raise NotImplementedError

    def close(self):
        pass

    def seed(self):
        raise NotImplementedError
    


payoff_values = [[12, -0.1, -0.1],
                    [-0.1, 0, 0],
                    [-0.1, 0, 0]]

# for mixer methods
def print_matrix_status(batch, mixer, mac_out):
    batch_size = batch.batch_size
    matrix_size = len(payoff_values)
    results = th.zeros((matrix_size, matrix_size))       
        
    with th.no_grad():
        for i in range(results.shape[0]):
            for j in range(results.shape[1]):
                actions = th.LongTensor([[[[i], [j]]]]).to(device=mac_out.device).repeat(batch_size, 1, 1, 1)
                if len(mac_out.size()) == 5: # n qvals
                    actions = actions.unsqueeze(-1).repeat(1, 1, 1, 1, mac_out.size(-1)) # b, t, a, actions, n
                qvals = th.gather(mac_out[:batch_size, 0:1], dim=3, index=actions).squeeze(3)
                
                global_q = mixer(qvals, batch["state"][:batch_size, 0:1]).mean()
                results[i][j] = global_q.item()
                
    th.set_printoptions(1, sci_mode=False)
    print(results)
    if len(mac_out.size()) == 5:
        mac_out = mac_out.mean(-1)
    print(mac_out.mean(dim=(0, 1)).detach().cpu())
    th.set_printoptions(4)