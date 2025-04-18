from envs.multiagentenv import MultiAgentEnv
from .PettingZooEnvWrapper import PettingZooEnvWrapper
import numpy as np

class PettingZooEnv(MultiAgentEnv):
    def __init__(self, **kwargs):
        self.env = PettingZooEnvWrapper()

    def reset(self):
        return self.env.reset()

    def step(self, actions):
        return self.env.step(actions)

    def get_obs(self):
        return self.env.get_obs()

    def get_obs_size(self):
        return self.env.get_obs_size()

    def get_state(self):
        return np.concatenate(self.env.get_obs())

    def get_state_size(self):
        return self.env.get_state_size()

    def get_avail_actions(self):
        return self.env.get_avail_actions()

    def get_total_actions(self):
        return self.env.get_total_actions()

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()
