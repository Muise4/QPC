import numpy as np
from pettingzoo.mpe import simple_tag_v2

class PettingZooEnvWrapper():
    def __init__(self):
        self.env = simple_tag_v2.parallel_env()
        self.agents = self.env.agents
        self.n_agents = len(self.agents)
        self.reset()

    def reset(self):
        self.env.reset()
        self.dones = {agent: False for agent in self.agents}
        obs = self.get_obs()
        return obs

    def step(self, actions):
        actions_dict = {agent: action for agent, action in zip(self.agents, actions)}
        self.env.step(actions_dict)
        next_obs = self.get_obs()
        rewards = [self.env.rewards[agent] for agent in self.agents]
        dones = [self.env.dones[agent] for agent in self.agents]
        infos = {agent: {} for agent in self.agents}
        return next_obs, rewards, dones, infos

    def get_obs(self):
        return [self.env.observe(agent) for agent in self.agents]

    def get_obs_size(self):
        return self.env.observation_space(self.agents[0]).shape[0]

    def get_state_size(self):
        return self.get_obs_size() * self.n_agents

    def get_avail_actions(self):
        return [np.ones(self.env.action_space(agent).n) for agent in self.agents]

    def get_total_actions(self):
        return self.env.action_space(self.agents[0]).n

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()