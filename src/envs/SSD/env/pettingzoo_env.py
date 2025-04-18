from functools import lru_cache
from gym.utils import EzPickle
from pettingzoo.utils import wrappers
from pettingzoo.utils.conversions import from_parallel_wrapper
from pettingzoo.utils.env import ParallelEnv

from envs.SSD.env.env_creator import get_env_creator

MAX_CYCLES = 1000

class ssd_parallel_env(ParallelEnv):
    def __init__(self, env, max_cycles):
        self.ssd_env = env
        self.max_cycles = max_cycles
        self.possible_agents = list(self.ssd_env.agents.keys())
        self.ssd_env.reset()
        self.observation_space = lru_cache(maxsize=None)(lambda agent_id: env.observation_space)
        self.action_space = lru_cache(maxsize=None)(lambda agent_id: env.action_space)
        self.observation_spaces = {agent: env.observation_space for agent in self.possible_agents}
        self.action_spaces = {agent: env.action_space for agent in self.possible_agents}

    def reset(self):
        self.agents = self.possible_agents[:]
        self.num_cycles = 0
        self.dones = {agent: False for agent in self.agents}
        return self.ssd_env.reset()

    def seed(self, seed=None):
        return self.ssd_env.seed(seed)

    def render(self, mode="rgb_array"):
        return self.ssd_env.render(mode=mode)

    def close(self):
        self.ssd_env.close()

    def step(self, actions):
        obss, rews, self.dones, infos = self.ssd_env.step(actions)
        del self.dones["__all__"]
        self.num_cycles += 1
        if self.num_cycles >= self.max_cycles:
            # 大于最大周期数，就结束，即terminated
            self.dones = {agent: True for agent in self.agents}
        self.agents = [agent for agent in self.agents if not self.dones[agent]]
        return obss, rews, self.dones, infos


class _parallel_env(ssd_parallel_env, EzPickle):
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, max_cycles, **ssd_args):
        EzPickle.__init__(self, max_cycles, **ssd_args)
        env = get_env_creator(**ssd_args)()
        # super().__init__：调用父类的初始化方法
        super().__init__(env, max_cycles)

def parallel_env(max_cycles=MAX_CYCLES, **ssd_args):
    return _parallel_env(max_cycles, **ssd_args)


def raw_env(max_cycles=MAX_CYCLES, **ssd_args):
    # 继承了AECEnv类，cleanup是作为一个参数实例传进去的，写了新的方法
    # unwrapped, observation_spaces, action_spaces, observation_space, action_space
    # seed, reset, observe, state, add_new_agent, step, last, render, close
    return from_parallel_wrapper(parallel_env(max_cycles, **ssd_args))


def env(max_cycles=MAX_CYCLES, **ssd_args):
    aec_env = raw_env(max_cycles, **ssd_args)
    # 没有重写，调用的全都是父类方法，只是进行了检测
    aec_env = wrappers.AssertOutOfBoundsWrapper(aec_env)
    # 没有重写，调用的全都是父类方法，只是进行了检测
    aec_env = wrappers.OrderEnforcingWrapper(aec_env)
    return aec_env



