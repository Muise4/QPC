import numpy as np
from envs.SSD.env.pettingzoo_env import env, raw_env
from envs.multiagentenv import MultiAgentEnv
from PIL import Image


class SSD_EnvWrapper(MultiAgentEnv):
    def __init__(self, **kwargs):
        self.env = env(max_cycles = kwargs["episode_limit"], **kwargs)
            # env继承了AECEnv类，cleanup是作为一个参数实例传进去的，写了新的方法
            # unwrapped, observation_spaces, action_spaces, observation_space, action_space
            # seed, reset, observe, state, add_new_agent, step, last, render, close
        self.args = kwargs
        self.env.reset()  # 确保在访问 agents 之前先调用 reset
        self.agents = self.env.agents
        self.n_agents = len(self.agents)
        self.episode_limit = kwargs["episode_limit"]
        self.ssd_env = self.env.unwrapped.ssd_env
        """
        # 在这两个文件中，self.world_map 中的智能体相关信息是通过 MapEnv
        # 类中的 get_map_with_agents 方法加入的。这个方法会将智能体的位置更新到 self.world_map 中。
        # Map without agents or beams!
        # self.world_map 是一个二维数组，表示环境的基本状态。它包含了环境中的静态元素，如墙壁、资源、清理区域等。
        # 在 step 方法中，self.world_map 不会直接更新光束等动态信息。
        # map_with_agents：
        # map_with_agents 是在 self.world_map 的基础上生成的，包含了智能体和光束等动态信息。
        # 通过 get_map_with_agents 方法生成，方法会将智能体的位置和光束的位置添加到 self.world_map 中。"""
        
        self.world_map = self.ssd_env.world_map
        self.world_map_color = self.ssd_env.world_map_color

        self.world_map_with_all = self.ssd_env.world_map_with_all
        self.world_map_with_all_color = self.ssd_env.world_map_with_all_color

        self.world_map_without_agents_color = self.ssd_env.world_map_without_agents_color


        self.H = self.world_map.shape[0]
        self.W = self.world_map.shape[1]
        self.C = self.world_map_color.shape[2]
        self.obs_H = self.env.observation_space(self.agents[0])['curr_obs'].shape[0]
        self.obs_W = self.env.observation_space(self.agents[0])['curr_obs'].shape[1]
        # (height, width, channels) tensorflow的默认维度顺序
        # (channels, height, width) torch的默认维度顺序
        self.reset()

    def reset(self):
        self.env.reset()
        self.dones = self.env.dones
        self.steps = self.env.unwrapped.num_cycles

        # self.dones = {agent: False for agent in self.agents}
        obs = self.get_obs()
        return obs

    def step(self, actions):
        # 这里的step是from_parallel_wrapper所重构的step
        actions_dict = {agent: action for agent, action in zip(self.agents, actions)}
        # 执行所有智能体的动作，并更新环境状态
        for agent in self.agents:
            if not self.dones[agent]:
                self.env.step(actions_dict[agent])
        # self.env.step(actions_dict)

        # print()
        # 问题在于world_map和world_map_color中有没有智能体的位置信息？
        # 另一方面观测的(15, 15)中有没有智能体的位置信息？
        self.world_map_with_all = self.ssd_env.world_map_with_all
        # self.world_map_with_all_color有所有agent和光线
        self.world_map_with_all_color = self.ssd_env.world_map_with_all_color
        # self.world_map_without_agents_color没有agent和光线
        self.world_map_without_agents_color = self.ssd_env.world_map_without_agents_color

        # 需要调换位置！输出与换位置前的get_obs()一样
        # next_obs = [self.env.env.env._observations[agent] for agent in self.agents] 
        # next_obs = self.get_obs()
        '''全局奖励,等于所有局部奖励的和,此处先用这个实现qmix''' 
        single_rewards = [self.env.rewards[agent] for agent in self.agents]
        total_reward = sum(single_rewards)
        dones = [self.env.dones[agent] for agent in self.agents]
        terminated = all(dones)
        # infos = {"total_rewards": total_rewards}
        # infos = {"episode_limit": all(dones.values())}
        infos = {"episode_limit": terminated}

        self.steps = self.env.unwrapped.num_cycles
        # if self.steps >= self.episode_limit:
        #     print(dones)
        # if self.steps >= self.args["episode_limit"] and not terminated:
        #     terminated = True
        #     infos["episode_limit"] = getattr(self, "truncate_episodes", True)  # by default True
        # else:
        #     infos["episode_limit"] = False

        # infos = {agent: self.env.infos[agent] for agent in self.agents}
        # self.world_map_color一直就没变过，一直是没有智能体的初始状态
        # self.world_map_without_agents_color不知道为什么这里的world_map_without_agents_color有智能体在，只是没有光束
        # image = Image.fromarray(self.world_map_color)
        # image.save('world_map_wrapper.png')

        # image1 = Image.fromarray(self.world_map_without_agents_color )
        # image1.save('world_map_without_agents_color_wrapper.png')

        # image3 = Image.fromarray(self.world_map_with_all_color)
        # # world_map_with_all_color确实全都有
        # image3.save('world_map_all_wrapper.png')

        return single_rewards, total_reward, terminated, infos
        # obss, rews, dones, infos = self.env.step(actions_dict)
        # obss = [np.transpose(obss['curr_obs'], (2, 0, 1)) for agent in self.agents]
        # # 更新 world_map


        # return obss, rews, dones, infos
        # # self.env.step(actions_dict)
        # # next_obs = self.get_obs()
        # # rewards = [self.env.rewards[agent] for agent in self.agents]
        # # dones = [self.env.dones[agent] for agent in self.agents]
        # # infos = {agent: {} for agent in self.agents}
        # # return next_obs, rewards, dones, infos

    def get_obs(self):
        # 转换为 (channels, height, width) 格式
        return [np.transpose(self.env.observe(agent)['curr_obs'], (2, 0, 1)) for agent in self.agents]
    
    def get_obs_other_agent_actions(self):
        return [self.env.observe(agent)['other_agent_actions'] for agent in self.agents]
    
    def get_obs_visible_agents(self):
        return [self.env.observe(agent)['visible_agents'] for agent in self.agents]
    
    def get_obs_prev_visible_agents(self):
        return [self.env.observe(agent)['prev_visible_agents'] for agent in self.agents]

    def get_obs_size(self):
        # 观察空间的大小，并且转化成torch。
        return (self.C, self.obs_H, self.obs_W)
        # return self.env.observation_space(self.agents[0]).shape[0]

    def get_obs_other_agent_actions(self):
        # 其他代理的动作。
        return self.env.observation_space(self.agents[0])['other_agent_actions'].shape

    def get_obs_prev_visible_agents(self):
        # 先前时间步中可见的代理。
        return self.env.observation_space(self.agents[0])['prev_visible_agents'].shape

    def get_obs_visible_agents(self):
        # 当前时间步中可见的代理。
        return self.env.observation_space(self.agents[0])['visible_agents'].shape

    def get_state_size(self):
        return (self.C, self.H, self.W)
    
    def get_state(self):
        # self.world_map_color是一个大图，其中心部分是self.world_map。
        # 周边是为了观测进行的填充
        start_x = (self.world_map_with_all_color.shape[0] - self.world_map.shape[0]) // 2
        start_y = (self.world_map_with_all_color.shape[1] - self.world_map.shape[1]) // 2
        # 裁剪中心部分
        cropped_world_map_with_all_color = self.world_map_with_all_color[start_x:start_x + self.world_map.shape[0],
                                                       start_y:start_y + self.world_map.shape[1], :]
        
        # 转换为 (channels, height, width) 格式
        torch_cropped_world_map_with_all_color = np.transpose(cropped_world_map_with_all_color, (2, 0, 1))
        
        return torch_cropped_world_map_with_all_color


    def get_eliminated_state(self):
        # self.world_map_color是一个大图，其中心部分是self.world_map。
        # 周边是为了观测进行的填充
        start_x = (self.world_map_without_agents_color.shape[0] - self.world_map.shape[0]) // 2
        start_y = (self.world_map_without_agents_color.shape[1] - self.world_map.shape[1]) // 2
        # 裁剪中心部分
        cropped_world_map_without_agents_color = self.world_map_without_agents_color[start_x:start_x + self.world_map.shape[0],
                                                       start_y:start_y + self.world_map.shape[1], :]
        
        # 转换为 (channels, height, width) 格式
        torch_cropped_world_map_without_agents_color = np.transpose(cropped_world_map_without_agents_color, (2, 0, 1))
        
        return torch_cropped_world_map_without_agents_color

    def get_state_color_size(self):
        return self.world_map_color.shape

    def get_avail_actions(self):
        """获取每个代理可用动作的列表。
        对于每个代理，返回一个全为 1 的数组，数组的长度等于该代理的动作空间大小。
        """
        return [np.ones(self.env.action_space(agent).n) for agent in self.agents]

    def get_total_actions(self):
        """
        获取单个代理的动作空间大小。
        只返回第一个代理的动作空间大小。
        """
        return self.env.action_space(self.agents[0]).n

    def render(self):
        self.env.render(mode='rgb_array')
       # self.env.render()

    def close(self):
        self.env.close()

    def get_env_info(self):
        env_info = {"state_shape": self.get_state_size(),
                    "state_color_shape": self.get_state_color_size(),
                    "obs_shape": self.get_obs_size(),
                    "obs_other_agent_actions_shape": self.get_obs_other_agent_actions(),
                    "obs_prev_visible_agents_shape": self.get_obs_prev_visible_agents(),
                    "obs_visible_agents_shape": self.get_obs_visible_agents(),
                    "avail_actions": self.get_avail_actions(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.n_agents,
                    "episode_limit": self.episode_limit}
        return env_info


    def get_stats(self):
        return {}

    # TODO: Temp hack
    def get_agg_stats(self, stats):
        return {}




# class SSD_EnvWrapper(MultiAgentEnv):
#     def __init__(self, **kwargs):
#         self.env = raw_env(max_cycles = kwargs["episode_limit"], **kwargs)
#         self.env.reset()  # 确保在访问 agents 之前先调用 reset
#         self.agents = self.env.agents
#         self.n_agents = len(self.agents)
#         self.episode_limit = kwargs["episode_limit"]
#         self.world_map = self.env.ssd_env.world_map
#         self.world_map_color = self.env.env.env.env.ssd_env.world_map_color
#         self.H = self.world_map.shape[0]
#         self.W = self.world_map.shape[1]
#         self.C = self.world_map_color.shape[2]
#         self.obs_H = self.env.observation_space(self.agents[0])['curr_obs'].shape[0]
#         self.obs_W = self.env.observation_space(self.agents[0])['curr_obs'].shape[1]
#         # (height, width, channels) tensorflow的默认维度顺序
#         # (channels, height, width) torch的默认维度顺序
#         self.reset()

#     def reset(self):
#         self.env.reset()
#         self.dones = {agent: False for agent in self.agents}
#         obs = self.get_obs()
#         return obs

#     def step(self, actions):
#         actions_dict = {agent: action for agent, action in zip(self.agents, actions)}
#         obss, rews, dones, infos = self.env.env.env.env.step(actions_dict)
#         return obss, rews, dones, infos
        
#         # self.env.step(actions_dict)
#         # next_obs = self.get_obs()
#         # rewards = [self.env.rewards[agent] for agent in self.agents]
#         # dones = [self.env.dones[agent] for agent in self.agents]
#         # infos = {agent: {} for agent in self.agents}
#         # return next_obs, rewards, dones, infos

#     def get_obs(self):
#         # 转换为 (channels, height, width) 格式
#         return [np.transpose(self.env.observe(agent)['curr_obs'], (2, 0, 1)) for agent in self.agents]
    
#     def get_obs_other_agent_actions(self):
#         return [self.env.observe(agent)['other_agent_actions'] for agent in self.agents]
    
#     def get_obs_visible_agents(self):
#         return [self.env.observe(agent)['visible_agents'] for agent in self.agents]
    
#     def get_obs_prev_visible_agents(self):
#         return [self.env.observe(agent)['prev_visible_agents'] for agent in self.agents]

#     def get_obs_size(self):
#         # 观察空间的大小，并且转化成torch。
#         return (self.C, self.obs_H, self.obs_W)
#         # return self.env.observation_space(self.agents[0]).shape[0]

#     def get_obs_other_agent_actions(self):
#         # 其他代理的动作。
#         return self.env.observation_space(self.agents[0])['other_agent_actions'].shape

#     def get_obs_prev_visible_agents(self):
#         # 先前时间步中可见的代理。
#         return self.env.observation_space(self.agents[0])['prev_visible_agents'].shape

#     def get_obs_visible_agents(self):
#         # 当前时间步中可见的代理。
#         return self.env.observation_space(self.agents[0])['visible_agents'].shape

#     def get_state_size(self):
#         return (self.C, self.H, self.W)
    
#     def get_state(self):
#         # self.world_map_color是一个大图，其中心部分是self.world_map。
#         # 周边是为了观测进行的填充
#         start_x = (self.world_map_color.shape[0] - self.world_map.shape[0]) // 2
#         start_y = (self.world_map_color.shape[1] - self.world_map.shape[1]) // 2
#         # 裁剪中心部分
#         cropped_world_map_color = self.world_map_color[start_x:start_x + self.world_map.shape[0],
#                                                        start_y:start_y + self.world_map.shape[1], :]
        
#         # 转换为 (channels, height, width) 格式
#         torch_cropped_world_map_color = np.transpose(cropped_world_map_color, (2, 0, 1))
        
#         return torch_cropped_world_map_color

#     def get_state_color_size(self):
#         return self.world_map_color.shape

#     def get_avail_actions(self):
#         """获取每个代理可用动作的列表。
#         对于每个代理，返回一个全为 1 的数组，数组的长度等于该代理的动作空间大小。
#         """
#         return [np.ones(self.env.action_space(agent).n) for agent in self.agents]

#     def get_total_actions(self):
#         """
#         获取单个代理的动作空间大小。
#         只返回第一个代理的动作空间大小。
#         """
#         return self.env.action_space(self.agents[0]).n

#     def render(self):
#         self.env.render(mode='rgb_array')
#        # self.env.render()

#     def close(self):
#         self.env.close()

#     def get_env_info(self):
#         env_info = {"state_shape": self.get_state_size(),
#                     "state_color_shape": self.get_state_color_size(),
#                     "obs_shape": self.get_obs_size(),
#                     "obs_other_agent_actions_shape": self.get_obs_other_agent_actions(),
#                     "obs_prev_visible_agents_shape": self.get_obs_prev_visible_agents(),
#                     "obs_visible_agents_shape": self.get_obs_visible_agents(),
#                     "avail_actions": self.get_avail_actions(),
#                     "n_actions": self.get_total_actions(),
#                     "n_agents": self.n_agents,
#                     "episode_limit": self.episode_limit}
#         return env_info
