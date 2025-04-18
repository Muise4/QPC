from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
from multiprocessing import Pipe, Process
import numpy as np
import torch as th
import warnings


# Based (very) heavily on SubprocVecEnv from OpenAI Baselines
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/subproc_vec_env.py
class SSD_ParallelRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run

        # Make subprocesses for the envs
        self.parent_conns, self.worker_conns = zip(*[Pipe() for _ in range(self.batch_size)])
        env_fn = env_REGISTRY[self.args.env]
        self.ps = [Process(target=env_worker, args=(worker_conn, CloudpickleWrapper(partial(env_fn, **self.args.env_args))))
                            for worker_conn in self.worker_conns]

        for p in self.ps:
            p.daemon = True
            p.start()

        self.parent_conns[0].send(("get_env_info", None))
        self.env_info = self.parent_conns[0].recv()
        self.episode_limit = self.env_info["episode_limit"]

        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        self.log_train_stats_t = -100000

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac
        self.scheme = scheme
        self.groups = groups
        self.preprocess = preprocess

    def get_env_info(self):
        return self.env_info

    def save_replay(self):
        pass

    def close_env(self):
        for parent_conn in self.parent_conns:
            parent_conn.send(("close", None))

    def reset(self):
        self.batch = self.new_batch()

        # Reset the envs
        for parent_conn in self.parent_conns:
            parent_conn.send(("reset", None))

        pre_transition_data = {
            "state": [],
            "eliminated_state": [],
            "avail_actions": [],
            "obs": []
        }
        # Get the obs, state and avail_actions back
        for parent_conn in self.parent_conns:
            data = parent_conn.recv()
            pre_transition_data["state"].append(data["state"])
            pre_transition_data["eliminated_state"].append(data["eliminated_state"])
            pre_transition_data["avail_actions"].append(data["avail_actions"])
            pre_transition_data["obs"].append(data["obs"])

        self.batch.update(pre_transition_data, ts=0)

        self.t = 0
        self.env_steps_this_run = 0

    def run(self, test_mode=False):
        self.reset()

        all_terminated = False
        episode_returns = [0 for _ in range(self.batch_size)]
        episode_lengths = [0 for _ in range(self.batch_size)]
        episode_agent_returns = np.zeros((self.args.n_agents, self.batch_size))  # 创建一个self.batch_size行，n列的矩阵
        self.mac.init_hidden(batch_size=self.batch_size)
        terminated = [False for _ in range(self.batch_size)]
        envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
        final_env_infos = []  # may store extra stats like battle won. this is filled in ORDER OF TERMINATION

        while True:

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch for each un-terminated env
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, bs=envs_not_terminated, test_mode=test_mode)
            cpu_actions = actions.to("cpu").detach().numpy()

            # Update the actions taken
            if "ddpg" in self.args.name:
                onehot_actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, bs=envs_not_terminated, test_mode=test_mode).detach()
                actions = onehot_actions.argmax(2)
                cpu_actions = actions.to("cpu").detach().numpy()
                actions_chosen = {
                    "actions": actions.unsqueeze(2).unsqueeze(1),
                    "actions_onehot": onehot_actions
                }
            else:
                actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, bs=envs_not_terminated, test_mode=test_mode)
                cpu_actions = actions.to("cpu").detach().numpy()
                actions_chosen = {
                    "actions": actions.unsqueeze(1)
                }
            self.batch.update(actions_chosen, bs=envs_not_terminated, ts=self.t, mark_filled=False)

            # Send actions to each env
            action_idx = 0
            for idx, parent_conn in enumerate(self.parent_conns):
                if idx in envs_not_terminated: # We produced actions for this env
                    if not terminated[idx]: # Only send the actions to the env if it hasn't terminated
                        parent_conn.send(("step", cpu_actions[action_idx]))
                    action_idx += 1 # actions is not a list over every env

            # Update envs_not_terminated
            envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
            all_terminated = all(terminated)
            if all_terminated:
                break

            # Post step data we will insert for the current timestep
            post_transition_data = {
                "reward": [],
                "single_rewards": [],
                "terminated": []
            }
            # Data for the next step we will insert in order to select an action
            pre_transition_data = {
                "state": [],
                "eliminated_state": [],
                "avail_actions": [],
                "obs": []
            }

            # Receive data back for each unterminated env
            for idx, parent_conn in enumerate(self.parent_conns):
                if not terminated[idx]:
                    data = parent_conn.recv()
                    # Remaining data for this current timestep
                    post_transition_data["reward"].append((data["reward"],))
                    # 为每个线程加上全局奖励,跟tensorflow中获得全局奖励的逻辑是一样的
                    # 在此处添加代码可以添加智能体奖励
                    episode_returns[idx] += data["reward"]
                    episode_lengths[idx] += 1

                    # 自己写的，在此处加入智能体奖励
                    post_transition_data["single_rewards"].append(tuple(data["single_rewards"],))
                    for agent_idx, agent_return in enumerate(data["single_rewards"]):
                        episode_agent_returns[agent_idx, idx] += agent_return
                    
                    if not test_mode:
                        self.env_steps_this_run += 1

                    # env_terminated = False
                    if data["terminated"]:
                        final_env_infos.append(data["info"])
                    # if data["terminated"] and not data["info"].get("episode_limit", False): 
                    # 这里有判断冲突，它追求自然停止，但是实际上没有自然停止的可能
                        # env_terminated = True
                    terminated[idx] = data["terminated"]
                    post_transition_data["terminated"].append((data["terminated"],))

                    # Data for the next timestep needed to select an action
                    pre_transition_data["state"].append(data["state"])
                    pre_transition_data["eliminated_state"].append(data["eliminated_state"])
                    pre_transition_data["avail_actions"].append(data["avail_actions"])
                    pre_transition_data["obs"].append(data["obs"])

            # Add post_transiton data into the batch
            self.batch.update(post_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=False)

            # Move onto the next timestep
            self.t += 1

            # Add the pre-transition data
            self.batch.update(pre_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=True)

        if not test_mode:
            self.t_env += self.env_steps_this_run

        # Get stats back for each env
        for parent_conn in self.parent_conns:
            parent_conn.send(("get_stats",None))

        env_stats = []
        for parent_conn in self.parent_conns:
            env_stat = parent_conn.recv()
            env_stats.append(env_stat)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        infos = [cur_stats] + final_env_infos

        cur_stats.update({k: sum(d.get(k, 0) for d in infos) for k in set.union(*[set(d) for d in infos])})
        
        cur_stats["n_episodes"] = self.batch_size + cur_stats.get("n_episodes", 0)
        # .get("ep_length", 0) 方法尝试获取字典中键 "ep_length" 的值，如果没有就得到0
        cur_stats["ep_length"] = sum(episode_lengths) + cur_stats.get("ep_length", 0)

        cur_returns.extend(episode_returns)

        n_test_runs = max(1, self.args.test_nepisode // self.batch_size) * self.batch_size
        if test_mode and (len(self.test_returns) == n_test_runs):
            self._log(cur_returns, cur_stats, log_prefix)
            # 自己写的
            for agent_idx, agent_return in enumerate(episode_agent_returns):
                self.logger.log_stat(f"{log_prefix}return_mean_agent_{agent_idx}", np.mean(agent_return), self.t_env)
             # 7是Fire，8是Clean，只有cleanup有clean
            all_count = actions_chosen["actions"].numel()  # 统计总元素数
            count_7 = []
            for agent_idx in range(self.args.n_agents):
                agent_count_7 = (actions_chosen["actions"][:,:,agent_idx] == 7).sum().item() 
                agent_count_7 = agent_count_7 * self.args.n_agents / all_count
                self.logger.log_stat(f"{log_prefix}action_Fire_agent_{agent_idx}", agent_count_7, self.t_env)
                count_7.append(agent_count_7)
            self.logger.log_stat(f"{log_prefix}action_Fire_mean", np.mean(count_7), self.t_env)
            self.logger.log_stat(f"{log_prefix}action_Fire_std", np.std(count_7), self.t_env)
            if self.args.env_args["scenario_name"] == "cleanup":
                count_8 = []
                for agent_idx in range(self.args.n_agents):
                    agent_count_8 = (actions_chosen["actions"][:,:,agent_idx] == 8).sum().item() 
                    agent_count_8 = agent_count_8 * self.args.n_agents / all_count
                    self.logger.log_stat(f"{log_prefix}action_Clean_agent_{agent_idx}", agent_count_8, self.t_env)
                    count_8.append(agent_count_8)
                self.logger.log_stat(f"{log_prefix}action_Clean_mean", np.mean(count_8), self.t_env)
                self.logger.log_stat(f"{log_prefix}action_Clean_std", np.std(count_8), self.t_env)
       
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
           
            # 自己写的
            for agent_idx, agent_return in enumerate(episode_agent_returns):
                self.logger.log_stat(f"{log_prefix}return_mean_agent_{agent_idx}", np.mean(agent_return), self.t_env)
            # 7是Fire，8是Clean，只有cleanup有clean
            all_count = actions_chosen["actions"].numel()  # 统计总元素数
            count_7 = []
            for agent_idx in range(self.args.n_agents):
                agent_count_7 = (actions_chosen["actions"][:,:,agent_idx] == 7).sum().item() 
                agent_count_7 = agent_count_7 * self.args.n_agents / all_count
                self.logger.log_stat(f"{log_prefix}action_Fire_agent_{agent_idx}", agent_count_7, self.t_env)
                count_7.append(agent_count_7)
            self.logger.log_stat(f"{log_prefix}action_Fire_mean", np.mean(count_7), self.t_env)
            self.logger.log_stat(f"{log_prefix}action_Fire_std", np.std(count_7), self.t_env)
            if self.args.env_args["scenario_name"] == "cleanup":
                count_8 = []
                for agent_idx in range(self.args.n_agents):
                    agent_count_8 = (actions_chosen["actions"][:,:,agent_idx] == 8).sum().item() 
                    agent_count_8 = agent_count_8 * self.args.n_agents / all_count
                    self.logger.log_stat(f"{log_prefix}action_Clean_agent_{agent_idx}", agent_count_8, self.t_env)
                    count_8.append(agent_count_8)
                self.logger.log_stat(f"{log_prefix}action_Clean_mean", np.mean(count_8), self.t_env)
                self.logger.log_stat(f"{log_prefix}action_Clean_std", np.std(count_8), self.t_env)
            
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        return self.batch

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()


def env_worker(remote, env_fn):
    # 忽略特定的警告
    warnings.filterwarnings("ignore", category=UserWarning, message="The `action_spaces` dictionary is deprecated. Use the `action_space` function instead.")
    warnings.filterwarnings("ignore", category=UserWarning, message="The `observation_spaces` dictionary is deprecated. Use the `observation_space` function instead.")
    # Make environment
    env = env_fn.x()
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            actions = data
            # 调试输出动作和动作空间的信息
            # for agent in env.agents:
            #     action = actions[agent]
            #     action_space = env.action_space(agent)
            #     print(f"Agent: {agent}, Action: {action}, Action Space: {action_space}")

            #     # 检查动作是否在动作空间内
            #     assert action_space.contains(action), f"Action {action} is not in action space for agent {agent}"

            # # Take a step in the environment
            single_rewards, total_reward, terminated, env_info = env.step(actions)
            # Return the observations, avail_actions and state to make the next action
            state = env.get_state()
            avail_actions = env.get_avail_actions()
            obs = env.get_obs()
            # 自己加的
            try:
                eliminated_state = env.get_eliminated_state()
            except Exception:
                eliminated_state = None
            remote.send({
                # Data for the next timestep needed to pick an action
                "state": state,
                "avail_actions": avail_actions,
                "obs": obs,
                "eliminated_state": eliminated_state,
                # Rest of the data for the current timestep
                "single_rewards": single_rewards,
                "reward": total_reward,
                "terminated": terminated,
                "info": env_info
            })
        elif cmd == "reset":
            env.reset()
            remote.send({
                "state": env.get_state(),
                "eliminated_state": env.get_eliminated_state(),
                "avail_actions": env.get_avail_actions(),
                "obs": env.get_obs()
            })
        elif cmd == "close":
            env.close()
            remote.close()
            break
        elif cmd == "get_env_info":
            remote.send(env.get_env_info())
        elif cmd == "get_stats":
            remote.send(env.get_stats())
        else:
            raise NotImplementedError


class CloudpickleWrapper():
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """
    def __init__(self, x):
        self.x = x
    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)
    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)

