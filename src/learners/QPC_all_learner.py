import copy
from components.episode_buffer import EpisodeBatch
from modules.critics.QPC import SSD_QPCCritic, SSD_QPC_chaowangluo_Critic
from utils.rl_utils import build_td_lambda_targets
import torch as th
from torch.optim import RMSprop


class QPC_ALL_Learner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.mac = mac
        self.logger = logger

        self.last_target_update_step = 0
        self.critic_training_steps = 0

        self.log_stats_t = -self.args.learner_log_interval - 1

        if args.chaowangluo:
            self.critic = SSD_QPC_chaowangluo_Critic(scheme, args)
        else:
            self.critic = SSD_QPCCritic(scheme, args)
        self.target_critic = copy.deepcopy(self.critic)

        self.agent_params = list(mac.parameters())
        self.critic_params = list(self.critic.parameters())
        self.params = self.agent_params + self.critic_params

        self.agent_optimiser = RMSprop(params=self.agent_params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        self.critic_optimiser = RMSprop(params=self.critic_params, lr=args.critic_lr, alpha=args.optim_alpha, eps=args.optim_eps)

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        bs = batch.batch_size
        max_t = batch.max_seq_length
        if self.args.use_single_rewards:
            rewards = batch["single_rewards"][:, :-1]
        else :
            rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"][:, :-1]

        critic_mask = mask.clone()

        mask = mask.repeat(1, 1, self.n_agents).view(-1)

        q_vals, critic_train_stats, grads = self._train_critic(batch, rewards, terminated, actions, avail_actions,
                                                        critic_mask, bs, max_t)
        actions = actions[:,:-1]
        
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length - 1):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time

        # Mask out unavailable actions, renormalise (as in action selection)
        mac_out[avail_actions == 0] = 0
        mac_out = mac_out/mac_out.sum(dim=-1, keepdim=True)
        mac_out[avail_actions == 0] = 0

        # Calculated baseline
        # 这里的q_vals获得的实际上也是每个智能体的buffer中对应的动作的联合价值，是Qi{a-1，ai}
        # 每个智能体的策略的Q值期望是baseline
        q_vals = q_vals.reshape(-1, self.n_actions)
        pi = mac_out.view(-1, self.n_actions)
        # 表示在 其他智能体动作固定 的情况下，智能体的动作对全局奖励的期望贡献。
        baseline = (pi * q_vals).sum(-1).detach()

        # Calculate policy grad with mask
        q_taken = th.gather(q_vals, dim=1, index=actions.reshape(-1, 1)).squeeze(1)
        # 实际使用的动作的联合Q值-反事实基线
        advantages = (q_taken - baseline).detach()
        
        pi_taken = th.gather(pi, dim=1, index=actions.reshape(-1, 1)).squeeze(1)
        pi_taken[mask == 0] = 1.0
        log_pi_taken = th.log(pi_taken)
        
        actor_loss = - ((advantages * log_pi_taken) * mask).sum() / mask.sum()

        if self.args.use_coo_grad:
            grad_taken = th.gather(grads.reshape(-1, self.n_actions), dim=1, index=actions.reshape(-1, 1)).squeeze(1).detach()
            ######### 使用pi还是log(pi)?这需要明天公式推到一下试试
            actor_loss -= ((log_pi_taken * grad_taken) * mask).sum() / mask.sum()

        # Optimise agents
        self.agent_optimiser.zero_grad()
        actor_loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.agent_params, self.args.grad_norm_clip)
        self.agent_optimiser.step()

        if (self.critic_training_steps - self.last_target_update_step) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_step = self.critic_training_steps

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            ts_logged = len(critic_train_stats["critic_loss"])
            for key in ["critic_loss", "critic_grad_norm", "td_error_abs", "q_taken_mean", "target_mean"]:
                self.logger.log_stat(key, sum(critic_train_stats[key])/ts_logged, t_env)

            self.logger.log_stat("advantage_mean", (advantages * mask).sum().item() / mask.sum().item(), t_env)
            self.logger.log_stat("pg_loss", actor_loss.item(), t_env)
            self.logger.log_stat("agent_grad_norm", grad_norm, t_env)
            self.logger.log_stat("pi_max", (pi.max(dim=1)[0] * mask).sum().item() / mask.sum().item(), t_env)
            if self.args.use_coo_grad:
                self.logger.log_stat("grad_taken_mean", (grad_taken * mask).sum().item() / mask.sum().item(), t_env)
            self.log_stats_t = t_env

    def _train_critic(self, batch, rewards, terminated, actions, avail_actions, mask, bs, max_t):
        # Optimise critic
        # 1. 获取目标网络（target critic）的 Q 值
        # target_q_vals 的形状一般为 [batch_size, timesteps, ..., n_actions]，这里取所有数据（[:, :]）
        target_Qtot, target_q_vals, target_grad = self.target_critic(batch, actions.squeeze(-1), grad=True)
        # 2. 根据每个智能体实际执行的动作，从 target_q_vals 中提取相应的 Q 值
        # 使用 th.gather 沿动作维度（dim=3）选取动作对应的 联合动作Q 值，因为其他智能体的动作保持不变嘛，然后 squeeze 去掉多余的维度
        # 这里的训练获得的实际上是每个智能体的buffer中对应的联合动作价值，是Qi{a-1，ai}
        targets_taken = th.gather(target_q_vals, dim=3, index=actions).squeeze(3)

        # Calculate td-lambda targets
        # 3. 利用 TD-λ 算法构造 TD 目标
        # 这个函数综合了奖励、终止信号、mask（有效数据标记）、以及 target Q 值，
        # 并结合折扣因子 gamma 和 TD-λ 参数，构造出每个时间步的目标值
        targets = build_td_lambda_targets(batch["single_rewards"][:, :-1], terminated, mask, targets_taken, self.n_agents, self.args.gamma, self.args.td_lambda)
        targets_Qtot = build_td_lambda_targets(batch["reward"][:, :-1], terminated, mask, target_Qtot, self.n_agents, self.args.gamma, self.args.td_lambda)
        q_vals = th.zeros_like(target_q_vals)[:, :-1]
        grads = th.zeros_like(target_grad)[:, :-1]
        # 4. 初始化一个用于保存预测 Q 值的张量，去掉最后一个时间步（一般最后一步没有对应的 target）

        running_log = {
            "critic_loss": [],
            "critic_grad_norm": [],
            "td_error_abs": [],
            "target_mean": [],
            "q_taken_mean": [],
        }
        # 6. 反向遍历每个时间步（从最后一个时间步回溯，有助于 TD-λ 目标的计算）
        for t in reversed(range(rewards.size(1))):
            # 获取当前时间步 t 的 mask，并扩展成 [batch_size, n_agents] 的形状
            mask_t = mask[:, t].expand(-1, self.n_agents)
            # 如果当前时间步没有任何有效数据，则跳过该时间步
            if mask_t.sum() == 0:
                continue
        # 7. 计算当前时间步 t 的 Q 值
        # 调用当前 critic 网络，输入 batch 和时间步 t，得到当前时刻各智能体的 Q 值
            # q_t = self.critic(batch, t)
            Qtot, Q_agent, grad = self.critic(batch, actions.squeeze(-1), grad=True, t = t)

        # 将 q_t reshape 成 [batch_size, n_agents, n_actions] 后保存
        # 这里保存了每个动作所对应的Q值，当然，其他智能体的动作还是保持自己的原样的情况下
            q_vals[:, t] = Q_agent.view(bs, self.n_agents, self.n_actions)
            grads[:, t] = grad.view(bs, self.n_agents, self.n_actions)
        # 8. 从当前 Q 值中，根据实际执行的动作（actions）提取各智能体对应的 Q 值
        # 这里 actions[:, t:t+1] 保持维度一致，gather 操作在 dim=3 选择相应动作的 Q 值，然后 squeeze 去除多余维度
            # 这里的训练获得的实际上也是每个智能体的buffer中对应的动作的联合价值，是Qi{a-1，ai}
            q_taken = th.gather(Q_agent, dim=3, index=actions[:, t:t+1]).squeeze(3).squeeze(1)
        # 获取当前时间步的 TD-λ 目标
            targets_t = targets[:, t]
            targets_Qtot_t = targets_Qtot[:, t]
        # 9. 计算 TD 误差：实际 Q 值与目标值之间的差（目标值使用 detach()，不传递梯度）
            td_error = (q_taken - targets_t.detach())
            td_tot_error = (Qtot.squeeze(-1) - targets_Qtot_t.detach())

            # 0-out the targets that came from padded data
            masked_td_error = td_error * mask_t
            masked_td_tot_error = td_tot_error * mask_t

            # Normal L2 loss, take mean over actual data
            loss = ((masked_td_error ** 2).sum()) / mask_t.sum()
            if self.args.use_TLoss:
                loss += (masked_td_tot_error ** 2).sum() / mask_t.sum()
            self.critic_optimiser.zero_grad()
            loss.backward()
            grad_norm = th.nn.utils.clip_grad_norm_(self.critic_params, self.args.grad_norm_clip)
            self.critic_optimiser.step()
            self.critic_training_steps += 1

            running_log["critic_loss"].append(loss.item())
            running_log["critic_grad_norm"].append(grad_norm)
            mask_elems = mask_t.sum().item()
            running_log["td_error_abs"].append((masked_td_error.abs().sum().item() / mask_elems))
            running_log["q_taken_mean"].append((q_taken * mask_t).sum().item() / mask_elems)
            running_log["target_mean"].append((targets_t * mask_t).sum().item() / mask_elems)
        # 这里的q_vals获得的实际上也是每个智能体的buffer中对应的动作的联合价值，是Qi{a-1，ai}
        return q_vals, running_log, grads

    def _update_targets(self):
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.critic.cuda()
        self.target_critic.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.critic.state_dict(), "{}/critic.th".format(path))
        th.save(self.agent_optimiser.state_dict(), "{}/agent_opt.th".format(path))
        th.save(self.critic_optimiser.state_dict(), "{}/critic_opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.critic.load_state_dict(th.load("{}/critic.th".format(path), map_location=lambda storage, loc: storage))
        # Not quite right but I don't want to save target networks
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.agent_optimiser.load_state_dict(th.load("{}/agent_opt.th".format(path), map_location=lambda storage, loc: storage))
        self.critic_optimiser.load_state_dict(th.load("{}/critic_opt.th".format(path), map_location=lambda storage, loc: storage))
