import torch as th


def build_td_lambda_targets(rewards, terminated, mask, target_qs, n_agents, gamma, td_lambda):
    # Assumes  <target_qs > in B*T*A and <reward >, <terminated >, <mask > in (at least) B*T-1*1
    # Initialise  last  lambda -return  for  not  terminated  episodes
    ret = target_qs.new_zeros(*target_qs.shape)
    # 最后一个单位时间步的奖励定义
    ret[:, -1] = target_qs[:, -1] * (1 - th.sum(terminated, dim=1))
    # Backwards  recursive  update  of the "forward  view"，逆向递归更新
    for t in range(ret.shape[1] - 2, -1,  -1):
        ret[:, t] = td_lambda * gamma * ret[:, t + 1] + mask[:, t] \
                    * (rewards[:, t] + (1 - td_lambda) * gamma * target_qs[:, t + 1] * (1 - terminated[:, t]))
    # Returns lambda-return from t=0 to t=T-1, i.e. in B*T-1*A
    # (B, T-1, D)
    return ret[:, 0:-1]



def build_gae_advantages(rewards, terminated, mask, target_qs, gamma, td_lambda):
    # 计算时序差分误差 (TD error)，但偏向是单智能体的
    td_error = rewards + gamma * target_qs[:, 1:] * (1 - terminated) - target_qs[:, :-1]
    # 递归计算 GAE
    gae = 0
    advantages = []
    for t in reversed(range(td_error.shape[1])):
        gae = td_error[:, t] + gamma * td_lambda * mask[:, t] * gae
        advantages.append(gae)  # 使用append提高效率
    advantages = th.stack(advantages[::-1], dim=1)  # 反转list

    # 返回优势函数
    # (B, T-1)
    return advantages


def build_gae_targets(rewards, terminated, masks, values, gamma, lambd):
    # 是多智能体的
    B, T, A = values.size()
    T-=1
    advantages = th.zeros(B, T, A).to(device=values.device)
    advantage_t = th.zeros(B, A).to(device=values.device)

    for t in reversed(range(T)):
        # 修改delta计算，加入terminated处理
        # 修改 delta 计算，显式处理 terminated（即使当前全为0）
        delta = rewards[:, t] + gamma * values[:, t+1] * (1 - terminated[:, t]) * masks[:, t] - values[:, t]
        # delta = rewards[:, t] + gamma * values[:, t+1] * (1 - terminated[:, t]) - values[:, t]
        # delta = rewards[:, t] + values[:, t+1] * gamma * masks[:, t] - values[:, t]
        advantage_t = delta + advantage_t * gamma * lambd * masks[:, t]
        advantages[:, t] = advantage_t

    returns = values[:, :T] + advantages
    return advantages, returns


def build_SLI_gae_targets(rewards, terminated, masks, values, gamma, lambd, alpha):
    # 是多智能体的

    B, T, A = values.size()
    T-=1
    SIL_rewards = rewards + alpha * rewards.sum(-1).unsqueeze(-1).repeat(1,1,A)
    SIL_values = values + alpha * values.sum(-1).unsqueeze(-1).repeat(1,1,A)
    
    advantages = th.zeros(B, T, A).to(device=values.device)
    advantage_t = th.zeros(B, A).to(device=values.device)

    SIL_advantages = th.zeros(B, T, A).to(device=values.device)
    SIL_advantage_t = th.zeros(B, A).to(device=values.device)
    for t in reversed(range(T)):
        # 修改delta计算，加入terminated处理
        # 修改 delta 计算，显式处理 terminated（即使当前全为0）
        delta = rewards[:, t] + gamma * values[:, t+1] * (1 - terminated[:, t]) * masks[:, t] - values[:, t]
        # delta = rewards[:, t] + gamma * values[:, t+1] * (1 - terminated[:, t]) - values[:, t]
        # delta = rewards[:, t] + values[:, t+1] * gamma * masks[:, t] - values[:, t]
        advantage_t = delta + advantage_t * gamma * lambd * masks[:, t]
        advantages[:, t] = advantage_t

        SIL_delta = SIL_rewards[:, t] + gamma * SIL_values[:, t+1] * (1 - terminated[:, t]) * masks[:, t] - SIL_values[:, t]
        SIL_advantage_t = SIL_delta + SIL_advantage_t * gamma * lambd * masks[:, t]
        SIL_advantages[:, t] = SIL_advantage_t
    returns = values[:, :T] + advantages
    return SIL_advantages, returns


def build_MAT_SLI_gae_targets(rewards, terminated, masks, values, gamma, lambd, alpha):
    B, T, A = values.size()
    SIL_rewards = rewards + alpha * rewards.sum(-1).unsqueeze(-1).repeat(1,1,A)
    SIL_values = values + alpha * values.sum(-1).unsqueeze(-1).repeat(1,1,A)
    SIL_advantages = SIL_rewards - SIL_values
    return SIL_advantages


def build_q_lambda_targets(rewards, terminated, mask, exp_qvals, qvals, gamma, td_lambda):
    # Assumes  <target_qs > in B*T*A and <reward >, <terminated >, <mask > in (at least) B*T-1*1
    # Initialise  last  lambda -return  for  not  terminated  episodes
    ret = exp_qvals.new_zeros(*exp_qvals.shape)
    ret[:, -1] = exp_qvals[:, -1] * (1 - th.sum(terminated, dim=1))
    # Backwards  recursive  update  of the "forward  view"
    for t in range(ret.shape[1] - 2, -1,  -1):
        reward = rewards[:, t] + exp_qvals[:, t] - qvals[:, t] #off-policy correction
        ret[:, t] = td_lambda * gamma * ret[:, t + 1] + mask[:, t] \
                    * (reward + (1 - td_lambda) * gamma * exp_qvals[:, t + 1] * (1 - terminated[:, t]))
    # Returns lambda-return from t=0 to t=T-1, i.e. in B*T-1*A
    return ret[:, 0:-1]


def build_target_q(td_q, target_q, mac, mask, gamma, td_lambda, n):
    aug = th.zeros_like(td_q[:, :1])

    #Tree diagram
    mac = mac[:, :-1]
    tree_q_vals = th.zeros_like(td_q)
    coeff = 1.0
    t1 = td_q[:]
    for _ in range(n):
        tree_q_vals += t1 * coeff
        t1 = th.cat(((t1 * mac)[:, 1:], aug), dim=1)
        coeff *= gamma * td_lambda
    return target_q + tree_q_vals


# def build_td_lambda_targets_ddpg(rewards, terminated, mask, target_qs, n_agents, gamma, td_lambda):
#     # Assumes  <target_qs > in B*T*A and <reward >, <terminated >, <mask > in (at least) B*T-1*1
#     # Initialise  last  lambda -return  for  not  terminated  episodes
#     ret = target_qs.new_zeros(*target_qs.shape)
#     ret[:, -1] = target_qs[:, -1] * (1 - th.sum(terminated, dim=1))
#     # Backwards  recursive  update  of the "forward  view"
#     for t in range(ret.shape[1] - 2, -1,  -1):
#         ret[:, t] = td_lambda * gamma * ret[:, t + 1] + mask[:, t] \
#                     * (rewards[:, t] + (1 - td_lambda) * gamma * target_qs[:, t + 1] * (1 - terminated[:, t]))
#     # Returns lambda-return from t=0 to t=T-1, i.e. in B*T-1*A
#     return ret[:, 0:-1]


def preprocess_scheme(scheme, preprocess):

    if preprocess is not None:
        for k in preprocess:
            assert k in scheme
            new_k = preprocess[k][0]
            transforms = preprocess[k][1]

            vshape = scheme[k]["vshape"]
            dtype = scheme[k]["dtype"]
            for transform in transforms:
                vshape, dtype = transform.infer_output_info(vshape, dtype)

            scheme[new_k] = {
                "vshape": vshape,
                "dtype": dtype
            }
            if "group" in scheme[k]:
                scheme[new_k]["group"] = scheme[k]["group"]
            if "episode_const" in scheme[k]:
                scheme[new_k]["episode_const"] = scheme[k]["episode_const"]

    return scheme