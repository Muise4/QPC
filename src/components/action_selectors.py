import torch as th
from torch.distributions import Categorical
from .epsilon_schedules import DecayThenFlatSchedule

REGISTRY = {}


class MultinomialActionSelector():

    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)
        self.test_greedy = getattr(args, "test_greedy", True)

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):
        masked_policies = agent_inputs.clone()
        masked_policies[avail_actions == 0.0] = 0.0

        self.epsilon = self.schedule.eval(t_env)

        if test_mode and self.test_greedy:
            picked_actions = masked_policies.max(dim=2)[1]
        else:
            picked_actions = Categorical(masked_policies).sample().long()

        return picked_actions


REGISTRY["multinomial"] = MultinomialActionSelector


class EpsilonGreedyActionSelector():

    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):

        # Assuming agent_inputs is a batch of Q-Values for each agent bav
        self.epsilon = self.schedule.eval(t_env)

        if test_mode:
            # Greedy action selection only
            self.epsilon = 0.0

        # mask actions that are excluded from selection
        masked_q_values = agent_inputs.clone()
        masked_q_values[avail_actions == 0.0] = -float("inf")  # should never be selected!

        random_numbers = th.rand_like(agent_inputs[:, :, 0])
        pick_random = (random_numbers < self.epsilon).long()
        random_actions = Categorical(avail_actions.float()).sample().long()

        picked_actions = pick_random * random_actions + (1 - pick_random) * masked_q_values.max(dim=2)[1]
        return picked_actions


REGISTRY["epsilon_greedy"] = EpsilonGreedyActionSelector


class PPOMultinomialActionSelector():

    def __init__(self, args):
        self.args = args
        self.test_greedy = getattr(args, "test_greedy", True)  # 是否在测试模式下使用贪婪策略

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):
        """
        选择动作的逻辑
        :param agent_inputs: 策略网络的输出，形状为 [batch_size, num_agents, num_actions]
        :param avail_actions: 可用动作的掩码，形状为 [batch_size, num_agents, num_actions]
        :param test_mode: 是否处于测试模式
        :return: 选择的动作，形状为 [batch_size, num_agents]
        """
        # 掩码处理：将不可用动作的概率置为 0
        masked_policies = agent_inputs.clone()
        masked_policies[avail_actions == 0.0] = 0.0

        # 在测试模式下使用贪婪策略
        if test_mode and self.test_greedy:
            picked_actions = masked_policies.max(dim=2)[1]  # 选择概率最大的动作
        else:
            # 创建多项式分布并采样动作
            picked_actions = Categorical(masked_policies).sample().long()
        # picked_actions = Categorical(masked_policies).sample().long()


        return picked_actions
    
REGISTRY["PPOmultinomial"] = PPOMultinomialActionSelector


from torch.distributions.one_hot_categorical import OneHotCategorical

# From https://github.com/mzho7212/LICA/blob/main/src/components/action_selectors.py
class GumbelSoftmax(OneHotCategorical):

    def __init__(self, logits, probs=None, temperature=1):
        super(GumbelSoftmax, self).__init__(logits=logits, probs=probs)
        self.eps = 1e-20
        self.temperature = temperature

    def sample_gumbel(self):
        U = self.logits.clone()
        U.uniform_(0, 1)
        return -th.log(-th.log(U + self.eps) + self.eps)

    def gumbel_softmax_sample(self):
        """ Draw a sample from the Gumbel-Softmax distribution. The returned sample will be a probability distribution
        that sums to 1 across classes"""
        y = self.logits + self.sample_gumbel()
        return th.softmax( y / self.temperature, dim=-1)

    def hard_gumbel_softmax_sample(self):
        y = self.gumbel_softmax_sample()
        return (th.max(y, dim=-1, keepdim=True)[0] == y).float()

    def rsample(self):
        return self.gumbel_softmax_sample()

    def sample(self):
        return self.rsample().detach()

    def hard_sample(self):
        return self.hard_gumbel_softmax_sample()


def onehot_from_logits(logits, avail_logits, eps=0.0):
    """
    Given batch of logits, return one-hot sample using epsilon greedy strategy
    (based on given epsilon)
    """
    # get best (according to current policy) actions in one-hot form
    argmax_acs = (logits == logits.max(-1, keepdim=True)[0]).float()
    if eps == 0.0:
        return argmax_acs

    # chooses between best and random actions using epsilon greedy
    agent_outs = th.nn.functional.softmax(logits, dim=-1)
    epsilon_action_num = avail_logits.sum(dim=-1, keepdim=True).float()
    agent_outs = ((1 - eps) * agent_outs + th.ones_like(agent_outs) * eps / epsilon_action_num)
    agent_outs[avail_logits == 0] = 0.0
    picked_actions = Categorical(agent_outs).sample()
    picked_actions = th.nn.functional.one_hot(picked_actions, num_classes=logits.shape[-1]).float()
    return picked_actions


class GumbelSoftmaxMultinomialActionSelector():

    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)
        self.test_greedy = getattr(args, "test_greedy", True)

    def select_action(self, agent_logits, avail_logits, t_env, test_mode=False, explore=False):
        masked_policies = agent_logits.clone()
        self.epsilon = self.schedule.eval(t_env)

        if test_mode and self.test_greedy:
            # return one-hot action
            picked_actions = (th.max(masked_policies, dim=-1, keepdim=True)[0] == masked_policies).float()
        else:
            if not explore:
                picked_actions = GumbelSoftmax(logits=masked_policies).gumbel_softmax_sample()
                picked_actions_hard = (th.max(picked_actions, dim=-1, keepdim=True)[0] == picked_actions).float()
                picked_actions = (picked_actions_hard - picked_actions).detach() + picked_actions
            
            else:
                # choose between the best and random actions using epsilon greedy
                agent_outs = th.nn.functional.softmax(masked_policies, dim=-1)
                epsilon_action_num = avail_logits.sum(dim=-1, keepdim=True).float()
                agent_outs = ((1 - self.epsilon) * agent_outs + th.ones_like(
                    agent_outs) * self.epsilon / epsilon_action_num)
                agent_outs[avail_logits == 0] = 0.0
                picked_actions = Categorical(agent_outs).sample()
                picked_actions = th.nn.functional.one_hot(picked_actions, num_classes=masked_policies.shape[-1]).float()

        return picked_actions


REGISTRY["gumbel"] = GumbelSoftmaxMultinomialActionSelector
