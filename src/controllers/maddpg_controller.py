from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th
from .basic_controller import BasicMAC


# This multi-agent controller shares parameters between agents
class DDPGMAC(BasicMAC):
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        input_shape = self._get_input_shape(scheme)
        self._build_agents(input_shape)
        self.agent_output_type = args.agent_output_type
        if args.action_selector is not None:
            self.action_selector = action_REGISTRY[args.action_selector](args)
        self.hidden_states = None
        self.e_hidden_states = None

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False, explore=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs = self.forward(ep_batch, t_ep, return_logits=(not test_mode))
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env,test_mode=test_mode, explore=explore)
        if getattr(self.args, "use_ent_reg", False):
            return chosen_actions, agent_outputs
        return chosen_actions


    def forward(self, ep_batch, t, return_logits=True):
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)

        if self.agent_output_type == "pi_logits":
            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10

            if return_logits:
                return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)

        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)
