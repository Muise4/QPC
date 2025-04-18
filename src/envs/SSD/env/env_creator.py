from envs.SSD.env.cleanup import CleanupEnv
from envs.SSD.env.harvest import HarvestEnv
from envs.SSD.env.switch import SwitchEnv


def get_env_creator(
    scenario_name,
    num_agents,
    more_com, 
    use_collective_reward=False,
    inequity_averse_reward=False,
    alpha=0.0,
    beta=0.0,
    num_switches=6,
    seed = 0,
    episode_limit = 100
):
    if scenario_name == "harvest":

        def env_creator():
            return HarvestEnv(
                num_agents=num_agents,
                more_com=more_com, 
                return_agent_actions=True,
                use_collective_reward=use_collective_reward,
                inequity_averse_reward=inequity_averse_reward,
                alpha=alpha,
                beta=beta,
            )

    elif scenario_name == "cleanup":

        def env_creator():
            return CleanupEnv(
                num_agents=num_agents,
                more_com=more_com, 
                return_agent_actions=True,
                use_collective_reward=use_collective_reward,
                inequity_averse_reward=inequity_averse_reward,
                alpha=alpha,
                beta=beta,
            )

    elif scenario_name == "switch":

        def env_creator():
            return SwitchEnv(num_agents=num_agents, more_com=more_com, num_switches=num_switches)

    else:
        raise ValueError(f"env must be one of harvest, cleanup, switch, not {scenario_name}")

    return env_creator
