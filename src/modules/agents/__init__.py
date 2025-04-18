REGISTRY = {}

from .rnn_agent import RNNAgent
REGISTRY["rnn"] = RNNAgent

from .SSD_agent import SSDAgent
REGISTRY["SSD"] = SSDAgent

from .ppo_agent import PPO_Agent
REGISTRY["PPO"] = PPO_Agent