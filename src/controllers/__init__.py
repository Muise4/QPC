REGISTRY = {}

from .basic_controller import BasicMAC
REGISTRY["basic_mac"] = BasicMAC

from .SSD_controller import SSD_MAC
REGISTRY["SSD_mac"] = SSD_MAC

from .maddpg_controller import DDPGMAC
REGISTRY["DDPG_mac"] = DDPGMAC

from .n_controller import NMAC
REGISTRY["n_mac"] = NMAC
