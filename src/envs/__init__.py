from functools import partial
import sys
import os
# Check if DISPLAY environment variable is set
if "DISPLAY" not in os.environ:
    print("DISPLAY environment variable is missing. Disabling rendering.")
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    import glfw
    glfw.init = lambda: False  # Override glfw.init to prevent initialization


from .multiagentenv import MultiAgentEnv
from .gymma import _GymmaWrapper

try:
    smac = True
    from .smac_v1 import StarCraft2EnvWrapper
except Exception as e:
    print(e)
    smac = False

try:
    smacv2 = True
    from .smac_v2 import StarCraft2Env2Wrapper
except Exception as e:
    print(e)
    smacv2 = False

try:
    MPE = True
    from .petting_zoo import PettingZooEnvWrapper
except Exception as e:
    print(e)
    MPE = False
    
from .SSD import SSD_EnvWrapper
try:
    SSD = True
    
except Exception as e:
    print(e)
    SSD = False



def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)


REGISTRY = {}



if smac:
    REGISTRY["sc2"] = partial(env_fn, env=StarCraft2EnvWrapper)
    if sys.platform == "linux":
        os.environ.setdefault("SC2PATH",
                              os.path.join(os.getcwd(), "3rdparty", "StarCraftII"))
else:
    print("SMAC V1 is not supported...")


if smacv2:
    REGISTRY["sc2_v2"] = partial(env_fn, env=StarCraft2Env2Wrapper)
    if sys.platform == "linux":
        os.environ.setdefault("SC2PATH",
                              os.path.join(os.getcwd(), "3rdparty", "StarCraftII"))
else:
    print("SMAC V2 is not supported...")


if MPE:
    REGISTRY["MPE"] = partial(env_fn, env=PettingZooEnvWrapper)
else:
    print("MPE is not supported...")


if SSD:
    REGISTRY["SSD"] = partial(env_fn, env=SSD_EnvWrapper)
else:
    print("SSD is not supported...")



REGISTRY["gymma"] = partial(env_fn, env=_GymmaWrapper)

from .matrix_game import OneStepMatrixGame

REGISTRY["one_step_matrix_game"] = partial(env_fn, env=OneStepMatrixGame)

print("Supported environments:", REGISTRY)
