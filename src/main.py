import numpy as np
import os
from collections.abc import Mapping
from os.path import dirname, abspath, join
from copy import deepcopy
from sacred import Experiment, SETTINGS
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds
import sys
import torch as th
from utils.logging import get_logger
import yaml

from run import run
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

SETTINGS['CAPTURE_MODE'] = "fd" # set to "no" if you want to see stdout/stderr in console
logger = get_logger()

ex = Experiment("pymarl")
ex.logger = logger
ex.captured_out_filter = apply_backspaces_and_linefeeds
ex.captured_out_filter = None

results_path = os.path.join(dirname(dirname(abspath(__file__))), "results")


@ex.main
def my_main(_run, _config, _log):
    # Setting the random seed throughout the modules
    config = config_copy(_config)
    np.random.seed(config["seed"])
    th.manual_seed(config["seed"])
    th.cuda.manual_seed(config["seed"])
    th.backends.cudnn.benchmark = False
    th.backends.cudnn.deterministic = True
    config["seed"] = config['env_args']['seed']

    # run the framework
    run(_run, config, _log)


def _get_config(params, arg_name, subfolder):
    config_name = None
    for _i, _v in enumerate(params):
        if _v.split("=")[0] == arg_name:
            config_name = _v.split("=")[1]
            del params[_i]
            break

    if config_name is not None:
        with open(os.path.join(os.path.dirname(__file__), "config", subfolder, "{}.yaml".format(config_name)), "r") as f:
            try:
                config_dict = yaml.load(f, Loader=yaml.FullLoader)
            except yaml.YAMLError as exc:
                assert False, "{}.yaml error: {}".format(config_name, exc)
        return config_dict

def _get_env_config_name(params, arg_name):
    config_name = None
    for _i, _v in enumerate(params):
        if _v.split("=")[0] == arg_name:
            config_name = _v.split("=")[1]
    return config_name

def recursive_dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, Mapping):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def config_copy(config):
    if isinstance(config, dict):
        return {k: config_copy(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [config_copy(v) for v in config]
    else:
        return deepcopy(config)

def parse_command(params, key, default):
    result = default
    for _i, _v in enumerate(params):
        if _v.split("=")[0].strip() == key:
            result = _v[_v.index('=') + 1:].strip()
            break
    return result

if __name__ == '__main__':
    params = deepcopy(sys.argv)

    params.append("--env-config=SSD") 
    params.append("--config=USE")

    # Get the defaults from default.yaml
    with open(os.path.join(os.path.dirname(__file__), "config", "default.yaml"), "r") as f:
        try:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            assert False, "default.yaml error: {}".format(exc)

    # Load algorithm and env base configs
    env_name = _get_env_config_name(params, "--env-config")
    env_config = _get_config(params, "--env-config", "envs")
    alg_config = _get_config(params, "--config", "algs")
    # config_dict = {**config_dict, **env_config, **alg_config}
    config_dict = recursive_dict_update(config_dict, env_config)
    config_dict = recursive_dict_update(config_dict, alg_config)

    # now add all the config to sacred
    ex.add_config(config_dict)

    # Save to disk by default for sacred
    # logger.info("Saving to FileStorageObserver in results/sacred.")
    # file_obs_path = os.path.join(results_path, "sacred")
    # ex.observers.append(FileStorageObserver.create(file_obs_path))

    if env_config['env'] in ["sc2", "sc2_v2", "one_step_matrix_game"]:
        map_name = parse_command(params, "env_args.map_name", config_dict['env_args']['map_name'])
    elif env_config['env'] in ["SSD", "SSD_debug", "SSD_com", "SSD_10s"]:
        map_name = parse_command(params, "env_args.scenario_name", config_dict['env_args']['scenario_name'])
    
    algo_name = parse_command(params, "name", config_dict['name'])
    if algo_name in ["QPC","QPC","MAT_QPC","MAT_QPC", "SSD_QPC"]:
                algo_name = os.path.join(algo_name,
                                  "coop={}".format(str(config_dict["cooperation"]))
                                  )        
    local_results_path = parse_command(params, "local_results_path", config_dict['local_results_path'])
    # file_obs_path = join(results_path, local_results_path, "sacred", map_name, algo_name)
    file_obs_path = join(results_path, "sacred", env_name+"_"+map_name, algo_name)

    logger.info("Saving to FileStorageObserver in {}.".format(file_obs_path))
    ex.observers.append(FileStorageObserver.create(file_obs_path))


    ex.run_commandline(params)
