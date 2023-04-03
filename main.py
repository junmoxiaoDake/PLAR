import numpy as np
import os
import collections
from os.path import dirname, abspath
from copy import deepcopy
from sacred import Experiment, SETTINGS
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds
import sys
import torch as th
from utils.logging import get_logger
import yaml

from run import run


os.environ['CUDA_VISIBLE_DEVICES'] = '7'
SETTINGS['CAPTURE_MODE'] = "fd" # set to "no" if you want to see stdout/stderr in console
logger = get_logger()

ex = Experiment("pymarl")
ex.logger = logger
ex.captured_out_filter = apply_backspaces_and_linefeeds


results_path = os.path.join(dirname(dirname(abspath(__file__))), "results")

@ex.main
def my_main(_run, _config, _log):
    config = config_copy(_config)
    np.random.seed(config["seed"])
    th.manual_seed(config["seed"])
    config['env_args']['seed'] = config["seed"]
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
                config_dict = yaml.load(f)
            except yaml.YAMLError as exc:
                assert False, "{}.yaml error: {}".format(config_name, exc)
        return config_dict


def recursive_dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
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


def parse_command(inputs, key, default):
    result = default
    for v in inputs:
        if v.split("=")[0].strip() == key:
            result = v.split("=")[1].strip()
            break
    return result


if __name__ == '__main__':

    params = deepcopy(sys.argv)

    with open(os.path.join(os.path.dirname(__file__), "config", "default.yaml"), "r") as f:
        try:
            config_dict = yaml.load(f)
        except yaml.YAMLError as exc:
            assert False, "default.yaml error: {}".format(exc)

    env_config = _get_config(params, "--env-config", "envs")
    alg_config = _get_config(params, "--config", "algs")


    config_dict = recursive_dict_update(config_dict, env_config)
    config_dict = recursive_dict_update(config_dict, alg_config)

    config_dict['choose_matrix'] = parse_command(params , "choose_matrix" , False)
    config_dict['choose_state'] = parse_command(params, "choose_state", True)


    ex.add_config(config_dict)

    game_map = parse_command(params , "env_args.map_name" , config_dict['env_args']['map_name'])
    alg_name = parse_command(params , "name" , config_dict['name'])
    logger.info("Saving to FileStorageObserver in results/sacred/"+game_map+"/"+alg_name)
    file_obs_path = os.path.join(results_path, "sacred", game_map, alg_name)
    ex.observers.append(FileStorageObserver.create(file_obs_path))

    ex.run_commandline(params)

