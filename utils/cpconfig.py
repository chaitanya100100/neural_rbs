"""
Author: Chaitanya Patel

I couldn't find a way to have a good ol' yaml config file which can be updated
easily from command line (like argparse). So I just decided to code it myself.
"""

import yaml
import socket
import argparse
from ast import literal_eval

def update_cfg_dict(cfg, extra_args):
    """Update cfg dict with command line arguments.

    This function provides a convenient way to update yaml cfg from command line.
    If cfg looks something like this:
    ```
        cfg = {
            'model': {
                'dynamic': {
                    'mid_dim': 256,
                    'ckpt_path': '/some/path.pth',
                }
            }
            'data': {
                'split_percentage': 0.9,
            }
        }
    ```
    Then you can update it by passing extra command line arguments and catching
    them as `args, extra_args = parser.parse_known_args()`. Example command line
    arguments for `cfg` above is:
    ```
        --model-dynamic-mid_dim 512
        --model-dynamic-ckpt_path /bla/foo/bar.pth
        --data-split_percentage 1e-2
    ```
    In short, hyphens(-) are used to go in nested config. Values are determined
    by literal_eval. 
    
    Args:
        cfg: a dict (possibly nested) of config. Typically a parsed yaml file.
        extra_args: a list or dict of args to update. Typically obtained as unknown args of argparse.
    Returns:
        Updated config.
    """
    if isinstance(extra_args, list):
        extra_args = dict(zip(extra_args[:-1:2],extra_args[1::2]))

    for key, value in extra_args.items():
        assert key.startswith('--')
        key = key[2:]
        if '-' in key:
            parent_key, child_key = key.split('-', 1)
            assert parent_key in cfg, f"{parent_key} not in cfg"
            cfg[parent_key].update(
                update_cfg_dict(cfg[parent_key], {"--"+child_key: value}))
        else:
            if isinstance(cfg[key], str):
                new_value = value
            elif isinstance(value, str):
                new_value = literal_eval(value)
            else:
                new_value = value
            assert type(new_value) == type(cfg[key])
            cfg[key] = new_value
    return cfg

def get_config_general():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="path to config", required=True)
    args, extra_args = parser.parse_known_args()
    with open(args.config, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    cfg = update_cfg_dict(cfg, extra_args)
    return cfg


# PROJECT specific stuff below
def update_to_gcp_paths(cfg):
    cfg['data']['physion_path'] = '/home/chpatel_stanford_edu/physion_dataset'
    cfg['data']['saved_data_path'] = '/home/chpatel_stanford_edu/physion_dataset/saved_data'
    if 'paths' in cfg:
        cfg['paths']['root_log_path'] = '/mnt/disks/sdb/test/'
    return cfg


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="path to config", required=True)
    args, extra_args = parser.parse_known_args()
    with open(args.config, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    if 'cfg' in cfg:
        cfg = cfg['cfg']

    cfg = update_cfg_dict(cfg, extra_args)
    return cfg
