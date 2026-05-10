#
# Copyright (C) 2024 Apple Inc. All rights reserved.
#


import importlib
from inspect import isfunction

from omegaconf import OmegaConf


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def get_obj_from_str(string):
    module, cls = string.rsplit(".", 1)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config, **kwargs):
    config = OmegaConf.to_container(OmegaConf.create(config), resolve=True)
    # print(f"instantiating {config['target']} with config:\n  {config}")

    if "target" not in config:
        raise KeyError("Expected key `target` to instantiate.")

    try:
        return get_obj_from_str(config["target"])(**config.get("params", dict()), **kwargs)
    except Exception as e:
        print(f"Failed to instantiate_from_config {config['target']} with error: {e}.\nConfig:\n{config}")
        raise
