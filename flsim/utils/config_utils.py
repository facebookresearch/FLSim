#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import collections
import json
from typing import Any, Dict, List, Tuple, Type

from hydra.experimental import initialize, compose
from omegaconf import OmegaConf, DictConfig


def fullclassname(cls: Type[Any]) -> str:
    """
    Returns the fully qualified class name of the input class.
    """
    module = cls.__module__
    name = cls.__qualname__
    if module is not None and module != "__builtin__":
        name = module + "." + name
    return name


def _validate_cfg(component_class: Type[Any], cfg: Any):
    """
    Validate that cfg doesn't have MISSING fields. This needs to be done only after
    all defaults are set, typically in the base class.
    We do this by making sure none of the parents have ``_set_defaults_in_cfg`` method.
    """
    if not any(
        hasattr(parent, "_set_defaults_in_cfg") for parent in component_class.__bases__
    ):
        # looping over the config fields throws incase of missing field
        for _ in cfg.items():
            pass


def init_self_cfg(
    component_obj: Any,
    *,
    component_class: Type,
    config_class: Type,
    **kwargs,
):
    """
    Initialize FL component config by constructing OmegaConf object,
    setting defaults, and validating config.
    """
    cfg = (
        config_class(**kwargs)
        if not hasattr(component_obj, "cfg")
        else component_obj.cfg
    )
    cfg = OmegaConf.create(cfg)  # convert any structure to OmegaConf
    component_class._set_defaults_in_cfg(cfg)  # set default cfg params for this class
    # convert any structure to OmegaConf again, after setting defaults
    cfg = OmegaConf.create(cfg)  # pyre-ignore [6]
    _validate_cfg(component_class, cfg)  # validate the config
    component_obj.cfg = cfg


# trainer config utils for consuming hydra configs
def _flatten_dict(
    d: collections.MutableMapping, parent_key="", sep="."
) -> Dict[str, str]:
    """
    Changes json of style
    ```
    {
        "trainer" : {
            "_base_": "base_sync_trainer",
            "aggregator": {
                "_base_": "base_fed_avg_with_lr_sync_aggregator",
                "lr": 0.1
            }
        }
    }
    ```
    to
    ```
    {
        "trainer._base_": "base_sync_trainer",
        "trainer.aggregator._base_": "base_fed_avg_with_lr_sync_aggregator",
        "trainer.aggregator.lr": 0.1,
    }
    ```
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        # if value is not a dict and is mutable, extend the items and flatten again.
        # > hacky way of preserving dict values by checking if key has _dict as suffix.
        if not new_key.endswith("_dict") and isinstance(v, collections.MutableMapping):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            # check if a number needs to be retained as a string
            # the repalce with one dot is needed to handle floats
            if type(v) is str and v.replace(".", "", 1).isdigit():
                v = f'"{v}"'  # enclose it with quotes if so.
            items.append((new_key, v))
    return dict(items)


def _handle_values_for_overrides_list(v: Any) -> Any:
    """
    Handle the special massaging of some values of JSON need to for it to be supplied
    to Hydra's overrides list.
    """
    # python's None --> cmd line null for override list
    v = "null" if v is None else v

    # if value is a dict, convert it to string to work with override list.
    # dump twice to escape quotes correctly.
    v = json.dumps(json.dumps(v)) if type(v) is dict else v
    # escape = char in value when present
    v = v.replace(r"=", r"\=") if type(v) is str else v
    return v


def _hydra_merge_order(dotlist_entry: str) -> Tuple:
    """
    The override list needs to be ordered as the last one wins in case of
    duplicates: https://hydra.cc/docs/advanced/defaults_list#composition-order
    This function arranges the list so that _base_ is at the top, and we
    proceed with overrides from top to bottom.
    """
    key = dotlist_entry.split("=")[0]
    # presence of "@" => it is a _base_ override
    default_list_item_indicator = key.count("@")  # 1 if true, 0 otherwise
    # level in hierarchy; based on number of "."
    hierarchy_level = key.count(".")
    # multiply by -1 to keep the default list items on top
    return (-1 * default_list_item_indicator, hierarchy_level, dotlist_entry)


def fl_json_to_dotlist(
    json_config: Dict[str, Any], append_or_override: bool = True
) -> List[str]:
    """
    Changes
    ```
    {
        "trainer._base_": "base_sync_trainer",
        "trainer.aggregator._base_": "base_fed_avg_with_lr_sync_aggregator",
        "trainer.aggregator.lr": 0.1,
    }
    ```
    to
    ```
    [
        "+trainer@trainer=base_sync_trainer",
        "+aggregator@trainer.aggregator=base_fed_avg_with_lr_sync_aggregator",
        "trainer.aggregator.lr=0.1",
    ]
    ```
    The override list grammar for reference:
        https://hydra.cc/docs/advanced/override_grammar/basic
    """
    dotlist_dict = _flatten_dict(json_config)
    dotlist_list = []
    for k, v in dotlist_dict.items():
        if k.endswith("._base_"):
            # trainer.aggregator._base_ --> trainer.aggregator
            k = k.replace("._base_", "")
            # extract aggregator from trainer.aggregator
            config_group = k.split(".")[-1]
            # trainer.aggregator --> +aggregator@trainer.aggregator
            k = f"+{config_group}@{k}"
            # +aggregator@trainer.aggregator=base_fed_avg_with_lr_sync_aggregator
            dotlist_list.append(f"{k}={v}")
        else:
            v = _handle_values_for_overrides_list(v)
            prefix = "++" if append_or_override else ""
            dotlist_list.append(f"{prefix}{k}={v}")
    sorted_dotlist_list = sorted(dotlist_list, key=_hydra_merge_order)
    return sorted_dotlist_list


def fl_config_from_json(
    json_config: Dict[str, Any], append_or_override: bool = True
) -> DictConfig:
    """
    Accepts the FLSim config in json format and constructs a Hydra config object.
    """
    with initialize(config_path=None):
        cfg = compose(
            config_name=None,
            overrides=fl_json_to_dotlist(json_config, append_or_override),
        )
        return cfg


def maybe_parse_json_config():
    """
    Parse the command line args and build a config object if json config is supplied.
    This comes in handy when we want to supply a json config file during to buck run.
    This function will no longer be relevant once FLSim entirely moves to YAML configs.
    """
    cfg = None
    parser = argparse.ArgumentParser(description="Run training loop for FL example")
    parser.add_argument("--config-file", type=str, default=None, help="JSON config")
    args, _ = parser.parse_known_args()
    # if JSON config is specified, build a DictConfig
    if args.config_file is not None:
        with open(args.config_file, "r") as config_file:
            json_config = json.load(config_file)
            cfg = fl_config_from_json(json_config["config"])
    # else:  assume yaml config, and let hydra handle config construction
    return cfg


def is_target(config, cls):
    return config._target_ == cls._target_
