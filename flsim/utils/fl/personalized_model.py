#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import abc
import copy
from typing import Dict, Iterable

import torch


class FLModelWithPrivateModules(abc.ABC):
    """
    This class emulates the user-private modules in FL by making them reference to
    class-level global attributes.
    The user of this class must make sure the invariance that the user-private
    modules must reference to class-level attributes.

    In federated learning, this emulation should be handled at the following points:
    when a new instance is created, when models are copied to each user, when doing
    forward propagation, when backprop, and when user models are averaged back to the
    server.
    """

    USER_PRIVATE_MODULE_PREFIX = "USER_PRIVATE_MODULE"

    user_private_module_dict: Dict[str, torch.nn.Module] = {}

    @classmethod
    def clear_user_private_module_dict(cls):
        cls.user_private_module_dict.clear()

    @classmethod
    def get_user_private_parameters(cls) -> Iterable[torch.Tensor]:
        """Return emulated mapping that maps each user to her private params."""
        for module in cls.user_private_module_dict.values():
            for param in module.parameters():
                yield param

    def _get_user_private_module_attr_name(self, module_name):
        return f"{self.USER_PRIVATE_MODULE_PREFIX}_{module_name}"

    def _maybe_set_up_user_private_modules(self, forced: bool = False):
        """
        Set an instance's private modules to class attributes to share among
        all users. This function runs only when all user-private attributes
        have been set.
        """
        if not forced:
            for module_name in self._get_user_private_module_names():
                # The user-private modules may not be set during component creation.
                if not hasattr(self, module_name) or getattr(self, module_name) is None:
                    return

        # Initialize the class attributes if not exist.
        for module_name in self._get_user_private_module_names():
            if module_name not in self.user_private_module_dict:
                self.user_private_module_dict[module_name] = getattr(self, module_name)

            # Replace instance-based private attributes with the class attributes.
            # for module_name in self._get_user_private_module_names():
            # Remove instance version if not removed.
            if hasattr(self, module_name):
                delattr(self, module_name)

            setattr(
                self,
                self._get_user_private_module_attr_name(module_name),
                self.user_private_module_dict[module_name],
            )

    def _set_forward_hooks(self):
        """Set forward hooks to reuse the forward() of the parent class.
        The pre-forward hook changes the name of the user-private parameters
        back to the original ones to reuse the forward() function of the parent
        class. The forward hook changes the name back to have the
        USER_PRIVATE_MODULE_PREFIX.
        """

        def set_user_private_modules(module, inputs):
            for key in module._get_user_private_module_names():
                setattr(module, key, module.user_private_module_dict[key])

        def remove_user_private_modules(module, inputs, outputs):
            for key in module._get_user_private_module_names():
                delattr(module, key)

        self.register_forward_pre_hook(set_user_private_modules)
        self.register_forward_hook(remove_user_private_modules)

    def __deepcopy__(self, memo):
        orig_deepcopy_method = self.__deepcopy__
        self.__deepcopy__ = None

        # Don't want to copy the user-private modules which point to the
        # class-level attributes.
        for module_name in self._get_user_private_module_names():
            delattr(self, self._get_user_private_module_attr_name(module_name))

        cp = copy.deepcopy(self, memo)

        # Re-set-up the user-private params to the class-level attributes.
        self._maybe_set_up_user_private_modules(forced=True)
        cp._maybe_set_up_user_private_modules(forced=True)

        self.__deepcopy__ = orig_deepcopy_method
        return cp

    def get_user_private_attr(self, module_name):
        return getattr(self, self._get_user_private_module_attr_name(module_name))

    @classmethod
    @abc.abstractmethod
    def _get_user_private_module_names(cls) -> Iterable[str]:
        """Return an iterable of the modules of the class to be private."""
        pass

    def federated_state_dict(self):
        """Return a state dict of federated modules."""
        state_dict = self.state_dict()
        # Do not copy user private param modules.
        for key in state_dict.keys():
            if key.startswith(self.USER_PRIVATE_MODULE_PREFIX):
                del state_dict[key]
        return state_dict

    def load_federated_state_dict(self, state_dict: Dict):
        """Load from a state dict of federated modules."""
        # pyre-fixme[16]: `FLModelWithPrivateModules` has no attribute
        #  `load_state_dict`.
        missing_keys, unexpected_keys = self.load_state_dict(
            state_dict=state_dict, strict=False
        )
        assert len(unexpected_keys) == 0, "There should be no unexpected keys"
        for key in missing_keys:
            assert key.startswith(
                self.USER_PRIVATE_MODULE_PREFIX
            ), f"Missing non-user-private parameter {key}"
        return missing_keys, unexpected_keys
