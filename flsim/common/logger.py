#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import sys


LOGGING_LEVEL = logging.WARNING
# If this flag is true logging will be printed to both stdout and stderr
PRINT_TO_STDOUT = False


class Logger:
    parent_name = "FLSimLogger"
    _instance = None
    logging_level = LOGGING_LEVEL
    print_to_stdout = PRINT_TO_STDOUT
    parent_logger = logging.getLogger(parent_name)
    parent_logger.setLevel(logging_level)
    children_loggers = []

    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """Returns a Logger which is a child of the PARENT_LOGGER. Logger hierarchy is
        through the name, with `.` used to denote hierarchy levels.
        """
        logger = logging.getLogger(cls.parent_name + "." + name)
        if cls.print_to_stdout:
            handler = logging.StreamHandler(sys.stdout)
            cls.parent_logger.addHandler(handler)

        cls.children_loggers.append(logger)
        return logger

    @classmethod
    def set_logging_level(cls, level: int) -> None:
        cls.parent_logger.setLevel(level)
        for child_logger in cls.children_loggers:
            child_logger.setLevel(0)
