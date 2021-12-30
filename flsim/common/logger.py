#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
