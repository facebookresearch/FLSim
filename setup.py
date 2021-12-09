#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys

from setuptools import find_packages, setup


# 3.6.8 is the final Windows binary release for 3.6.x
REQUIRED_MAJOR = 3
REQUIRED_MINOR = 6
REQUIRED_MICRO = 8


version = {}
with open("flsim/version.py") as fp:
    exec(fp.read(), version)

__version__ = version["__version__"]

# Check for python version
if sys.version_info < (REQUIRED_MAJOR, REQUIRED_MINOR, REQUIRED_MICRO):
    error = (
        "Your version of python ({major}.{minor}.{micro}) is too old. You need "
        "python >= {required_major}.{required_minor}.{required_micro}"
    ).format(
        major=sys.version_info.major,
        minor=sys.version_info.minor,
        micro=sys.version_info.micro,
        required_major=REQUIRED_MAJOR,
        required_minor=REQUIRED_MINOR,
        required_micro=REQUIRED_MICRO,
    )
    sys.exit(error)


src_dir = os.path.abspath(os.path.dirname(__file__))

with open("README.md", "r", encoding="utf8") as fh:
    long_description = fh.read()

requirements_txt = os.path.join(src_dir, "requirements.txt")
with open(requirements_txt, encoding="utf8") as f:
    required = f.read().splitlines()

dev_required = []

setup(
    name="flsim",
    version=__version__,
    author="The FLSim Team",
    description="Federated Learning Simulator (FLSim) is a flexible, standalone core library that simulates FL settings with a minimal, easy-to-use API. FLSim is domain-agnostic and accommodates many use cases such as vision and text.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://flsim.ai",
    project_urls={
        "Documentation": "https://flsim.ai/api",
        "Source": "https://github.com/facebookresearch/flsim",
    },
    license="Apache-2.0",
    install_requires=required,
    extras_require={"dev": dev_required},
    packages=find_packages(),
    keywords=[
        "PyTorch",
        "Federated Learning",
        "FL",
        "On device training",
        "Differential Privacy",
        "Secure Aggregation",
        "Privacy Preserving Machine Learning",
        "PPML",
        "PPAI",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering",
    ],
    python_requires=f">={REQUIRED_MAJOR}.{REQUIRED_MINOR}.{REQUIRED_MICRO}",
)
