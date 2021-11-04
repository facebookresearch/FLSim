#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

from abc import abstractmethod

import numpy as np


class SkewnessType:
    UNIFORM: str = "uniform"
    POWERLAW: str = "powerlaw"


class SkewnessFactory:
    """This class implements various data quantity skews between clients"""

    class Skewness:
        @abstractmethod
        def num_samples(self, client_idx: int) -> float:
            """Override this function to return the unnormalized number of
            samples corresponding to client_idx"""
            pass

    class Uniform(Skewness):
        """Uniform skew, data is evenly split between clients"""

        def num_samples(self, client_idx: int) -> float:
            return 1.0

    class Powerlaw(Skewness):
        """Quantity skew follows the powerlaw distribution"""

        def __init__(self, num_clients: int, alpha: float):
            assert num_clients is not None, "num_clients must be provided."
            assert alpha is not None, "alpha must be provided."
            assert 0.0 < alpha <= 10.0, "alpha must be in the interval (0, 10]"
            self._weights = np.random.power(alpha, num_clients)

        def num_samples(self, client_idx: int) -> float:
            return self._weights[client_idx]

    @staticmethod
    def create(skewness_type: str, **kwargs):
        """Returns a Skewness object, where Skewness.num_samples maps client_idx
        to the unnormalized number of samples for that client and is meant to be
        passed to CustomSSLSharding as num_samples_skewness_client
        """
        if skewness_type == SkewnessType.UNIFORM:
            return SkewnessFactory.Uniform()
        elif skewness_type == SkewnessType.POWERLAW:
            return SkewnessFactory.Powerlaw(
                kwargs.get("num_clients"), kwargs.get("alpha")
            )
        else:
            assert f"Invalid skewness type: {skewness_type}."
