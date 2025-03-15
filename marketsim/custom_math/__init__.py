"""
Mathematical utilities for the EGTA framework.
This module provides math operations and utilities used in game theory calculations.
"""

from marketsim.custom_math.log_multimodal import logmultinomial, logmultinomial_torch
from marketsim.custom_math.simplex_operations import (
    uniform_mixture, random_mixture, simplex_projection, simplex_normalize,
    sample_profile, sample_profiles, simplex_projection_sum, filter_unique
)
