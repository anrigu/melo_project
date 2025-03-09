import pytest
import torch
import numpy as np
import math
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.log_multimodal import logmultinomial, logmultinomial_torch

def test_logmultinomial_basic():
    # Test simple cases
    assert np.isclose(logmultinomial(1, 1), np.log(2))  # 2!/(1!1!) = 2/1 = 2
    assert np.isclose(logmultinomial(2, 1), np.log(3))  # 3!/(2!1!) = 6/2 = 3
    assert np.isclose(logmultinomial(3, 2, 1), np.log(60))  # 6!/(3!2!1!) = 720/12 = 60

def test_logmultinomial_zeros():
    # Test with zeros
    assert np.isclose(logmultinomial(5, 0), 0.0)  # 5!/(5!0!) = 1
    assert np.isclose(logmultinomial(0, 0), 0.0)  # 0!/(0!0!) = 1

def test_logmultinomial_large_values():
    # Test with larger values to show log stability
    large_a = 100
    large_b = 200
    # Using gammaln directly to avoid factorial overflow
    from scipy.special import gammaln
    direct = gammaln(large_a + large_b + 1) - gammaln(large_a + 1) - gammaln(large_b + 1)
    log_result = logmultinomial(large_a, large_b)
    assert np.isclose(log_result, direct)

def test_logmultinomial_torch_basic():
    # Test simple cases with the correct expected values
    assert torch.isclose(logmultinomial_torch(1, 1), torch.tensor(np.log(2), dtype=torch.float64))
    assert torch.isclose(logmultinomial_torch(2, 1), torch.tensor(np.log(3), dtype=torch.float64))
    assert torch.isclose(logmultinomial_torch(3, 2, 1), torch.tensor(np.log(60), dtype=torch.float64))

def test_logmultinomial_torch_zeros():
    # Test with zeros
    assert torch.isclose(logmultinomial_torch(5, 0), torch.tensor(0.0, dtype=torch.float64))
    assert torch.isclose(logmultinomial_torch(0, 0), torch.tensor(0.0, dtype=torch.float64))

def test_numpy_torch_consistency():
    # Test that both implementations return the same values
    args = [3, 4, 5]
    np_result = logmultinomial(*args)
    torch_result = logmultinomial_torch(*args)
    assert np.isclose(np_result, torch_result)
    
    args = [10, 20, 30, 40]
    np_result = logmultinomial(*args)
    torch_result = logmultinomial_torch(*args)
    assert np.isclose(np_result, torch_result)