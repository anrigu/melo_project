import pytest
import torch
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.random_functions import random_polynomial, random_sin_func, random_gaussian

def test_polynomial_basic_functionality():
    # Test that the function returns a callable
    poly_func = random_polynomial()
    assert callable(poly_func)
    
    # Test that the function works with scalar input
    x = torch.tensor(0.5)
    result = poly_func(x)
    assert isinstance(result, torch.Tensor)
    assert result.dim() == 0  # Scalar output for scalar input
    
    # Test with vector input
    x_vector = torch.tensor([0.1, 0.5, 1.0])
    result_vector = poly_func(x_vector)
    assert isinstance(result_vector, torch.Tensor)
    assert result_vector.shape == x_vector.shape
    
    # Test with numpy array input
    np_x = np.array([0.2, 0.6, 0.9])
    np_result = poly_func(np_x)
    assert isinstance(np_result, torch.Tensor)
    assert np_result.shape == np_x.shape

def test_polynomial_parameters():
    # Test with specific degree probabilities
    degree_probs = [0.0, 0.0, 1.0]  # Force degree 2
    poly_func = random_polynomial(degr_probs=degree_probs)
    
    # Test with specific coefficient ranges
    min_coefs = [1, 2, 3]
    max_coefs = [2, 3, 4]
    poly_func = random_polynomial(min_coefs=min_coefs, max_coefs=max_coefs)
    
    # Test with specific beta parameters
    beta_params = [[2, 5], [5, 2], [3, 3]]
    poly_func = random_polynomial(beta_params=beta_params)

def test_sin_func_basic_functionality():
    # Test that the function returns a callable
    sin_func = random_sin_func()
    assert callable(sin_func)
    
    # Test that the function works with scalar input
    x = torch.tensor(0.5)
    result = sin_func(x)
    assert isinstance(result, torch.Tensor)
    assert result.dim() == 0  # Scalar output for scalar input
    
    # Test with vector input
    x_vector = torch.tensor([0.1, 0.5, 1.0])
    result_vector = sin_func(x_vector)
    assert isinstance(result_vector, torch.Tensor)
    assert result_vector.shape == x_vector.shape
    
    # Test with numpy array input
    np_x = np.array([0.2, 0.6, 0.9])
    np_result = sin_func(np_x)
    assert isinstance(np_result, torch.Tensor)
    assert np_result.shape == np_x.shape

def test_sin_func_parameters():
    # Test with specific parameter ranges
    period_range = [5, 6]
    amplitude_range = [2, 3]
    sin_func = random_sin_func(
        period_range=period_range,
        amplitude_range=amplitude_range
    )
    
    # Test outputs are within expected amplitude range
    x = torch.linspace(0, 10, 100)
    y = sin_func(x)
    assert y.min() >= -amplitude_range[1]
    assert y.max() <= amplitude_range[1]

def test_gaussian_basic_functionality():
    # Test with a simple mean vector
    mean_vector = np.array([0.0, 0.0])
    gaussian_func = random_gaussian(mean_vector)
    assert callable(gaussian_func)
    
    # Test with scalar input (single point)
    x = torch.tensor([0.0, 0.0], dtype=torch.float32)
    result = gaussian_func(x)
    assert isinstance(result, torch.Tensor)
    assert result.dim() == 0  # Scalar output for single point
    
    # Test with batch input
    batch_x = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=torch.float32)
    batch_result = gaussian_func(batch_x)
    assert isinstance(batch_result, torch.Tensor)
    assert batch_result.shape == torch.Size([3])  # One result per input point

def test_gaussian_properties():
    # Test that PDF integrates approximately to 1
    # We'll use a simple 2D case and check over a reasonable range
    mean_vector = np.array([0.0, 0.0])
    gaussian_func = random_gaussian(mean_vector)
    
    # Create a grid of points
    x = torch.linspace(-5, 5, 50)
    y = torch.linspace(-5, 5, 50)
    
    grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
    grid_points = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
    
    # Evaluate PDF at all grid points
    pdf_values = gaussian_func(grid_points)
    
    # Approximate integration over the grid
    delta_x = 10.0 / 50
    delta_y = 10.0 / 50
    integral = torch.sum(pdf_values) * delta_x * delta_y
    
    # The integral should be approximately 1
    assert torch.isclose(integral, torch.tensor(1.0), atol=0.1)