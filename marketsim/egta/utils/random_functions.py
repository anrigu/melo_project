import torch
import numpy as np
from scipy import stats
import math

import torch
import numpy as np
from scipy import stats
import math
'''
original version of Bryce's code: 
https://github.com/Davidson-Game-Theory-Research/gameanalysis.jl/blob/master/RandomFunctions.jl
'''
'''
TODO: validate wrtie test cases
'''
def random_polynomial(degr_probs=None, min_coefs=None, max_coefs=None, beta_params=None, device="cpu"):
    """
    Generate a random polynomial function.
    
    Parameters:
    degr_probs : list
        Probability of a polynomial with each degree, starting from 0
    min_coefs : list
        Minimum coefficient for each term (in order starting from x^0)
    max_coefs : list
        Maximum coefficient for each term
    beta_params : list or 2D array
        Parameters to pass to Beta distribution for generating coefficients
        Can be either a 1D array with 2 elements or a 2D array with shape (max_deg+1, 2)
    device : str
        Device to place tensors on ("cpu" or "cuda")
        
    Returns:
    function : A callable function representing the polynomial
    """
    # Default parameters 
    if degr_probs is None:
        degr_probs = [0, 0.6, 0.4]
    if min_coefs is None:
        min_coefs = [-5, -1, -0.2]
    if max_coefs is None:
        max_coefs = [5, 1, 0.2]
    if beta_params is None:
        beta_params = [[1, 1], [1, 1], [2, 2]]
    
    #sample degree based on probabilities
    degree = np.random.choice(range(len(degr_probs)), p=degr_probs)
    
    #generate coefficients
    if isinstance(beta_params[0], (int, float)):
        # Beta params is just [alpha, beta]
        coefficients = np.random.beta(beta_params[0], beta_params[1], size=degree+1)
    else:
        # Beta params is a 2D array with different parameters for each coefficient
        coefficients = np.array([
            np.random.beta(beta_params[i][0], beta_params[i][1]) 
            for i in range(degree+1)
        ])
    
    # Scale coefficients to the desired range
    coefficients = coefficients * (np.array(max_coefs[:degree+1]) - np.array(min_coefs[:degree+1]))
    coefficients = coefficients + np.array(min_coefs[:degree+1])
    
    # Convert to PyTorch tensors
    coefficients_tensor = torch.tensor(coefficients, dtype=torch.float32, device=device)
    
    # Define the polynomial function
    def polynomial_func(x):
        """
        Evaluate the polynomial at points x.
        
        Parameters:
        x : torch.Tensor or numpy.ndarray
            Points to evaluate the polynomial at
            
        Returns:
        torch.Tensor : Polynomial values at x
        """
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32, device=device)
        
        # Handle scalar vs vector input
        is_scalar = (x.dim() == 0)
        if is_scalar:
            x = x.unsqueeze(0)
        
        #evaluate the polynomial
        result = torch.zeros_like(x)
        for i in range(degree + 1):
            result = result + coefficients_tensor[i] * torch.pow(x, i)
        
        #return in the appropriate shape
        if is_scalar:
            return result[0]
        return result
    
    return polynomial_func

def random_sin_func(period_range=None, amplitude_range=None, 
                   period_beta_params=None, amplitude_beta_params=None, device="cpu"):
    """
    Generate a random sine function.
    
    Parameters:
    period_range : list
        [min, max] for the period
    amplitude_range : list
        [min, max] for the amplitude
    period_beta_params : list
        [alpha, beta] for the period Beta distribution
    amplitude_beta_params : list
        [alpha, beta] for the amplitude Beta distribution
    device : str
        Device to place tensors on ("cpu" or "cuda")
        
    Returns:
    function : A callable function representing the sine function
    """
    #default parameters
    if period_range is None:
        period_range = [1, 10]
    if amplitude_range is None:
        amplitude_range = [1, 10]
    if period_beta_params is None:
        period_beta_params = [1, 1]
    if amplitude_beta_params is None:
        amplitude_beta_params = [1, 1]
    
    # Sample period from Beta distribution scaled to period_range
    period = np.random.beta(period_beta_params[0], period_beta_params[1])
    period = period * (period_range[1] - period_range[0]) + period_range[0]
    
    # Sample amplitude from Beta distribution scaled to amplitude_range
    amplitude = np.random.beta(amplitude_beta_params[0], amplitude_beta_params[1])
    amplitude = amplitude * (amplitude_range[1] - amplitude_range[0]) + amplitude_range[0]
    
    # Sample phase shift uniformly from [0, period]
    shift = np.random.uniform(0, period)
    
    # Convert to PyTorch tensors
    period_tensor = torch.tensor(period, dtype=torch.float32, device=device)
    amplitude_tensor = torch.tensor(amplitude, dtype=torch.float32, device=device)
    shift_tensor = torch.tensor(shift, dtype=torch.float32, device=device)
    
    # Define the sine function
    def sin_func(x):
        """
        Evaluate the sine function at points x.
        
        Parameters:
        x : torch.Tensor or numpy.ndarray
            Points to evaluate the sine function at
            
        Returns:
        torch.Tensor : Sine function values at x
        """
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32, device=device)
        
        return amplitude_tensor * torch.sin(2 * math.pi * (x + shift_tensor) / period_tensor)
    
    return sin_func

def random_gaussian(mean_distr, mean_scale=1, var_scale=1, covar_scale=1, corr_strength=1, device="cpu"):
    """
    generate a random multivariate gaussian distribution.
    
    Parameters:
    mean_distr : function or numpy.ndarray
        Distribution to sample the mean vector from, or the mean vector itself
    mean_scale : float
        Scale factor for the mean vector
    var_scale : float
        Scale factor for the variance
    covar_scale : float
        Scale factor for the covariance
    corr_strength : float
        Parameter controlling the correlation strength (used in LKJ distribution)
    device : str
        Device to place tensors on ("cpu" or "cuda")
        
    Returns:
    function : A callable function representing the Gaussian PDF
    """
    if callable(mean_distr):
        μ = mean_distr() * mean_scale
    else:
        μ = mean_distr * mean_scale
    

    μ = torch.tensor(μ, dtype=torch.float32, device=device)
    d = len(μ)
    
  
    
    #generate random orthogonal matrix
    A = torch.randn(d, d, device=device)
    Q, R = torch.linalg.qr(A)
    
    #generate eigenvalues with correlation strength
    eigenvalues = torch.rand(d, device=device) * (1/corr_strength)
    eigenvalues = eigenvalues / eigenvalues.sum() * d
    
    #build correlation matrix
    Σ = Q @ torch.diag(eigenvalues) @ Q.t()
    
    #ensure it's a valid correlation matrix
    diag_sqrt_inv = torch.diag(1.0 / torch.sqrt(torch.diag(Σ)))
    Σ = diag_sqrt_inv @ Σ @ diag_sqrt_inv
    
    # scale variance and covariance
    # iagonal elements (these are variances)
    mask = torch.eye(d, device=device, dtype=torch.bool)
    Σ[mask] *= var_scale
    
    # off-diagonal elements (covariances)
    Σ[~mask] *= covar_scale
    
    # compute inverse and determinant for the PDF calculation
    inverse = torch.inverse(Σ)
    determinant = torch.det(Σ)
    
    #define the Gaussian PDF function
    def gaussian_pdf(x):
        """
        evaluate the Gaussian PDF at points x.
        Parameters:
        x : torch.Tensor or numpy.ndarray
            Points to evaluate the Gaussian PDF at, can be single point or batch
            
        Returns:
        torch.Tensor : Gaussian PDF values at x
        """
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32, device=device)
        
        #handle single point vs batch
        original_shape = x.shape
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Add batch dimension
        
        # compute (x - μ)
        x_centered = x - μ
        
        # compute quadratic form (x - μ)' Σ^-1 (x - μ) for each row
        quad_form = torch.sum(x_centered @ inverse * x_centered, dim=1)
        
        # compute PDF
        norm_const = torch.sqrt((2 * math.pi) ** d * determinant)
        pdf = torch.exp(-0.5 * quad_form) / norm_const
        
        # return in the appropriate shape
        if len(original_shape) == 1:
            return pdf[0]
        return pdf
    
    return gaussian_pdf