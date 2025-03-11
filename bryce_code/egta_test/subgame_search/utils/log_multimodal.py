import numpy as np
import torch
from scipy.special import gammaln
from math import log, factorial

def logmultinomial(*args):
    """
    Compute the log of the multinomial coefficient efficiently.
    
    This computes log(n! / (k_1! * k_2! * ... * k_m!)) where n = sum(k_i)
    
    Parameters:
    *args : tuple of int
        The counts for each category
        
    Returns:
    float : The log of the multinomial coefficient
    """
    if len(args) == 0:
        return 0.0
    
    # Remove zeros since they don't contribute
    counts = [k for k in args if k > 0]
    
    if len(counts) == 0:
        return 0.0
    
    n = sum(counts)
    
    # Use the log-gamma function for large factorials
    log_numerator = gammaln(n + 1)
    log_denominator = sum(gammaln(k + 1) for k in counts)
    
    return log_numerator - log_denominator

def logmultinomial_torch(counts, device="cpu"):
    """
    PyTorch version of logmultinomial
    
    Parameters:
    counts : torch.Tensor
        Tensor of counts for each category
    device : str
        Device to place tensors on
        
    Returns:
    torch.Tensor : The log of the multinomial coefficient
    """
    counts = torch.as_tensor(counts, device=device)
    
    # Remove zeros since they don't contribute
    nonzero_mask = counts > 0
    nonzero_counts = counts[nonzero_mask]
    
    if len(nonzero_counts) == 0:
        return torch.tensor(0.0, device=device)
    
    n = torch.sum(nonzero_counts)
    
    # Use the log-gamma function for large factorials
    log_numerator = torch.lgamma(n + 1)
    log_denominator = torch.sum(torch.lgamma(nonzero_counts + 1))
    
    return log_numerator - log_denominator

def logsumexp(a, axis=None):
    """
    Compute the log of the sum of exponentials of input elements.
    
    Parameters:
    a : array_like
        Input array
    axis : int, optional
        Axis or axes over which the sum is taken
        
    Returns:
    res : ndarray
        The result of log(sum(exp(a))) calculated in a numerically stable way
    """
    a_max = np.max(a, axis=axis, keepdims=True)
    tmp = np.exp(a - a_max)
    s = np.sum(tmp, axis=axis, keepdims=True)
    return a_max + np.log(s)

def logsumexp_torch(a, dim=None):
    """
    PyTorch version of logsumexp
    
    Parameters:
    a : torch.Tensor
        Input tensor
    dim : int, optional
        Dimension over which the sum is taken
        
    Returns:
    torch.Tensor : The result of log(sum(exp(a))) calculated in a numerically stable way
    """
    a_max, _ = torch.max(a, dim=dim, keepdim=True)
    tmp = torch.exp(a - a_max)
    s = torch.sum(tmp, dim=dim, keepdim=True)
    return a_max + torch.log(s) 