import numpy as np
import torch
from .simplex_operations import uniform_mixture

def set_random_seed(seed):
    """
    Set random seeds for reproducibility
    
    Parameters:
    seed : int
        Random seed to use
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def random_uniform(size, device="cpu"):
    """
    Generate uniform random numbers
    
    Parameters:
    size : tuple
        Size of the output tensor
    device : str
        Device to place the tensor on
        
    Returns:
    torch.Tensor : Tensor of uniform random numbers in [0, 1]
    """
    return torch.rand(size, device=device)

def random_normal(size, mean=0.0, std=1.0, device="cpu"):
    """
    Generate normal random numbers
    
    Parameters:
    size : tuple
        Size of the output tensor
    mean : float
        Mean of the normal distribution
    std : float
        Standard deviation of the normal distribution
    device : str
        Device to place the tensor on
        
    Returns:
    torch.Tensor : Tensor of normal random numbers
    """
    return torch.normal(mean=mean, std=std, size=size, device=device)

def sample_random_equilibrium(game, num_tries=100, regret_threshold=1e-3, device="cpu"):
    """
    Try to find an equilibrium by sampling random points
    
    Parameters:
    game : AbstractGame
        Game to find equilibrium for
    num_tries : int
        Number of random points to try
    regret_threshold : float
        Maximum regret to consider a point an equilibrium
    device : str
        Device to place the tensor on
        
    Returns:
    torch.Tensor : Best mixture found
    float : Regret of the best mixture
    """
    best_regret = float('inf')
    best_mixture = uniform_mixture(game.num_actions, device=device)
    
    for _ in range(num_tries):
        # Generate a random mixture
        mixture = torch.distributions.Dirichlet(
            torch.ones(game.num_actions, device=device)
        ).sample()
        
        # Compute regret
        regret = game.regret(mixture).item()
        
        # Update best mixture
        if regret < best_regret:
            best_regret = regret
            best_mixture = mixture
            
        # Early stopping if we found an equilibrium
        if regret < regret_threshold:
            break
    
    return best_mixture, best_regret 