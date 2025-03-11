import numpy as np
import torch
from scipy.special import comb
from itertools import combinations_with_replacement
from .log_multimodal import logmultinomial, logmultinomial_torch
'''
original code from Bryce here: 
https://github.com/Davidson-Game-Theory-Research/gameanalysis.jl/blob/master/SimplexOperations.jl
'''
def uniform_mixture(num_actions, device="cpu"):
    '''
    This returns the cener point of the simplex
    '''

    return torch.ones(num_actions, device=device) / num_actions

def random_mixture(num_actions, alpha=1.0, device="cpu"):
    '''
    this returns random points on the simplex
    Parameters:
    num_actions : int
        dimensions in the simplex
    alpha : float or list
        concentration parameter for Dirichlet distribution #taken from Bryce's code
    device : str
        Device to place the tensor on ("cpu" or "cuda")
    Returns: 
    torch.Tensor : tensor of shape (num_actions,) with random point on the simplex
    '''
    if isinstance(alpha, (list, tuple, np.ndarray)):
        #if alpha is an array, use it directly as concentration parameters
        alpha_tensor = torch.tensor(alpha, device=device)
        return torch.distributions.Dirichlet(alpha_tensor).sample().squeeze()
    else:
        #use uniform concentration parameter
        alpha_tensor = torch.ones(num_actions, device=device) * alpha
        return torch.distributions.Dirichlet(alpha_tensor).sample().squeeze()
    
def sample_profile(num_players, mixture, device="cpu"):
    '''
    this samples a profile from a given mixture
    '''
    mixture = torch.as_tensor(mixture, device=device)
    return torch.distributions.Multinomial(num_players, probs=mixture).sample()

def sample_profiles(num_players, mixtures, device="cpu"):
    '''
    sample profiles from multiple mixtures
    Parameters:
    num_players : int
        number of players in the game
    mixtures : torch.Tensor
        matrix where each column is a mixture
    device : str
        device to place the tensor on
    returns:
    torch.Tensor : matrix where each column is a sampled profile
    '''
    mixtures = torch.as_tensor(mixtures, device=device)
    profiles = torch.zeros_like(mixtures)
    for m in range(mixtures.shape[1]):
        profiles[:, m] = sample_profile(num_players, mixtures[:, m], device)
    return profiles

def num_profiles(num_players, num_actions):
    '''
    calculates the number of distinct profiles in a symmetric game
    
    num_players : int
        Number of players in the game
    num_actions : int
        Number of available actions

    Returns:
    int : Number of distinct profiles
    '''
    return comb(num_players + num_actions - 1, num_actions - 1, exact=True)

def simplex_projection(y, device="cpu"):
    '''
    This projects a vector onto the probability simplex
    This follows what Bryce did in his julia code

    returns a projected pytorch tensor 
    '''
    y = torch.as_tensor(y, device=device)
    if len(y.shape) == 1: #if we have a single vector 
        u = torch.sort(y, descending=True)[0]
        cumulative_sum = torch.cumsum(u, dim=0)
        rho = torch.nonzero(u > (cumulative_sum - 1)/ (torch.arange(1, len(y) + 1, device=device)))[-1]
        theta = (cumulative_sum[rho] - 1) / (rho + 1)
        return torch.max(torch.zeros_like(y), y - theta)
    else: #matrix case, here we project to each column
        result = torch.zeros_like(y)
        for i in range(y.shape[1]):
            result[:,i] = simplex_projection(y[:,i], device=device) #recurse on each column
        return result

def simplex_normalize(d, epsilon=1e-10):
    '''
    Normalize a vector to be on the simplex
    
    Parameters:
    d : torch.Tensor
        Vector or matrix to normalize
    epsilon : float
        Small constant to avoid division by zero
        
    Returns:
    torch.Tensor : Normalized tensor
    '''
    d_sum = torch.sum(d, dim=0, keepdim=True)
    return d / (d_sum + epsilon)

def profile_rankings(profile):
    '''
    Rank strategies based on their counts in the profile
    
    Parameters:
    profile : torch.Tensor
        Profile to rank
        
    Returns:
    list : Ranked indices of strategies
    '''
    return torch.argsort(profile, descending=True).tolist()

def mixture_grid(num_actions, points_per_dim, device="cpu"):
    '''
    Generate a grid of points on the simplex
    
    Parameters:
    num_actions : int
        Number of strategies (dimension of the simplex)
    points_per_dim : int
        Number of points per dimension
    device : str
        Device to place the tensor on
        
    Returns:
    torch.Tensor : Grid of points on the simplex
    '''
    if num_actions == 2:
        # For 2 strategies, just use a 1D grid
        x = torch.linspace(0, 1, points_per_dim, device=device)
        mixtures = torch.stack([x, 1-x], dim=0)
        return mixtures
    elif num_actions == 3:
        # For 3 strategies, use a triangular grid
        mixtures = []
        for i in range(points_per_dim):
            for j in range(points_per_dim - i):
                k = points_per_dim - 1 - i - j
                mixture = torch.tensor([i, j, k], dtype=torch.float32, device=device) / (points_per_dim - 1)
                mixtures.append(mixture)
        return torch.stack(mixtures, dim=1)
    else:
        # For higher dimensions, use a random grid
        mixtures = random_mixture(num_actions, points_per_dim**num_actions, device=device)
        return mixtures

def simplex_projection_sum(y, simplex_sum=1.0, device="cpu"):
    '''
    Project a vector onto the simplex with a specific sum
    
    Parameters:
    y : torch.Tensor
        Vector or matrix to project
    simplex_sum : float
        Target sum for the projected simplex
    device : str
        Device to place the tensor on
        
    Returns:
    torch.Tensor : Projected tensor
    '''
    y = torch.as_tensor(y, device=device)
    
    # Scale the input to have sum = 1, then project and rescale
    if simplex_sum != 1.0:
        y_scaled = y / simplex_sum
        projection = simplex_projection(y_scaled, device=device)
        return projection * simplex_sum
    else:
        return simplex_projection(y, device=device)

def filter_unique(mixtures, max_diff=1e-2, device="cpu"):
    '''
    Filter out similar mixtures to keep only unique ones
    
    Parameters:
    mixtures : torch.Tensor
        Matrix of mixtures (each column is a mixture)
    max_diff : float
        Maximum L1 difference to consider mixtures the same
    device : str
        Device to place the tensor on
        
    Returns:
    torch.Tensor : Filtered mixtures
    '''
    if mixtures.shape[1] <= 1:
        return mixtures
    
    # Convert to device
    mixtures = torch.as_tensor(mixtures, device=device)
    
    # Initialize with the first mixture
    unique_indices = [0]
    
    # Compare each mixture to the ones we've already selected
    for i in range(1, mixtures.shape[1]):
        is_unique = True
        for j in unique_indices:
            # Check L1 distance
            if torch.sum(torch.abs(mixtures[:, i] - mixtures[:, j])) < max_diff:
                is_unique = False
                break
        if is_unique:
            unique_indices.append(i)
    
    return mixtures[:, unique_indices]

def num_payoffs(num_players, num_actions, dev=True):
    '''
    Calculate the number of payoffs in a symmetric game
    
    Parameters:
    num_players : int
        Number of players in the game
    num_actions : int
        Number of available actions
    dev : bool
        Whether to count deviation payoffs only
        
    Returns:
    int : Number of payoffs
    '''
    if dev:
        # For each strategy, we need the payoff when deviating to it
        return num_actions * num_profiles(num_players - 1, num_actions)
    else:
        # Count all payoffs in all profiles
        return num_actions * num_profiles(num_players, num_actions) 