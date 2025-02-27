import numpy as np
import torch
from scipy.special import comb
from itertools import combinations_with_replacement
from egta.utils.log_multimodal import logmultinomial, logmultinomial_torch
'''
original code from Bryce here: 
https://github.com/Davidson-Game-Theory-Research/gameanalysis.jl/blob/master/SimplexOperations.jl
'''
def uniform_mixture(num_actions):
    '''
    This returns the cener point of the simplex
    '''

    return torch.ones(num_actions) / num_actions

def random_mixture(num_actions, num_mixtures, alpha=1.0, device="cpu"):
    '''
    This returns random points on the simplex
    
    Parameters:
    num_actions : int
        Number of dimensions in the simplex
    num_mixtures : int
        Number of random points to generate
    alpha : float or list
        Concentration parameter for Dirichlet distribution
    device : str
        Device to place the tensor on ("cpu" or "cuda")
    Returns: 
    torch.Tensor : Tensor of shape (num_actions, num_mixtures) with random points on the simplex
    '''
    if isinstance(alpha, (list, tuple, np.ndarray)):
        #i f alpha is an array, use it directly as concentration parameters
        alpha_tensor = torch.tensor(alpha, device=device)
        return torch.distributions.Dirichlet(alpha_tensor).sample((num_mixtures,)).T
    else:
        #use uniform concentration parameter
        alpha_tensor = torch.ones(num_actions, device=device) * alpha
        return torch.distributions.Dirichlet(alpha_tensor).sample((num_mixtures,)).T
    
def sample_profile(num_players, mixture, device="cpu"):
    '''
    This samples a profile from a given mixture
    '''
    mixture = torch.as_tensor(mixture, device=device)
    return torch.distributions.Multinomial(num_players, probs=mixture).sample()

def sample_profiles(num_players, mixtures, device="cpu"):
    '''
    Sample profiles from multiple mixtures
    Parameters:
    num_players : int
        Number of players in the game
    mixtures : torch.Tensor
        Matrix where each column is a mixture
    device : str
        Device to place the tensor on
    Returns:
    torch.Tensor : Matrix where each column is a sampled profile
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
    normalize to the probability simplex
    returns normalized pytorch tensor
    '''
    d = torch.as_tensor(d)
    d = torch.clamp(d, min=epsilon)
    return d / torch.sum(d)


def profile_rankings(profile):
    '''
    ranking algorithm from Bryce's paper with replacement 
    used to determine the index of the profile in the configuration table
    '''
    profile = torch.as_tensor(profile)
    num_actions = len(profile)
    num_opponents = torch.sum(profile) 
    preceding_profiles = 0

    for a in range(num_actions):
        num_opponents -= profile[a] 
        stars = num_opponents - 1
        bars = num_actions + 1 - a
        if stars >= 0 and bars > 0:
            preceding_profs += torch.special.binom(stars + bars - 1, stars)
    
    return int(preceding_profs + 1)

def mixture_grid(num_actions, points_per_dim, device="cpu"):

    '''
    creates a grid equally spaced points throughout the simplex
    Parameters:

    num_actions : int
        Number of dimensions in the simplex
    points_per_dim : int
        Number of points along each dimension
    device : str
        Device to place the tensor on

    Returns:
    torch.Tensor : Matrix where each column is a point on the grid
    '''
    num_mixtures = comb(num_actions - 1, points_per_dim - 1, exact = True)
    mixtures = torch.zeros((num_actions, num_mixtures), device=device)

    #generate points on unit simplex using stars and bars algorithm:
    # https://en.wikipedia.org/wiki/Stars_and_bars_(combinatorics)
    for m, config in enumerate(combinations_with_replacement(range(1, num_actions + 1), points_per_dim - 1)):
        #count occurrences of each action in the configuration
        counts = torch.zeros(num_actions, device=device)
        for c in config:
            counts[c - 1] += 1
        
        #normalize to create a point on the simplex
        mix = counts / (points_per_dim - 1)
        mixtures[:, m] = mix
    
    return mixtures

def simplex_projection_sum(y, simplex_sum=1.0, device="cpu"):
    '''
    projects onto a smaller sub-simplex where the dimensions add to simplex_sum
    
    Parameters:
    y : torch.Tensor
        Vector or matrix to project
    simplex_sum : float
        Target sum for the simplex
    device : str
        Device to place the tensor on

    Returns:
    torch.Tensor: projected vector/matrix
    '''
    y = torch.as_tensor(y, device = device)

    if len(y.shape) == 1: #just a single vector
        D = len(y)
        u = torch.sort(y, descending=True)[0]

        cumulative_sum = torch.cumsum(u, dim=0)
        lambda_vector = (simplex_sum - cumulative_sum) / torch.arange(1, D+1, device=device)

        valid_indicies = torch.nonzero(u + lambda_vector > 0)
        if len(valid_indicies) == 0:
            lambda_value = lambda_vector[-1]

        else:
            lambda_value = lambda_vector[valid_indicies[-1]]
        return torch.max(torch.zeros_like(y), y + lambda_value) + (1 - simplex_sum) / D
    
    else: #matrix case
        D = y.shape[0]
        u = torch.sort(y, dim=0, descending=True)[0]
        cumulative_sum = torch.cumsum(u, dim=0)
        lambda_vector = (simplex_sum - cumulative_sum) / torch.arange(1, D+1, 
                                                                   device=device).reshape(-1, 1)
        
        lambda_values = torch.zeros(y.shape[1], device=device)

        for i in range(y.shape[1]):
            valid_indices = torch.nonzero(u[:, i] + lambda_vector[:, i] > 0)

            if len(valid_indices) == 0:
                lambda_values[i] = lambda_vector[-1, i]
            else:
                lambda_values[i] = lambda_vector[valid_indices[-1], i]
        return torch.max(torch.zeros_like(y), y + lambda_values.reshape(1, -1)) + (1 - simplex_sum) / D
    

def simplex_normalize(d, epsilon=1e-10):
    '''
    normalize to the probability simplex
    returns normalized pytorch tensor
    '''
    d = torch.as_tensor(d)
    d = torch.clamp(d, min=epsilon)
    return d / torch.sum(d, dim=0)



def filter_unique(mixtures, max_diff=1e-2, device="cpu"):
    '''
    filter a collection of mixed strategies of mixed strategies
    to remove approximate duplicates 
    keep the first appearance of the group

    Parameters:
    mixtures : torch.Tensor
        Matrix where each column is a mixture
    max_diff : float
        Maximum difference for two mixtures to be considered duplicates
    device : str
        Device to place the tensor on

    Returns:
    torch.Tensor : Matrix with duplicates removed
    '''

    mixtures = torch.as_tensor(mixtures, device=device)
    if mixtures.shape[1] == 0:
        return mixtures
    
    unique_indices = [0]  #keeps first
    
    for m in range(1, mixtures.shape[1]):
        is_unique = True
        for i in unique_indices:
            if torch.max(torch.abs(mixtures[:, i] - mixtures[:, m])) < max_diff:
                is_unique = False
                break
        if is_unique:
            unique_indices.append(m)
    
    return mixtures[:, unique_indices]

def num_payoffs(num_players, num_actions, dev=True):
    '''
    compute the number of payoffs in a symmetric games
    Parameters:
    num_players : int
        Number of players in the game
    num_actions : int
        Number of available actions
    dev : bool
        Whether to count deviation payoffs (True) or pure payoffs (False)
    Returns:
    int : Number of payoffs
    '''
    if dev:
        return int(np.exp(logmultinomial(num_players-1, num_actions-1)) * num_actions)
    else:
        return int(np.exp(logmultinomial(num_players, num_actions-1)) * num_actions)
