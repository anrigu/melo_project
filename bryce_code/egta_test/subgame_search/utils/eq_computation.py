import torch
from .simplex_operations import simplex_normalize, uniform_mixture, random_mixture, simplex_projection_sum, simplex_projection



def replicator_dynamics(game, mix, iterations=1000, offset=0, return_history=False):

    """
    run replicator dynamics to find Nash equilibrium.
    inputs:
    game : AbstractGame
        game to find equilibrium for
    mix : torch.Tensor
        initial mixture (if None, use uniform)
    iters : int
        number of iterations to run
    offset : float
        offset for payoffs for better convergence
    return_history : bool
        whether to return history of mixtures 
    returns:
    torch.Tensor : Final mixture (or tuple with history if return_history=True)
    """

    if mix is None:
        mix = torch.ones(game.num_actions, device=game.device) / game.num_actions
    elif not torch.is_tensor(mix):
        mix = torch.tensor(mix, dtype=torch.float32, device=game.device)

    if not torch.is_tensor(mix):
        mix = torch.tensor(mix, dtype=torch.float32, device=game.device)

    is_batch = len(mix.shape) > 1
 
    if return_history:
        if is_batch:
            batch_size = mix.shape[1]
            trace = torch.zeros((game.num_actions, batch_size, iterations+1), device=game.device)
            trace[:, :, 0] = mix
        else:
            trace = torch.zeros((game.num_actions, iterations+1), device=game.device)
            trace[:, 0] = mix



    for i in range(iterations):
        #rd update is x_i' = x_i * (payoff_i - offset) / sum_j x_j * (payoff_j - offset)
        dev_pays = game.deviation_payoffs(mix) 
        mix = simplex_normalize(mix * (dev_pays - offset)) 
        if return_history:
            if is_batch:
                trace[:, :, i+1] = mix
            else:
                trace[:, i+1] = mix
        
    if return_history:
        return mix, trace
    return mix

def logged_replicator_dynamics(game, mix, iters=1000, offset=0):
    return replicator_dynamics(game, mix, iters, offset, return_history=True)

def gain_descent(game, mix, iterations=1000, step_size=.001, return_history=False):
    """
    run gain descent to find Nash equilibrium.
    
    inputs:
    game : AbstractGame
        game to find equilibrium for
    mix : torch.Tensor
        initial mixture
    iters : int
        number of iterations to run
    step_size : float or torch.Tensor
        step size or sequence of step sizes
    return_history : bool
        whether to return history of mixtures
        
    returns:
    torch.Tensor : final mixture (or tuple with history if return_history=True)
    """
    if mix is None:
        mix = torch.ones(game.num_actions, device=game.device) / game.num_actions

    elif not torch.is_tensor(mix):
        mix = torch.tensor(mix, dtype=torch.float32, device=game.device)
    
    if isinstance(step_size, (int, float)):
        step_size = torch.ones(iterations, device=game.device) * step_size

    if return_history:
        trace = torch.zeros((game.num_actions, iterations+1), device=game.device)
        trace[:, 0] = mix
    
    for i in range(iterations):
        #grad descent step with projection back to simplex
        grads = game.gain_gradients(mix)
        mix = simplex_projection_sum(mix - step_size[i] * grads, device=game.device)

        if return_history:
            trace[:, i+1] = mix
    
    if return_history:
        return mix, trace
    return mix

def logged_gain_descent(game, mix, iterations=1000, step_size=.001):
    return gain_descent(game, mix, iterations, step_size, return_history=True)

def iterated_better_response(game, mix, iterations=1000, scale_factor=1.0, return_history=False):
    """
    run iterated better response to find Nash equilibrium.
    
    inputs:
    game : AbstractGame
        game to find equilibrium for
    mix : torch.Tensor
        initial mixture
    iters : int
        number of iterations to run
    scale_factor : float
        scale factor for better response
    return_history : bool
        whether to return history of mixtures
        
    returns:
    torch.Tensor : final mixture (or tuple with history if return_history=True)
    """
    if mix is None:
        mix = torch.ones(game.num_actions, device=game.device) / game.num_actions
    elif not torch.is_tensor(mix):
        mix = torch.tensor(mix, dtype=torch.float32, device=game.device)
    
    if return_history:
        trace = torch.zeros((game.num_actions, iterations+1), device=game.device)
        trace[:, 0] = mix
    
    for i in range(iterations):
        mix = game.better_response(mix, scale_factor)
        if return_history:
            trace[:, i+1] = mix
    
    if return_history:
        return mix, trace
    return mix

def logged_better_response(game, mix, iterations=1000, scale_factor=1.0):
    return iterated_better_response(game, mix, iterations, scale_factor, return_history=True)

def ficticious_play(game, mix, iterations=1000, step_size=None, return_history=False):
    """
    run ficticious play to find Nash equilibrium.
    
    inputs:
    game : AbstractGame
        game to find equilibrium for
    mix : torch.Tensor
        initial mixture
    iters : int
        number of iterations to run
    step_size : float or None
        step size for updates; if None, use 1/(t+1)
    return_history : bool
        whether to return history of mixtures
        
    returns:
    torch.Tensor : final mixture (or tuple with history if return_history=True)
    """
    if mix is None:
        mix = torch.ones(game.num_actions, device=game.device) / game.num_actions
    elif not torch.is_tensor(mix):
        mix = torch.tensor(mix, dtype=torch.float32, device=game.device)
    
    # Initialize the beliefs
    beliefs = mix.clone()
    
    if return_history:
        trace = torch.zeros((game.num_actions, iterations+1), device=game.device)
        trace[:, 0] = mix
    
    for i in range(iterations):
        # Compute best response to beliefs
        dev_pays = game.deviation_payoffs(beliefs)
        best_resp_idx = torch.argmax(dev_pays).item()
        best_resp = torch.zeros_like(mix)
        best_resp[best_resp_idx] = 1.0
        
        # Update beliefs
        if step_size is None:
            beliefs = beliefs * (i + 1) / (i + 2) + best_resp / (i + 2)
        else:
            beliefs = beliefs * (1 - step_size) + best_resp * step_size
        
        # Update mix (equal to beliefs in fictitious play)
        mix = beliefs.clone()
        
        if return_history:
            trace[:, i+1] = mix
    
    if return_history:
        return mix, trace
    return mix

def logged_ficticious_play(game, mix, iterations=1000, step_size=None):
    return ficticious_play(game, mix, iterations, step_size, return_history=True)

def find_equilibria(game, method="replicator_dynamics", num_restarts=1, logging=False, **kwargs):
    """
    find multiple equilibria using different initial conditions.
    
    inputs:
    game : AbstractGame
        game to find equilibrium for
    method : str
        equilibrium-finding method to use
    num_restarts : int
        number of random initial mixtures to try
    logging : bool
        whether to log the algorithm's progress
    **kwargs : dict
        additional arguments to pass to the equilibrium-finding method
        
    returns:
    torch.Tensor : best equilibrium found
    float : regret of the best equilibrium
    torch.Tensor : all equilibria found
    """
    best_mixture = None
    best_regret = float('inf')
    all_mixtures = []
    
    # Choose equilibrium-finding function
    if method == "replicator_dynamics":
        eq_func = logged_replicator_dynamics if logging else replicator_dynamics
    elif method == "gain_descent":
        eq_func = logged_gain_descent if logging else gain_descent
    elif method == "iterated_better_response":
        eq_func = logged_better_response if logging else iterated_better_response
    elif method == "ficticious_play":
        eq_func = logged_ficticious_play if logging else ficticious_play
    else:
        raise ValueError(f"Unknown equilibrium-finding method: {method}")
    
    # Try multiple random initial mixtures
    for i in range(num_restarts):
        if i == 0:
            # First try uniform mixture
            initial_mix = uniform_mixture(game.num_actions, device=game.device)
        else:
            # Then try random mixtures
            initial_mix = random_mixture(game.num_actions, device=game.device)
        
        # Find equilibrium
        if logging:
            mixture, _ = eq_func(game, initial_mix, **kwargs)
        else:
            mixture = eq_func(game, initial_mix, **kwargs)
        
        # Compute regret
        regret = game.regret(mixture).item()
        
        # Store all mixtures
        all_mixtures.append(mixture)
        
        # Update best mixture
        if regret < best_regret:
            best_regret = regret
            best_mixture = mixture
    
    return best_mixture, best_regret, torch.stack(all_mixtures, dim=1) 