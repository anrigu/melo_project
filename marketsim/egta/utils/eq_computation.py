import torch
from egta.utils.simplex_operations import simplex_normalize, uniform_mixture, random_mixture, simplex_projection_sum, simplex_projection



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
        
    if return_history:
        trace = torch.zeros((game.num_actions, iterations+1), device=game.device)
        trace[:, 0] = mix



    for i in range(iterations):
        #rd update is x_i' = x_i * (payoff_i - offset) / sum_j x_j * (payoff_j - offset)
        dev_pays = game.deviation_payoffs(mix) 
        mix = simplex_normalize(mix * (dev_pays - offset)) 
        if return_history:
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

def logged_gain_descent(game, mix, iters=1000, step_size=.001):
    return gain_descent(game, mix, iters, step_size, return_history=True)

def ficticious_play(game, mix, iters=1000, initial_weight=100, return_history=False):
    '''
    run fp play to find Nash.
    inputs:
    game : AbstractGame
        game to find equilibrium for
    mix : torch.Tensor
        initial mixture (will be ignored if initial_weight > 0)
    iters : int
        number of iterations to run
    initial_weight : float
        weight to give initial mixture (0 to use provided mix)
    return_history : bool
        whether to return history of mixtures
        
    returns:
    torch.Tensor : final mixture (or tuple with history if return_history=True)
    '''

    #initialize counts
    if initial_weight > 0:
        #initialize with random mixture
        counts = torch.ones(game.num_actions, device=game.device) * (initial_weight / game.num_actions)
    else:
        #use provided mixture
        if not torch.is_tensor(mix):
            mix = torch.tensor(mix, dtype=torch.float32, device=game.device)
        counts = mix * initial_weight
    
    mix = simplex_normalize(counts)

    if return_history:
        trace = torch.zeros((game.num_actions, iters+1), device=game.device)
        trace[:, 0] = mix

    for i in range(iters):
        dev_payoffs = game.deviation_payoffs(mix)
        br = torch.zeros_like(mix)
        br[torch.argmax(dev_payoffs)] = 1.0

        counts = counts + br

        mix = simplex_normalize(counts)

        if return_history:
                trace[:, i+1] = mix
    
    if return_history:
        return mix, trace
    return mix

def logged_fictitious_play(game, mix, iters=1000, initial_weight=100):
    return ficticious_play(game, mix, iters, initial_weight, return_history=True)


def iterated_better_response(game, mix, iters=1000, step_size=1e-6, return_history=False):
    """
    runs iterated br to find Nash.
    
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

    if not torch.is_tensor(mix):
        mix = torch.tensor(mix, dtype=torch.float32, device=game.device)
    
    if isinstance(step_size, (int, float)):
        step_size = torch.ones(iters, device=game.device) * step_size

    if return_history:
        trace = torch.zeros((game.num_actions, iters+1), device=game.device)
        trace[:, 0] = mix

    for i in range(iters):
        mix = game.better_response(mix, scale_factor=step_size[i])
        if return_history:
            trace[:, i+1] = mix

    if return_history:
        return mix, trace
    return mix
    
def logged_iterated_better_response(game, mix, iters=1000, step_size=1e-6):
    return iterated_better_response(game, mix, iters, step_size, return_history=True)

def batch_nash(nash_func, game, mixtures, batch_size, **kwargs):
    """
    run Nash finding on batches of initial mixtures.
    
    inputs:
    nash_func : function
        nash-finding function to use
    game : AbstractGame
        game to find equilibrium for
    mixtures : torch.Tensor
        initial mixtures, shape (num_actions, num_mixtures)
    batch_size : int
        batch size for processing
    **kwargs : dict
        additional parameters to pass to nash_func       
    returns:
    torch.Tensor : equilibrium candidates
    """
    if not torch.is_tensor(mixtures):
        mixtures = torch.tensor(mixtures, dtype=torch.float32, device=game.device)
    
    eq_candidates = mixtures.clone()
    num_mixtures = mixtures.shape[1]
    for i in range(0, num_mixtures, batch_size):
        start_idx = i
        end_idx = min(i + batch_size, num_mixtures)
        batch = mixtures[:, start_idx:end_idx]
        
        # apply nash finding with function passed in
        results = nash_func(game, batch, **kwargs)
        
        if isinstance(results, tuple) and len(results) == 2:
            eq_candidates[:, start_idx:end_idx] = results[0]
        else:
            eq_candidates[:, start_idx:end_idx] = results
    
    return eq_candidates


def find_equilibria(game, method='replicator_dynamics', num_restarts=10, logging = False, **kwargs):
    """
    find Nash using multiple random restarts.
    
    inputs:
    game : AbstractGame
        game to find equilibrium for
    method : str
        method to use: 'replicator_dynamics', 'gain_descent', 
                       'fictitious_play', or 'iterated_better_response'
    num_restarts : int
        number of random initial mixtures to try
    **kwargs : dict
        additional parameters to pass to the method
        
    returns:
    tuple : (best_mixture, all_mixtures, all_regrets)
    """
    mixtures = torch.zeros((game.num_actions, num_restarts), device=game.device)
    mixtures[:, 0] = uniform_mixture(game.num_actions)
    method_map = {}
    if num_restarts > 1:
        mixtures[:, 1:] = random_mixture(game.num_actions, num_restarts-1, device=game.device)
    if logging:
        method_map = {
            'replicator_dynamics': logged_replicator_dynamics,
            'gain_descent': logged_gain_descent,
            'fictitious_play': logged_fictitious_play,
            'iterated_better_response': logged_iterated_better_response
        }
    else:
        method_map = {
            'replicator_dynamics': replicator_dynamics,
            'gain_descent': gain_descent,
            'fictitious_play': ficticious_play,
            'iterated_better_response': iterated_better_response
        }
    
    if method not in method_map:
        raise ValueError(f"Unknown method: {method}. Choose from: {list(method_map.keys())}")
    
    nash_func = method_map[method]
    
    #run the method with batches
    eq_candidates = batch_nash(nash_func, game, mixtures, batch_size=1, **kwargs)
    
    #calculate regrets
    regrets = game.regret(eq_candidates)
    
    #find the best mixture (lowest regret)
    best_idx = torch.argmin(regrets)
    best_mixture = eq_candidates[:, best_idx]
    
    return best_mixture, eq_candidates, regrets

def logged_find_equilibria(game, method='replicator_dynamics', num_restarts=10, logging = False, **kwargs):
    return find_equilibria(game, method='replicator_dynamics', num_restarts=10, logging = True, **kwargs)




