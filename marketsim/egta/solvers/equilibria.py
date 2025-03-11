"""
Equilibrium finding algorithms for EGTA.
Modernized implementations based on Bryce's original code but using PyTorch.
"""
import torch
import numpy as np
from typing import Dict, List, Tuple, Union, Optional, Callable
import time
from marketsim.egta.core.game import Game
from marketsim.math.simplex_operations import simplex_normalize


def replicator_dynamics(game: Game, 
                       mixture: torch.Tensor, 
                       iters: int = 1000, 
                       offset: float = 0,
                       converge_threshold: float = 1e-10,
                       return_trace: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
    """
    Find equilibria using replicator dynamics.
    
    Args:
        game: Game to analyze
        mixture: Initial mixture (or batch of mixtures)
        iters: Maximum number of iterations
        offset: Offset used in the update rule
        converge_threshold: Convergence threshold for early stopping
        return_trace: Whether to return the trace of mixtures
        
    Returns:
        Final mixture(s) or tuple of (final_mixture, trace) if return_trace is True
    """
    if not torch.is_tensor(mixture):
        mixture = torch.tensor(mixture, dtype=torch.float32, device=game.game.device)
    
    is_vector = len(mixture.shape) == 1
    if is_vector:
        mixture = mixture.reshape(-1, 1)
    
    trace = [mixture.clone()] if return_trace else None
    
    prev_mixture = None
    for i in range(iters):
        payoffs = game.deviation_payoffs(mixture)
        shifted_payoffs = payoffs - offset
        new_mixture = simplex_normalize(mixture * shifted_payoffs)
        
        # Check for convergence
        if prev_mixture is not None and torch.max(torch.abs(new_mixture - prev_mixture)) < converge_threshold:
            break
            
        prev_mixture = mixture.clone()
        mixture = new_mixture
        
        if return_trace:
            trace.append(mixture.clone())
    
    if is_vector:
        mixture = mixture.squeeze(1)
    
    if return_trace:
        return mixture, trace
    return mixture


def fictitious_play(game: Game, 
                   mixture: torch.Tensor, 
                   iters: int = 1000, 
                   initial_weight: float = 100,
                   converge_threshold: float = 1e-8,
                   return_trace: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
    """
    Find equilibria using fictitious play.
    
    Args:
        game: Game to analyze
        mixture: Initial mixture (or batch of mixtures)
        iters: Maximum number of iterations
        initial_weight: Initial weight for the counts
        converge_threshold: Convergence threshold for early stopping
        return_trace: Whether to return the trace of mixtures
        
    Returns:
        Final mixture(s) or tuple of (final_mixture, trace) if return_trace is True
    """
    if not torch.is_tensor(mixture):
        mixture = torch.tensor(mixture, dtype=torch.float32, device=game.game.device)
    
    is_vector = len(mixture.shape) == 1
    if is_vector:
        mixture = mixture.reshape(-1, 1)
    
    trace = [mixture.clone()] if return_trace else None
    
    # Initialize counts
    counts = mixture * initial_weight
    
    prev_mixture = None
    for i in range(iters):
        # Find best responses
        br_indices = best_responses(game, mixture)
        
        # Update counts
        for m in range(mixture.shape[1]):
            br_idx = br_indices[:, m].nonzero(as_tuple=True)[0]
            # If multiple best responses, choose one randomly
            if len(br_idx) > 1:
                selected_br = br_idx[torch.randint(0, len(br_idx), (1,))]
                counts[selected_br, m] += 1
            else:
                counts[br_idx, m] += 1
        
        # Normalize to get new mixture
        new_mixture = simplex_normalize(counts)
        
        # Check for convergence
        if prev_mixture is not None and torch.max(torch.abs(new_mixture - prev_mixture)) < converge_threshold:
            break
            
        prev_mixture = mixture.clone()
        mixture = new_mixture
        
        if return_trace:
            trace.append(mixture.clone())
    
    if is_vector:
        mixture = mixture.squeeze(1)
    
    if return_trace:
        return mixture, trace
    return mixture


def gain_descent(game: Game, 
                mixture: torch.Tensor, 
                iters: int = 1000, 
                step_size: Union[float, List[float]] = 1e-6,
                converge_threshold: float = 1e-10,
                return_trace: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
    """
    Find equilibria using gain descent.
    
    Args:
        game: Game to analyze
        mixture: Initial mixture (or batch of mixtures)
        iters: Maximum number of iterations
        step_size: Step size(s) for gradient descent
        converge_threshold: Convergence threshold for early stopping
        return_trace: Whether to return the trace of mixtures
        
    Returns:
        Final mixture(s) or tuple of (final_mixture, trace) if return_trace is True
    """
    if not torch.is_tensor(mixture):
        mixture = torch.tensor(mixture, dtype=torch.float32, device=game.game.device)
    
    is_vector = len(mixture.shape) == 1
    if is_vector:
        mixture = mixture.reshape(-1, 1)
    
    trace = [mixture.clone()] if return_trace else None
    
    # Convert step_size to list if necessary
    if isinstance(step_size, (int, float)):
        step_size = [step_size] * iters
    
    prev_mixture = None
    for i in range(min(iters, len(step_size))):
        gradients = game.game.gain_gradients(mixture)
        new_mixture = simplex_project(mixture - step_size[i] * gradients)
        
        # Check for convergence
        if prev_mixture is not None and torch.max(torch.abs(new_mixture - prev_mixture)) < converge_threshold:
            break
            
        prev_mixture = mixture.clone()
        mixture = new_mixture
        
        if return_trace:
            trace.append(mixture.clone())
    
    if is_vector:
        mixture = mixture.squeeze(1)
    
    if return_trace:
        return mixture, trace
    return mixture


def best_responses(game: Game, mixture: torch.Tensor, atol: float = 1e-8) -> torch.Tensor:
    """
    Find best responses to a mixture.
    
    Args:
        game: Game to analyze
        mixture: Strategy mixture (or batch of mixtures)
        atol: Absolute tolerance for considering strategies as best responses
        
    Returns:
        Boolean tensor indicating which strategies are best responses
    """
    return game.best_responses(mixture, atol)


def regret(game: Game, mixture: torch.Tensor) -> torch.Tensor:
    """
    Calculate regret for a mixture.
    
    Args:
        game: Game to analyze
        mixture: Strategy mixture (or batch of mixtures)
        
    Returns:
        Regret for each mixture
    """
    return game.regret(mixture)


def quiesce(game: Game, 
           num_iters: int = 10, 
           num_random_starts: int = 20,
           solver: str = 'replicator',
           solver_iters: int = 1000,
           regret_threshold: float = 1e-4,
           verbose: bool = True) -> List[Tuple[torch.Tensor, float]]:
    """
    Find all equilibria of a game using the QUIESCE algorithm.
    
    Args:
        game: Game to analyze
        num_iters: Number of QUIESCE iterations
        num_random_starts: Number of random starting points
        solver: Solver to use ('replicator', 'fictitious_play', or 'gain_descent')
        solver_iters: Number of iterations for the solver
        regret_threshold: Regret threshold for considering a mixture an equilibrium
        verbose: Whether to print progress
        
    Returns:
        List of (mixture, regret) tuples, sorted by regret
    """
    device = game.game.device
    
    # Choose solver
    if solver == 'replicator':
        solver_fn = lambda mix: replicator_dynamics(game, mix, iters=solver_iters)
    elif solver == 'fictitious_play':
        solver_fn = lambda mix: fictitious_play(game, mix, iters=solver_iters)
    elif solver == 'gain_descent':
        solver_fn = lambda mix: gain_descent(game, mix, iters=solver_iters)
    else:
        raise ValueError(f"Unknown solver: {solver}")
    
    # Initialize with uniform mixture and random mixtures
    num_strategies = game.num_strategies
    candidates = []
    
    # Add uniform mixture
    uniform = torch.ones(num_strategies, device=device) / num_strategies
    candidates.append(uniform)
    
    # Add pure strategies
    for s in range(num_strategies):
        pure = torch.zeros(num_strategies, device=device)
        pure[s] = 1.0
        candidates.append(pure)
    
    # Add random mixtures
    for _ in range(num_random_starts):
        random_mix = torch.rand(num_strategies, device=device)
        random_mix /= random_mix.sum()
        candidates.append(random_mix)
    
    # Convert to batch for faster processing
    candidates_batch = torch.stack(candidates)
    
    # Main QUIESCE loop
    found_equilibria = []
    for i in range(num_iters):
        if verbose:
            print(f"QUIESCE iteration {i+1}/{num_iters} with {len(candidates)} candidates")
        
        # Solve from all candidate points
        if len(candidates) > 0:
            candidates_batch = torch.stack(candidates)
            start_time = time.time()
            eq_candidates = solver_fn(candidates_batch.t()).t()
            solve_time = time.time() - start_time
            
            if verbose:
                print(f"  Solved in {solve_time:.2f} seconds")
            
            # Calculate regrets
            eq_regrets = regret(game, eq_candidates.t())
            
            # Filter by regret threshold
            low_regret_indices = (eq_regrets < regret_threshold).nonzero(as_tuple=True)[0]
            
            for idx in low_regret_indices:
                eq_mix = eq_candidates[idx]
                eq_reg = eq_regrets[idx].item()
                
                # Check if this equilibrium is already in the list
                is_duplicate = False
                for mix, _ in found_equilibria:
                    if torch.max(torch.abs(mix - eq_mix)) < 1e-3:
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    found_equilibria.append((eq_mix, eq_reg))
                    
                    if verbose:
                        strat_str = ", ".join([f"{game.strategy_names[s]}: {eq_mix[s].item():.4f}" 
                                            for s in range(num_strategies)])
                        print(f"  Found equilibrium with regret {eq_reg:.6f}: {strat_str}")
        
        # Generate new candidates based on best responses
        candidates = []
        for mix, _ in found_equilibria:
            br = best_responses(game, mix)
            br_indices = br.nonzero(as_tuple=True)[0]
            
            for br_idx in br_indices:
                # Create candidate mixing observed eq with pure best response
                for alpha in [0.25, 0.5, 0.75]:
                    new_candidate = mix.clone()
                    new_candidate *= (1 - alpha)
                    new_candidate[br_idx] += alpha
                    candidates.append(new_candidate)
    
    # Sort by regret
    found_equilibria.sort(key=lambda x: x[1])
    
    return found_equilibria 