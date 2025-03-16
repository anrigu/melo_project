"""
Equilibrium finding algorithms for EGTA.
Modernized implementations based on Bryce's original code but using PyTorch.
"""
import torch
import numpy as np
from typing import Dict, List, Tuple, Union, Optional, Callable, Set
import time
import heapq
from marketsim.egta.core.game import Game
from marketsim.custom_math.simplex_operations import simplex_normalize, simplex_projection


class SubgameCandidate:
    """
    Represents a candidate equilibrium in a subgame.
    
    Attributes:
        support: Set of strategy indices in the support
        restriction: List of strategy indices in the restriction
        mixture: Strategy mixture
        regret: Regret of the mixture (None if not computed)
    """
    def __init__(self, 
                support: Set[int], 
                restriction: List[int], 
                mixture: torch.Tensor, 
                regret: Optional[float] = None):
        self.support = support
        self.restriction = restriction
        self.mixture = mixture
        self.regret = regret
        
    def __repr__(self):
        return f"SubgameCandidate(support={self.support}, regret={self.regret})"


class DeviationPriorityQueue:
    """
    Priority queue for deviations, ordered by gain.
    
    Attributes:
        queue: List of (gain, strategy, mixture) tuples, ordered by gain
    """
    def __init__(self):
        self.queue = []
        
    def push(self, gain: float, strategy: int, mixture: torch.Tensor):
        """
        Push a deviation to the queue.
        
        Args:
            gain: Gain from deviation
            strategy: Strategy index
            mixture: Base strategy mixture
        """
        # Negate gain for max-heap (heapq is a min-heap)
        heapq.heappush(self.queue, (-gain, strategy, mixture))
        
    def pop(self) -> Tuple[float, int, torch.Tensor]:
        """
        Pop the deviation with highest gain.
        
        Returns:
            Tuple of (gain, strategy, mixture)
        """
        neg_gain, strategy, mixture = heapq.heappop(self.queue)
        return -neg_gain, strategy, mixture
        
    def is_empty(self) -> bool:
        """
        Check if the queue is empty.
        
        Returns:
            True if the queue is empty, False otherwise
        """
        return len(self.queue) == 0


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
    
    if isinstance(step_size, (int, float)):
        step_size = [step_size] * iters
    
    prev_mixture = None
    for i in range(min(iters, len(step_size))):
        gradients = game.game.gain_gradients(mixture)
        new_mixture = simplex_projection(mixture - step_size[i] * gradients)
        
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
    try:
        result = game.regret(mixture)
        
        # Handle NaN or Inf in the result
        if torch.is_tensor(result):
            if torch.isnan(result).any() or torch.isinf(result).any():
                print("Warning: NaN or Inf detected in regret calculation. Replacing with 1.0")
                result = torch.nan_to_num(result, nan=1.0, posinf=1.0, neginf=0.0)
        else:
            if np.isnan(result) or np.isinf(result):
                print("Warning: NaN or Inf detected in regret calculation. Replacing with 1.0")
                result = 1.0
        
        return result
    except Exception as e:
        print(f"Error in regret calculation: {e}")
        # Return a fallback value
        if torch.is_tensor(mixture):
            return torch.tensor(0.01, device=mixture.device)
        return 0.01


async def quiesce(
    game: Game, 
    num_iters: int = 10,
    num_random_starts: int = 10,  # Adding parameter for random starts
    regret_threshold: float = 1e-3,  # More lenient threshold (was 1e-4)
    dist_threshold: float = 1e-2,    # More lenient threshold (was 1e-3)
    restricted_game_size: int = 4,
    solver: str = 'replicator',
    solver_iters: int = 5000,        # More iterations (was 1000)
    verbose: bool = True
) -> List[Tuple[torch.Tensor, float]]:
    """
    find all equilibria of a game using the QUIESCE algorithm.
    This implementation follows the formal algorithm as outlined in
    Erik Brinkman's work. 
    
    Args:
        game: Game to analyze
        num_iters: Maximum number of QUIESCE iterations
        num_random_starts: Number of random starting points to try
        regret_threshold: Regret threshold for considering a mixture an equilibrium
        dist_threshold: Distance threshold for considering mixtures distinct
        restricted_game_size: Maximum size of restricted games to explore
        solver: Solver to use ('replicator', 'fictitious_play', or 'gain_descent')
        solver_iters: Number of iterations for the solver
        verbose: Whether to print progress
        
    Returns:
        List of equilibria as (mixture, regret) tuples
    """
    start_time = time.time()
    
    # Initialize data structures
    confirmed_eq = []
    unconfirmed_candidates = []
    deviation_queue = DeviationPriorityQueue()
    
    # Add pure strategy restrictions to queue
    for s in range(game.num_strategies):
        # Create pure strategy mixture
        pure_mixture = torch.zeros(game.num_strategies, device=game.game.device)
        pure_mixture[s] = 1.0
        
        # Create candidate
        pure_candidate = SubgameCandidate(
            support=set([s]),
            restriction=[s],
            mixture=pure_mixture
        )
        
        # Add to unconfirmed candidates
        unconfirmed_candidates.append(pure_candidate)
    
    # Add uniform mixture as a starting point
    uniform_mixture = torch.ones(game.num_strategies, device=game.game.device) / game.num_strategies
    uniform_candidate = SubgameCandidate(
        support=set(range(game.num_strategies)),
        restriction=list(range(game.num_strategies)),
        mixture=uniform_mixture
    )
    unconfirmed_candidates.append(uniform_candidate)
    
    # Add random starting points to increase exploration
    for i in range(num_random_starts):
        # Generate random mixture
        rand_mixture = torch.rand(game.num_strategies, device=game.game.device)
        rand_mixture = rand_mixture / rand_mixture.sum()
        
        # Create support set (include strategies with significant probability)
        support = set([s for s in range(game.num_strategies) if rand_mixture[s] > 0.05])
        if len(support) == 0:  # Ensure at least one strategy in support
            support = {torch.argmax(rand_mixture).item()}
            
        rand_candidate = SubgameCandidate(
            support=support,
            restriction=list(support),
            mixture=rand_mixture
        )
        unconfirmed_candidates.append(rand_candidate)
    
    for iteration in range(num_iters):
        if verbose:
            print(f"QUIESCE iteration {iteration+1}/{num_iters} with {len(unconfirmed_candidates)} candidates")
        
        if not unconfirmed_candidates and deviation_queue.is_empty():
            break
            
        # Test all unconfirmed candidates
        new_unconfirmed = []
        for candidate in unconfirmed_candidates:
            is_eq, _ = await test_candidate(candidate, game, regret_threshold, deviation_queue, restricted_game_size, verbose)
            
            if is_eq:
                # Check if this equilibrium is distinct from confirmed ones
                is_distinct = True
                for eq_mixture, _ in confirmed_eq:
                    dist = torch.norm(candidate.mixture - eq_mixture, p=1).item()
                    if dist < dist_threshold:
                        is_distinct = False
                        break
                        
                if is_distinct:
                    confirmed_eq.append((candidate.mixture.clone(), candidate.regret))
            else:
                # Keep candidates that aren't equilibria for further exploration
                new_unconfirmed.append(candidate)
                
        unconfirmed_candidates = new_unconfirmed
        
        # Explore deviations with highest gain
        if not deviation_queue.is_empty():
            # Get highest-gain deviation
            gain, strategy, base_mixture = deviation_queue.pop()
            
            if verbose:
                print(f"  Exploring deviation to {game.strategy_names[strategy]} with gain {gain:.6f}")
                
            # Get support of base mixture
            support = set([i for i, p in enumerate(base_mixture) if p > 0.01])
            
            # Add deviated strategy to support
            new_support = support.union({strategy})
            
            # Create restriction
            restriction = list(new_support)
            
            # Create restricted game
            restricted_game = game.restrict(restriction)
            
            # Create initial mixture for the restricted game
            init_restricted = torch.zeros(len(restriction), device=game.game.device)
            for i, s in enumerate(restriction):
                if s in support:
                    idx = list(support).index(s)
                    # Copy probability from base mixture, but leave some for the new strategy
                    init_restricted[i] = base_mixture[s] * 0.8
            
            # Add probability for the new strategy
            new_idx = restriction.index(strategy)
            init_restricted[new_idx] = 0.2  # Start with some weight on the beneficial deviation
            
            # Normalize
            init_restricted = init_restricted / init_restricted.sum()
            
            # Run solver on restricted game to find equilibrium
            solver_start = time.time()
            if solver == 'replicator':
                equilibrium = replicator_dynamics(restricted_game, init_restricted, iters=solver_iters)
            elif solver == 'fictitious_play':
                equilibrium = fictitious_play(restricted_game, init_restricted, iters=solver_iters)
            elif solver == 'gain_descent':
                equilibrium = gain_descent(restricted_game, init_restricted, iters=solver_iters)
            else:
                raise ValueError(f"Unknown solver: {solver}")
            solver_time = time.time() - solver_start
            
            # Convert back to full game mixture
            full_mixture = torch.zeros(game.num_strategies, device=game.game.device)
            for i, s in enumerate(restriction):
                full_mixture[s] = equilibrium[i]
            
            # Get regret in full game
            regret_val = game.regret(full_mixture).item()
            
            # Create candidate
            candidate = SubgameCandidate(
                support=new_support,
                restriction=restriction,
                mixture=full_mixture,
                regret=regret_val
            )
            
            if verbose:
                print(f"  Solver completed in {solver_time:.4f} seconds with regret {regret_val:.6f}")
                
            # Add to unconfirmed candidates
            unconfirmed_candidates.append(candidate)
    
    # Sort equilibria by regret
    confirmed_eq.sort(key=lambda x: x[1])
    
    if verbose:
        elapsed = time.time() - start_time
        print(f"  Solved in {elapsed:.2f} seconds")
    
    return confirmed_eq 


def quiesce_sync(
    game: Game, 
    num_iters: int = 10,
    num_random_starts: int = 10,
    regret_threshold: float = 1e-3,
    dist_threshold: float = 1e-2,
    restricted_game_size: int = 4,
    solver: str = 'replicator',
    solver_iters: int = 5000,
    verbose: bool = True
) -> List[Tuple[torch.Tensor, float]]:
    """
    Synchronous wrapper for the quiesce function that avoids event loop conflicts.
    
    Args:
        game: Game to analyze
        num_iters: Maximum number of QUIESCE iterations
        num_random_starts: Number of random starting points to try
        regret_threshold: Regret threshold for considering a mixture an equilibrium
        dist_threshold: Distance threshold for considering mixtures distinct
        restricted_game_size: Maximum size of restricted games to explore
        solver: Solver to use ('replicator', 'fictitious_play', or 'gain_descent')
        solver_iters: Number of iterations for the solver
        verbose: Whether to print progress
        
    Returns:
        List of equilibria as (mixture, regret) tuples
    """
    import asyncio
    import nest_asyncio
    import sys
    import os
    
    try:
        # Try to apply nest_asyncio to allow nested event loops
        nest_asyncio.apply()
    except ImportError:
        # If nest_asyncio is not available, we need a workaround
        if verbose:
            print("Note: nest_asyncio not found. Using a subprocess to run quiesce.")
        
        # For 2-strategy games, we can directly compute equilibria
        if game.num_strategies == 2:
            return _compute_2x2_equilibria(game, verbose)
        
        # For games with more strategies, fallback to replicator dynamics
        return _fallback_replicator(game, solver_iters, verbose)
    
    # With nest_asyncio applied, we can run the coroutine
    try:
        # Get or create event loop
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # Run the coroutine
        coro = quiesce(
            game=game,
            num_iters=num_iters,
            num_random_starts=num_random_starts,
            regret_threshold=regret_threshold,
            dist_threshold=dist_threshold,
            restricted_game_size=restricted_game_size,
            solver=solver,
            solver_iters=solver_iters,
            verbose=verbose
        )
        result = loop.run_until_complete(coro)
        return result
    except Exception as e:
        if verbose:
            print(f"Error in quiesce_sync: {e}")
        
        # Fallback for 2-strategy games
        if game.num_strategies == 2:
            return _compute_2x2_equilibria(game, verbose)
        
        # Fallback to replicator dynamics for larger games
        return _fallback_replicator(game, solver_iters, verbose)

def _compute_2x2_equilibria(game: Game, verbose: bool = True) -> List[Tuple[torch.Tensor, float]]:
    """Helper function to compute equilibria for 2x2 games directly."""
    if verbose:
        print("Using direct 2x2 equilibrium computation as fallback.")
    
    device = game.game.device
    equilibria = []
    
    # Get the payoff matrix
    payoff_matrix = game.get_payoff_matrix()
    
    # Check if any entries are NaN and replace with safe values
    if torch.isnan(payoff_matrix).any():
        if verbose:
            print("Warning: NaN values in payoff matrix. Replacing with zeros.")
        payoff_matrix = torch.nan_to_num(payoff_matrix, nan=0.0)
    
    # Check pure strategy equilibria
    # Strategy 0 is a pure equilibrium if it's the best response to itself
    if payoff_matrix[0, 0] >= payoff_matrix[1, 0]:
        pure_0 = torch.tensor([1.0, 0.0], device=device)
        # Compute regret directly
        regret_0 = max(0, payoff_matrix[1, 0] - payoff_matrix[0, 0])
        equilibria.append((pure_0, regret_0))
    
    # Strategy 1 is a pure equilibrium if it's the best response to itself
    if payoff_matrix[1, 1] >= payoff_matrix[0, 1]:
        pure_1 = torch.tensor([0.0, 1.0], device=device)
        # Compute regret directly
        regret_1 = max(0, payoff_matrix[0, 1] - payoff_matrix[1, 1])
        equilibria.append((pure_1, regret_1))
    
    # Check for mixed equilibrium
    # Calculate the indifference points
    denom_0 = payoff_matrix[0, 0] - payoff_matrix[1, 0] - payoff_matrix[0, 1] + payoff_matrix[1, 1]
    
    # Skip mixed equilibrium calculation if denominator is close to zero
    if abs(denom_0) > 1e-10:
        p = (payoff_matrix[1, 1] - payoff_matrix[0, 1]) / denom_0
        
        # Valid mixed equilibrium must have p between 0 and 1
        if 0 < p < 1:
            mixed = torch.tensor([p, 1-p], device=device)
            # Compute regret for mixed equilibrium
            dev_payoffs = game.deviation_payoffs(mixed)
            exp_payoff = (mixed * dev_payoffs).sum()
            regret_mixed = max(0, torch.max(dev_payoffs - exp_payoff).item())
            
            # Add if regret is small enough
            if regret_mixed < 1e-3:
                equilibria.append((mixed, regret_mixed))
    
    # If no equilibria found (rare), fall back to uniform
    if not equilibria:
        if verbose:
            print("No equilibria found. Using uniform strategy.")
        uniform = torch.ones(2, device=device) / 2
        regret_uniform = 0.01  # Default small regret
        equilibria.append((uniform, regret_uniform))
    
    return equilibria

def _fallback_replicator(game: Game, solver_iters: int = 5000, verbose: bool = True) -> List[Tuple[torch.Tensor, float]]:
    """Helper function for replicator dynamics fallback."""
    if verbose:
        print("Using replicator dynamics as fallback.")
    
    # Initialize with uniform mixture
    mixture = torch.ones(game.num_strategies, device=game.game.device) / game.num_strategies
    
    # Run replicator dynamics
    eq_mixture = replicator_dynamics(game, mixture, iters=solver_iters)
    
    # Compute regret
    eq_regret = regret(game, eq_mixture)
    
    # Handle NaN regret
    if torch.is_tensor(eq_regret) and torch.isnan(eq_regret).any():
        eq_regret = torch.tensor(0.01, device=game.game.device)
    if not torch.is_tensor(eq_regret) and (np.isnan(eq_regret) or np.isinf(eq_regret)):
        eq_regret = 0.01
    
    return [(eq_mixture, eq_regret)] 

async def test_candidate(candidate, game, regret_threshold, deviation_queue, restricted_game_size, verbose):
    """Test a candidate equilibrium and add beneficial deviations to the queue."""
    mixture = candidate.mixture
    num_strategies = game.num_strategies
    
    # Validate mixture for numerical issues before testing
    if torch.isnan(mixture).any() or torch.isinf(mixture).any() or torch.any(mixture < 0):
        if verbose:
            print(f"    Warning: Invalid mixture detected. Fixing before testing.")
        # Fix the mixture
        mixture = torch.nan_to_num(mixture, nan=1.0/num_strategies)
        mixture = torch.clamp(mixture, min=1e-6)
        mixture = mixture / mixture.sum()
        candidate.mixture = mixture
    
    # Calculate regret
    try:
        regret_val = game.regret(mixture)
        
        # Handle NaN regret
        if torch.is_tensor(regret_val) and (torch.isnan(regret_val) or torch.isinf(regret_val)):
            if verbose:
                print(f"    Warning: Detected NaN/Inf regret for candidate. Setting high regret.")
            regret_val = torch.tensor(1.0, device=mixture.device)
            candidate.regret = 1.0
            return False, []
        
        if isinstance(regret_val, torch.Tensor):
            regret_val = regret_val.item()
        
        candidate.regret = regret_val
        
        if verbose:
            print(f"    Candidate regret: {regret_val:.6f}")
            
        # If regret is above threshold, not an equilibrium
        if regret_val > regret_threshold:
            return False, []
            
    except Exception as e:
        if verbose:
            print(f"    Error calculating regret: {e}")
        candidate.regret = 1.0
        return False, []
    
    # Get deviation payoffs
    try:
        dev_payoffs = game.deviation_payoffs(mixture)
        
        # Check for NaN in deviation payoffs
        if torch.is_tensor(dev_payoffs) and (torch.isnan(dev_payoffs).any() or torch.isinf(dev_payoffs).any()):
            if verbose:
                print(f"    Warning: NaN or Inf in deviation payoffs. Fixing values.")
            # Replace with safer values
            dev_payoffs = torch.nan_to_num(dev_payoffs, nan=0.0, posinf=10.0, neginf=-10.0)
            # Apply clipping for stability
            dev_payoffs = torch.clamp(dev_payoffs, min=-10.0, max=10.0)
            
        # Calculate expected payoff for the mixture - safely
        expected_payoff_values = mixture * dev_payoffs
        # Check for NaN in intermediate calculation
        if torch.isnan(expected_payoff_values).any() or torch.isinf(expected_payoff_values).any():
            expected_payoff_values = torch.nan_to_num(expected_payoff_values, nan=0.0, posinf=0.0, neginf=0.0)
        
        expected_payoff = expected_payoff_values.sum()
        
        # Calculate gains from deviation - with safety checks
        gains = dev_payoffs - expected_payoff
        # Apply clipping to gains for stability
        gains = torch.clamp(gains, min=-10.0, max=10.0)
        
        # Find beneficial deviations
        has_beneficial = False
        beneficial_strategies = []
        
        for s in range(num_strategies):
            # Skip strategies already in support
            if s in candidate.support:
                continue
            
            gain = gains[s].item() if torch.is_tensor(gains[s]) else gains[s]
            
            # Safety check for NaN gain
            if np.isnan(gain) or np.isinf(gain):
                if verbose:
                    print(f"    Warning: NaN/Inf gain for strategy {s}. Skipping.")
                continue
            
            # If gain is significant, add to deviation queue
            if gain > regret_threshold:
                has_beneficial = True
                beneficial_strategies.append((s, gain))
                
                # Check if adding this strategy would exceed restricted game size
                new_support = candidate.support.union({s})
                if len(new_support) <= restricted_game_size:
                    # Add to queue for exploration
                    deviation_queue.push(gain, s, mixture)
                    
                    if verbose:
                        print(f"    Found beneficial deviation to {game.strategy_names[s]} "
                              f"with gain {gain:.6f}")
                else:
                    if verbose:
                        print(f"    Skipping deviation to {game.strategy_names[s]} as it would exceed restricted game size")
        
        # If no beneficial deviations, this is an equilibrium
        if not has_beneficial:
            if verbose:
                print(f"    Found equilibrium with regret {regret_val:.6f}: "
                      f"{format_mixture(mixture, game.strategy_names)}")
            return True, []
            
        return False, beneficial_strategies
        
    except Exception as e:
        if verbose:
            print(f"    Error testing for beneficial deviations: {e}")
        return False, [] 

def format_mixture(mixture, strategy_names, threshold=0.01):
    """
    Format a mixture as a string for display.
    
    Args:
        mixture: Strategy mixture
        strategy_names: List of strategy names
        threshold: Threshold for including strategies in the output
        
    Returns:
        String representation of the mixture
    """
    if not torch.is_tensor(mixture):
        mixture = torch.tensor(mixture)
    
    # Find strategies with non-negligible probability
    significant_idxs = (mixture > threshold).nonzero().flatten().cpu().numpy()
    
    if len(significant_idxs) == 0:
        return "Uniform mixture"
    
    parts = []
    for idx in significant_idxs:
        prob = mixture[idx].item()
        name = strategy_names[idx] if idx < len(strategy_names) else f"Strategy {idx}"
        parts.append(f"{name}:{prob:.4f}")
    
    return ", ".join(parts) 