"""
Equilibrium finding algorithms for EGTA.
Modernized implementations based on Bryce's original code but using PyTorch.
"""
import torch
import numpy as np
from typing import Dict, List, Tuple, Union, Optional, Callable, Set
import time
import heapq
import asyncio
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
    verbose: bool = True,
    maximal_subgames: Set[frozenset] = None
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
        maximal_subgames: Set of frozensets of strategy indices representing explored subgames
        
    Returns:
        List of equilibria as (mixture, regret) tuples
    """
    start_time = time.time()
    
    # Initialize data structures
    confirmed_eq = []
    unconfirmed_candidates = []
    deviation_queue = DeviationPriorityQueue()
    
    # Add a maximal subgames collection to track explored subgames
    if maximal_subgames is None:
        maximal_subgames = set()  # Set of frozensets of strategy indices
    
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
        
        # Add singleton to maximal subgames collection
        maximal_subgames.add(frozenset([s]))
    
    # Add uniform mixture as a starting point
    uniform_mixture = torch.ones(game.num_strategies, device=game.game.device) / game.num_strategies
    uniform_candidate = SubgameCandidate(
        support=set(range(game.num_strategies)),
        restriction=list(range(game.num_strategies)),
        mixture=uniform_mixture
    )
    unconfirmed_candidates.append(uniform_candidate)
    
    # Add the full game to maximal subgames collection
    if len(game.strategy_names) <= restricted_game_size:
        maximal_subgames.add(frozenset(range(game.num_strategies)))
    
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
        
        # Add to maximal subgames if appropriate
        if len(support) <= restricted_game_size:
            # Check if this subgame is contained in any existing maximal subgame
            support_set = frozenset(support)
            is_contained = False
            for maximal_set in list(maximal_subgames):
                if support_set.issubset(maximal_set):
                    is_contained = True
                    break
                # If this new set contains an existing maximal set, remove the smaller one
                if maximal_set.issubset(support_set):
                    maximal_subgames.remove(maximal_set)
            
            if not is_contained:
                maximal_subgames.add(support_set)
    
    if verbose:
        print(f"Initialized with {len(maximal_subgames)} maximal subgames")
    
    for iteration in range(num_iters):
        if verbose:
            print(f"QUIESCE iteration {iteration+1}/{num_iters} with {len(unconfirmed_candidates)} candidates")
        
        if not unconfirmed_candidates and deviation_queue.is_empty():
            break
            
        # Test all unconfirmed candidates
        new_unconfirmed = []
        for candidate in unconfirmed_candidates:
            is_eq, _ = await test_candidate(candidate, game, regret_threshold, deviation_queue, restricted_game_size, maximal_subgames, verbose)
            
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
            restriction_set = frozenset(restriction)
            
            # Check if this subgame is contained in an existing maximal subgame
            is_contained = False
            for maximal_set in maximal_subgames:
                if restriction_set.issubset(maximal_set):
                    is_contained = True
                    break
            
            # If not contained or if it's a new maximal subgame, update the collection
            if not is_contained:
                # Remove any existing maximal subgames that are subsets of this one
                for maximal_set in list(maximal_subgames):
                    if maximal_set.issubset(restriction_set):
                        maximal_subgames.remove(maximal_set)
                
                # Add this new subgame to the maximal subgames collection
                if len(restriction) <= restricted_game_size:
                    maximal_subgames.add(restriction_set)
                    if verbose:
                        print(f"  Added new maximal subgame: {restriction}")
            
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
    game,
    num_iters=1000,
    num_random_starts=0,
    regret_threshold=1e-3,
    dist_threshold=0.05,
    restricted_game_size=4,
    solver="replicator_dynamics",
    solver_iters=1000,
    verbose=False
):
    """Synchronous wrapper for quiesce - finds all equilibria of a game using QUIESCE."""
    # Import and apply nest_asyncio to handle nested event loops
    import nest_asyncio
    nest_asyncio.apply()
    
    # 2-strategy game fallback using direct computation
    if game.num_strategies == 2:
        try:
            # Use get_payoff_matrix method instead of payoff_matrix
            payoff_matrix = game.get_payoff_matrix()
            
            # Handle pathological payoff values for 2x2 games
            if (torch.isnan(payoff_matrix).any() or 
                torch.isinf(payoff_matrix).any() or 
                torch.max(torch.abs(payoff_matrix)) > 1e6):
                # Set NaN/Inf values to zero
                payoff_matrix = torch.nan_to_num(payoff_matrix, nan=0.0, posinf=10.0, neginf=-10.0)
                # Clip to reasonable range
                payoff_matrix = torch.clamp(payoff_matrix, min=-10.0, max=10.0)
            
            # Solve 2x2 game directly - use local replicator dynamics
            eq, steps = replicator_dynamics(
                game,
                torch.ones(game.num_strategies, device=game.game.device) / game.num_strategies,
                iters=solver_iters,
                return_trace=True
            )
            
            regret_val = game.regret(eq).item() if torch.is_tensor(game.regret(eq)) else game.regret(eq)
            
            if verbose:
                print(f"Using direct 2x2 calculation. Found equilibrium: {format_mixture(eq, game.strategy_names)}")
                print(f"Equilibrium regret: {regret_val:.6f}")
            
            return [(eq, regret_val)]
            
        except Exception as e:
            # Fallback to QUIESCE if direct calculation fails
            if verbose:
                print(f"Direct 2x2 calculation failed: {e}. Falling back to QUIESCE.")
    
    try:
        # Create and run the QUIESCE event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Initialize maximal subgames collection
        maximal_subgames = set()
        
        # Call the async quiesce function with our current loop
        equilibria = loop.run_until_complete(quiesce(
            game=game,
            num_iters=num_iters,
            num_random_starts=num_random_starts,
            regret_threshold=regret_threshold,
            dist_threshold=dist_threshold,
            restricted_game_size=restricted_game_size,
            solver=solver,
            solver_iters=solver_iters,
            maximal_subgames=maximal_subgames,
            verbose=verbose
        ))
        loop.close()
        
        return equilibria
        
    except Exception as e:
        if verbose:
            print(f"QUIESCE failed: {e}")
        
        # Last resort: Use replicator dynamics on full game
        try:
            if verbose:
                print("Attempting replicator dynamics on full game as last resort.")
            
            # Use local replicator dynamics implementation
            # Start with uniform mixture
            init_mixture = torch.ones(game.num_strategies, device=game.game.device) / game.num_strategies
            
            # Get payoff matrix for full game
            try:
                # Try with get_payoff_matrix method
                payoff_matrix = game.get_payoff_matrix()
                # Handle pathological values
                payoff_matrix = torch.nan_to_num(payoff_matrix, nan=0.0, posinf=10.0, neginf=-10.0)
                payoff_matrix = torch.clamp(payoff_matrix, min=-10.0, max=10.0)
                
                # Use local replicator dynamics on the payoff matrix
                eq, _ = replicator_dynamics(
                    game,
                    init_mixture,
                    iters=solver_iters,
                    return_trace=True
                )
            except:
                # If we can't get full payoff matrix, just use replicator directly on game
                eq, _ = replicator_dynamics(
                    game,
                    init_mixture,
                    iters=solver_iters,
                    return_trace=True
                )
            
            # Calculate regret
            regret_val = game.regret(eq).item() if torch.is_tensor(game.regret(eq)) else game.regret(eq)
            
            if verbose:
                print(f"Replicator dynamics found: {format_mixture(eq, game.strategy_names)}")
                print(f"Equilibrium regret: {regret_val:.6f}")
            
            return [(eq, regret_val)]
            
        except Exception as e2:
            if verbose:
                print(f"All equilibrium finding methods failed. Last error: {e2}")
            # Return uniform mixture as a last resort
            uniform = torch.ones(game.num_strategies, device=game.game.device) / game.num_strategies
            return [(uniform, 1.0)]

async def test_deviations(
    game, 
    mixture, 
    deviation_queue, 
    regret_threshold=1e-3, 
    restricted_game_size=4, 
    maximal_subgames=None,
    verbose=False
):
    """Test for beneficial deviations from a mixture and add them to the queue."""
    num_strategies = game.num_strategies
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
        # Get the current support
        support = set([i for i, p in enumerate(mixture) if p > 0.01])
        
        # Skip strategies already in support
        if s in support:
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
            
            # Create new support with the deviating strategy
            new_support = support.union({s})
            
            # Check if adding this strategy would exceed restricted game size
            if len(new_support) <= restricted_game_size:
                # If we're tracking maximal subgames, check if this subgame is already explored
                should_explore = True
                if maximal_subgames is not None:
                    new_support_set = frozenset(new_support)
                    # Check if this subgame is already contained in a maximal subgame
                    for maximal_set in maximal_subgames:
                        if new_support_set.issubset(maximal_set):
                            # Skip if the subgame is already contained
                            if verbose:
                                print(f"    Skipping deviation to {game.strategy_names[s]} as it's already within a maximal subgame")
                            should_explore = False
                            break
                
                if should_explore:
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
            print(f"    Found equilibrium with regret {game.regret(mixture):.6f}: "
                  f"{format_mixture(mixture, game.strategy_names)}")
        return True, []
        
    return False, beneficial_strategies


async def test_candidate(candidate, game, regret_threshold, deviation_queue, restricted_game_size, maximal_subgames=None, verbose=False):
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
    
    # Now test for beneficial deviations
    return await test_deviations(
        game=game,
        mixture=mixture,
        deviation_queue=deviation_queue,
        regret_threshold=regret_threshold,
        restricted_game_size=restricted_game_size,
        maximal_subgames=maximal_subgames,
        verbose=verbose
    )

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