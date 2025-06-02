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


def role_aware_normalize(mixture: torch.Tensor, game: Game) -> torch.Tensor:
    """
    Normalize mixture respecting role symmetric constraints.
    For role symmetric games, each role should sum to 1.
    For symmetric games, falls back to regular simplex normalization.
    
    Args:
        mixture: Strategy mixture tensor
        game: Game instance
        
    Returns:
        Normalized mixture tensor
    """
    if not game.is_role_symmetric:
        return simplex_normalize(mixture)
    
    # Multi-population normalization: each role sums to 1
    normalized = mixture.clone()
    is_batch = len(mixture.shape) == 2
    
    if is_batch:
        # Batch processing: mixture is [num_strategies, batch_size]
        global_idx = 0
        for role_strategies in game.strategy_names_per_role:
            num_role_strats = len(role_strategies)
            if num_role_strats > 0:
                role_slice = slice(global_idx, global_idx + num_role_strats)
                role_part = normalized[role_slice, :]
                
                # Clamp to non-negative
                role_part = torch.clamp(role_part, min=1e-10)
                
                # Normalize each role for each batch element
                role_sums = role_part.sum(dim=0, keepdim=True)
                normalized[role_slice, :] = role_part / torch.clamp(role_sums, min=1e-10)
                
            global_idx += num_role_strats
    else:
        # Single vector: mixture is [num_strategies]
        global_idx = 0
        for role_strategies in game.strategy_names_per_role:
            num_role_strats = len(role_strategies)
            if num_role_strats > 0:
                role_slice = slice(global_idx, global_idx + num_role_strats)
                role_part = normalized[role_slice]
                
                # Clamp to non-negative
                role_part = torch.clamp(role_part, min=1e-10)
                
                # Normalize role to sum to 1
                role_sum = role_part.sum()
                if role_sum > 1e-10:
                    normalized[role_slice] = role_part / role_sum
                else:
                    # If role has zero weight, give uniform distribution
                    normalized[role_slice] = 1.0 / num_role_strats
                    
            global_idx += num_role_strats
    
    return normalized


def multi_population_replicator_step(mixture: torch.Tensor, game: Game, offset: float = 0.0, dt: float = 0.01) -> torch.Tensor:
    """
    Perform one step of multi-population replicator dynamics following the exact mathematical formulation.
    
    For each role r and strategy i:
    dx_{r,i} = x_{r,i} * (u_{r,i}(x) - u_bar_r(x))
    where u_bar_r(x) = sum_j x_{r,j} * u_{r,j}(x)
    
    Args:
        mixture: Current mixture tensor where each role sums to 1
        game: Game instance  
        offset: Offset parameter
        dt: Time step size
        
    Returns:
        Updated mixture tensor
    """
    if not game.is_role_symmetric:
        # Fall back to standard replicator dynamics for symmetric games
        payoffs = game.deviation_payoffs(mixture)
        shifted_payoffs = payoffs - offset
        return simplex_normalize(mixture * shifted_payoffs)
    
    # Multi-population replicator dynamics
    # Step 1: Compute payoffs for all pure strategies using the full joint state
    payoffs = game.deviation_payoffs(mixture)  # u_{r,i}(x) for all strategies
    
    # Step 2: Apply replicator dynamics separately for each role
    deriv_pieces = []
    global_idx = 0
    
    for role_idx, role_strategies in enumerate(game.strategy_names_per_role):
        num_role_strats = len(role_strategies)
        if num_role_strats == 0:
            continue
            
        role_slice = slice(global_idx, global_idx + num_role_strats)
        
        # Extract current role mixture and payoffs: x_r and u_r
        x_r = mixture[role_slice]  # Should sum to 1
        u_r = payoffs[role_slice]  # Payoffs for this role's strategies
        
        # Compute average payoff for this role: phi_r = sum_j x_{r,j} * u_{r,j}
        phi_r = (x_r * u_r).sum()
        
        # Apply replicator equation for this role: dx_r = x_r * (u_r - phi_r - offset)
        dx_r = x_r * (u_r - phi_r - offset)
        
        deriv_pieces.append(dx_r)
        global_idx += num_role_strats
    
    # Step 3: Concatenate derivatives and take Euler step
    deriv_flat = torch.cat(deriv_pieces)
    next_mixture = mixture + dt * deriv_flat
    
    # Step 4: Renormalize each role slice independently to maintain simplex constraints
    global_idx = 0
    for role_idx, role_strategies in enumerate(game.strategy_names_per_role):
        num_role_strats = len(role_strategies)
        if num_role_strats == 0:
            continue
            
        role_slice = slice(global_idx, global_idx + num_role_strats)
        
        # Clamp to avoid negative values
        next_mixture[role_slice] = torch.clamp(next_mixture[role_slice], min=1e-10)
        
        # Renormalize this role's slice to sum to 1
        role_sum = next_mixture[role_slice].sum()
        if role_sum > 1e-10:
            next_mixture[role_slice] = next_mixture[role_slice] / role_sum
        else:
            # If role goes to zero, reset to uniform distribution
            next_mixture[role_slice] = 1.0 / num_role_strats
            
        global_idx += num_role_strats
    
    return next_mixture


def create_role_symmetric_mixture(game: Game, device: str) -> torch.Tensor:
    """
    Create a proper role symmetric mixture where each role sums to 1.
    
    Args:
        game: Game instance
        device: PyTorch device
        
    Returns:
        Role symmetric mixture tensor where each role sums to 1
    """
    if not game.is_role_symmetric:
        return torch.ones(game.num_strategies, device=device) / game.num_strategies
    
    mixture = torch.zeros(game.num_strategies, device=device)
    global_idx = 0
    
    for role_strategies in game.strategy_names_per_role:
        num_role_strats = len(role_strategies)
        if num_role_strats > 0:
            # Each role gets uniform distribution (sums to 1)
            mixture[global_idx:global_idx + num_role_strats] = 1.0 / num_role_strats
        global_idx += num_role_strats
    
    return mixture


def replicator_dynamics(game: Game, 
                       mixture: Optional[torch.Tensor] = None,
                       iters: int = 1000, 
                       offset: float = 0,
                       converge_threshold: float = 1e-10,
                       return_trace: bool = False,
                       use_multiple_starts: bool = True,
                       num_random_starts: int = 50,
                       epsilon: float = 1e-6,
                       similarity_threshold: float = 0.05) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
    """
    Find equilibria using replicator dynamics.
    For role symmetric games, uses multi-population replicator dynamics.
    
    Args:
        game: Game to analyze
        mixture: Initial mixture. For role symmetric games, each role should sum to 1.
        iters: Maximum number of iterations
        offset: Offset used in the update rule
        converge_threshold: Convergence threshold for early stopping
        return_trace: Whether to return the trace of mixtures
        use_multiple_starts: Whether to use multiple starting points
        num_random_starts: Number of random starting points to generate
        epsilon: Small value for numerical stability
        similarity_threshold: Threshold for considering two mixtures as the same equilibrium
    Returns:
        Final mixture(s) or tuple of (final_mixture, trace) if return_trace is True
    """
    device = game.game.device
    num_strategies = game.num_strategies
    
    if mixture is None and use_multiple_starts:
        starting_mixtures = []
        
        # Create proper role symmetric uniform mixture
        uniform_mixture = create_role_symmetric_mixture(game, device)
        starting_mixtures.append(uniform_mixture)
        
        # Create role-aware pure strategy mixtures
        for s in range(num_strategies):
            if game.is_role_symmetric:
                # Find which role this strategy belongs to
                global_idx = 0
                strategy_role_idx = None
                strategy_local_idx = None
                for role_idx, role_strategies in enumerate(game.strategy_names_per_role):
                    if global_idx <= s < global_idx + len(role_strategies):
                        strategy_role_idx = role_idx
                        strategy_local_idx = s - global_idx
                        break
                    global_idx += len(role_strategies)
                
                if strategy_role_idx is not None:
                    # Create mixture where this strategy gets 0.8 and others in same role get remaining 0.2
                    skewed_mixture = create_role_symmetric_mixture(game, device)
                    
                    # Modify the role containing strategy s
                    role_global_start = sum(len(game.strategy_names_per_role[i]) for i in range(strategy_role_idx))
                    role_size = len(game.strategy_names_per_role[strategy_role_idx])
                    
                    if role_size > 1:
                        # Set all strategies in this role to small value, then make target strategy dominant
                        skewed_mixture[role_global_start:role_global_start + role_size] = 0.2 / role_size
                        skewed_mixture[s] = 0.8
                        # Renormalize this role to sum to 1
                        role_sum = skewed_mixture[role_global_start:role_global_start + role_size].sum()
                        skewed_mixture[role_global_start:role_global_start + role_size] /= role_sum
                    
                    starting_mixtures.append(skewed_mixture)
            else:
                # Regular symmetric game - create skewed mixture
                skewed_mixture = torch.ones(num_strategies, device=device) * 0.2 / (num_strategies - 1) if num_strategies > 1 else torch.ones(1, device=device)
                if num_strategies > 1:
                    skewed_mixture[s] = 0.8
                starting_mixtures.append(skewed_mixture)
        
        # Add MELO/non-MELO biased starting points
        if hasattr(game, 'strategy_names') and game.strategy_names:
            melo_indices = []
            non_melo_indices = []
            
            for idx, strategy_name in enumerate(game.strategy_names):
                if "MELO" in strategy_name or "melo" in strategy_name:
                    melo_indices.append(idx)
                else:
                    non_melo_indices.append(idx)
            
            if melo_indices and game.is_role_symmetric:
                # Create role symmetric MELO mixture
                pure_melo = create_role_symmetric_mixture(game, device)
                global_idx = 0
                for role_strategies in game.strategy_names_per_role:
                    role_melo_indices = []
                    role_non_melo_indices = []
                    
                    for i, strategy in enumerate(role_strategies):
                        global_strategy_idx = global_idx + i
                        if global_strategy_idx in melo_indices:
                            role_melo_indices.append(global_idx + i)
                        else:
                            role_non_melo_indices.append(global_idx + i)
                    
                    if role_melo_indices:
                        # Set this role to favor MELO strategies, but still sum to 1
                        pure_melo[role_melo_indices] = 0.8 / len(role_melo_indices)
                        if role_non_melo_indices:
                            pure_melo[role_non_melo_indices] = 0.2 / len(role_non_melo_indices)
                        # Renormalize this role to sum to 1
                        role_start = global_idx
                        role_end = global_idx + len(role_strategies)
                        role_sum = pure_melo[role_start:role_end].sum()
                        pure_melo[role_start:role_end] /= role_sum
                    
                    global_idx += len(role_strategies)
                starting_mixtures.append(pure_melo)
            elif melo_indices:
                # Regular symmetric game
                pure_melo = torch.zeros(num_strategies, device=device)
                pure_melo[melo_indices] = 1.0 / len(melo_indices)
                starting_mixtures.append(pure_melo)
        
        # Generate role symmetric random starting points
        for _ in range(num_random_starts):
            if game.is_role_symmetric:
                random_mixture = torch.zeros(num_strategies, device=device)
                global_idx = 0
                
                for role_strategies in game.strategy_names_per_role:
                    num_role_strats = len(role_strategies)
                    if num_role_strats > 0:
                        # Generate random distribution for this role and normalize to sum to 1
                        role_random = torch.rand(num_role_strats, device=device)
                        role_random = role_random / role_random.sum()
                        random_mixture[global_idx:global_idx + num_role_strats] = role_random
                    global_idx += num_role_strats
            else:
                # Regular symmetric game
                random_mixture = torch.rand(num_strategies, device=device)
                random_mixture = random_mixture / random_mixture.sum()
            
            starting_mixtures.append(random_mixture)
        
        equilibria = []  
        basin_counts = []  
        
        for start_mixture in starting_mixtures:
            if return_trace:
                result_mixture, result_trace = replicator_dynamics(
                    game=game,
                    mixture=start_mixture,
                    iters=iters,
                    offset=offset,
                    converge_threshold=converge_threshold,
                    return_trace=True,
                    use_multiple_starts=False,
                    epsilon=epsilon
                )
            else:
                result_mixture = replicator_dynamics(
                    game=game,
                    mixture=start_mixture,
                    iters=iters,
                    offset=offset,
                    converge_threshold=converge_threshold,
                    return_trace=False,
                    use_multiple_starts=False,
                    epsilon=epsilon
                )
                result_trace = None
            
            # Calculate regret
            current_regret = regret(game, result_mixture)
            if torch.is_tensor(current_regret):
                current_regret = current_regret.item()
            
            try:
                dev_payoffs = game.deviation_payoffs(result_mixture)
                expected_utility = torch.sum(result_mixture * dev_payoffs).item()
            except:
                expected_utility = 0.0
            
            found_match = False
            for i, (existing_mixture, existing_regret, _) in enumerate(equilibria):
                distance = torch.norm(existing_mixture - result_mixture, p=1).item()
                if distance < similarity_threshold:
                    basin_counts[i] += 1
                    found_match = True
                    break
            
            if not found_match:
                equilibria.append((result_mixture, current_regret, expected_utility))
                basin_counts.append(1)
        
        if equilibria:
            max_basin_size = max(basin_counts)
            candidates = [eq for i, eq in enumerate(equilibria) if basin_counts[i] == max_basin_size]
            
            candidates.sort(key=lambda x: x[1])
            
            # Return the best candidate
            best_mixture, best_regret, _ = candidates[0]
            best_trace = None  
            
            if return_trace:
                best_mixture, best_trace = replicator_dynamics(
                    game=game,
                    mixture=best_mixture,
                    iters=iters,
                    offset=offset,
                    converge_threshold=converge_threshold,
                    return_trace=True,
                    use_multiple_starts=False,
                    epsilon=epsilon
                )
                return best_mixture, best_trace
            if best_regret <= 1e-6:
                return best_mixture
        
        # Fallback to role symmetric uniform mixture
        return create_role_symmetric_mixture(game, device)
    
    # Default to uniform mixture if none provided
    if mixture is None:
        mixture = create_role_symmetric_mixture(game, device)

    if not torch.is_tensor(mixture):
        mixture = torch.tensor(mixture, dtype=torch.float32, device=device)
    
    is_vector = len(mixture.shape) == 1
    if is_vector:
        mixture = mixture.reshape(-1, 1)
    
    trace = [mixture.clone()] if return_trace else None
    
    prev_mixture = None
    for i in range(iters):
        if game.is_role_symmetric:
            # Use multi-population replicator dynamics
            new_mixture = multi_population_replicator_step(mixture.squeeze(1) if is_vector else mixture, game, offset)
            if is_vector:
                new_mixture = new_mixture.reshape(-1, 1)
        else:
            # Use standard replicator dynamics for symmetric games
            payoffs = game.deviation_payoffs(mixture)
            
            # Handle case where payoffs is 1D but mixture is 2D
            if len(mixture.shape) == 2 and len(payoffs.shape) == 1:
                payoffs = payoffs.unsqueeze(1)
            
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
        new_mixture = role_aware_normalize(counts, game)
        
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
    num_random_starts: int = 10,
    regret_threshold: float = 1e-3,
    dist_threshold: float = 1e-4,
    restricted_game_size: int = 4,
    solver: str = 'replicator',
    solver_iters: int = 5000,       
    verbose: bool = True, 
    maximal_subgames: Set[frozenset] = None,
    full_game: Optional[Game] = None
) -> List[Tuple[torch.Tensor, float]]:
    """
    find all equilibria of a game using the QUIESCE algorithm.
    this implementation follows the formal algorithm as outlined in
    Erik Brinkman's work. 
    
    Args:
        game: Game to analyze (may be reduced game when using DPR)
        num_iters: Maximum number of QUIESCE iterations
        num_random_starts: Number of random starting points to try
        regret_threshold: Regret threshold for considering a mixture an equilibrium
        dist_threshold: Distance threshold for considering mixtures distinct
        restricted_game_size: Maximum size of restricted games to explore
        solver: Solver to use ('replicator', 'fictitious_play', or 'gain_descent')
        solver_iters: Number of iterations for the solver
        verbose: Whether to print progress
        maximal_subgames: Set of frozensets of strategy indices representing explored subgames
        full_game: Optional full game for testing equilibria (used with DPR)
        
    Returns:
        List of equilibria as (mixture, regret) tuples
    """
    # Determine which game to use for testing equilibria
    test_game = full_game if full_game is not None else None
    if test_game is None:
        raise ValueError("A True Game was Not Generated/Passed in!")
    
    if verbose and full_game is not None:
        print(f"Using DPR: Finding equilibria in reduced game ({game.num_strategies} strategies, {game.num_players} players)")
        print(f"Testing equilibria against full game ({full_game.num_strategies} strategies, {full_game.num_players} players)")
    
    start_time = time.time()
    
    confirmed_eq = []
    unconfirmed_candidates = []
    deviation_queue = DeviationPriorityQueue()
    
    if maximal_subgames is None:
        maximal_subgames = set() 
    
    for s in range(game.num_strategies):
        # Create pure strategy mixture - but role symmetric
        if game.is_role_symmetric:
            # Find which role this strategy belongs to and create a pure mixture within that role
            pure_mixture = create_role_symmetric_mixture(game, game.game.device)
            
            # Find the role of strategy s
            global_idx = 0
            strategy_role_idx = None
            for role_idx, role_strategies in enumerate(game.strategy_names_per_role):
                if global_idx <= s < global_idx + len(role_strategies):
                    strategy_role_idx = role_idx
                    break
                global_idx += len(role_strategies)
            
            if strategy_role_idx is not None:
                # Set this role to be pure on strategy s, other roles get uniform
                role_start = sum(len(game.strategy_names_per_role[i]) for i in range(strategy_role_idx))
                role_size = len(game.strategy_names_per_role[strategy_role_idx])
                
                # Zero out this role and set pure strategy
                pure_mixture[role_start:role_start + role_size] = 0.0
                pure_mixture[s] = 1.0
        else:
            # Standard pure strategy mixture for symmetric games
            pure_mixture = torch.zeros(game.num_strategies, device=game.game.device)
            pure_mixture[s] = 1.0
        
        # Create candidate
        pure_candidate = SubgameCandidate(
            support=set([s]),
            restriction=[s],
            mixture=pure_mixture
        )
        
        unconfirmed_candidates.append(pure_candidate)
        
        maximal_subgames.add(frozenset([s]))
    
    # Create proper role symmetric uniform mixture
    if game.is_role_symmetric:
        uniform_mixture = create_role_symmetric_mixture(game, game.game.device)
    else:
        uniform_mixture = torch.ones(game.num_strategies, device=game.game.device) / game.num_strategies
    
    uniform_candidate = SubgameCandidate(
        support=set(range(game.num_strategies)),
        restriction=list(range(game.num_strategies)),
        mixture=uniform_mixture
    )
    unconfirmed_candidates.append(uniform_candidate)
    
    # Add joint pure strategy combinations for role symmetric games
    if game.is_role_symmetric and len(game.role_names) == 2:
        # Add important joint pure strategy starting points
        joint_pure_strategies = []
        
        # Identify CDA and MELO strategies for each role
        role_cda_strategies = [[], []]  # [MOBI_cda_strategies, ZI_cda_strategies]
        role_melo_strategies = [[], []]  # [MOBI_melo_strategies, ZI_melo_strategies]
        
        for role_idx, role_strategies in enumerate(game.strategy_names_per_role):
            for strat_idx, strategy_name in enumerate(role_strategies):
                global_strat_idx = sum(len(game.strategy_names_per_role[i]) for i in range(role_idx)) + strat_idx
                
                if "100_0" in strategy_name:  # CDA strategy
                    role_cda_strategies[role_idx].append(global_strat_idx)
                elif "0_100" in strategy_name:  # MELO strategy  
                    role_melo_strategies[role_idx].append(global_strat_idx)
        
        # Add both-CDA joint pure strategy
        if role_cda_strategies[0] and role_cda_strategies[1]:
            both_cda_mixture = create_role_symmetric_mixture(game, game.game.device)
            # Set MOBI to pure CDA, ZI to pure CDA
            both_cda_mixture[:] = 0.0
            both_cda_mixture[role_cda_strategies[0][0]] = 1.0  # MOBI_100_0
            both_cda_mixture[role_cda_strategies[1][0]] = 1.0  # ZI_100_0
            
            both_cda_candidate = SubgameCandidate(
                support=set([role_cda_strategies[0][0], role_cda_strategies[1][0]]),
                restriction=[role_cda_strategies[0][0], role_cda_strategies[1][0]],
                mixture=both_cda_mixture
            )
            unconfirmed_candidates.append(both_cda_candidate)
            
            if verbose:
                print(f"Added joint pure CDA starting point: both roles play CDA strategies")
        
        # Add both-MELO joint pure strategy  
        if role_melo_strategies[0] and role_melo_strategies[1]:
            both_melo_mixture = create_role_symmetric_mixture(game, game.game.device)
            # Set MOBI to pure MELO, ZI to pure MELO
            both_melo_mixture[:] = 0.0
            both_melo_mixture[role_melo_strategies[0][0]] = 1.0  # MOBI_0_100
            both_melo_mixture[role_melo_strategies[1][0]] = 1.0  # ZI_0_100
            
            both_melo_candidate = SubgameCandidate(
                support=set([role_melo_strategies[0][0], role_melo_strategies[1][0]]),
                restriction=[role_melo_strategies[0][0], role_melo_strategies[1][0]],
                mixture=both_melo_mixture
            )
            unconfirmed_candidates.append(both_melo_candidate)
            
            if verbose:
                print(f"Added joint pure MELO starting point: both roles play MELO strategies")
        
        # Add cross-combinations: MOBI-CDA + ZI-MELO
        if role_cda_strategies[0] and role_melo_strategies[1]:
            cross1_mixture = create_role_symmetric_mixture(game, game.game.device)
            cross1_mixture[:] = 0.0
            cross1_mixture[role_cda_strategies[0][0]] = 1.0   # MOBI_100_0 (CDA)
            cross1_mixture[role_melo_strategies[1][0]] = 1.0  # ZI_0_100 (MELO)
            
            cross1_candidate = SubgameCandidate(
                support=set([role_cda_strategies[0][0], role_melo_strategies[1][0]]),
                restriction=[role_cda_strategies[0][0], role_melo_strategies[1][0]],
                mixture=cross1_mixture
            )
            unconfirmed_candidates.append(cross1_candidate)
            
            if verbose:
                print(f"Added cross combination: MOBI-CDA + ZI-MELO")
        
        # Add cross-combinations: MOBI-MELO + ZI-CDA
        if role_melo_strategies[0] and role_cda_strategies[1]:
            cross2_mixture = create_role_symmetric_mixture(game, game.game.device)
            cross2_mixture[:] = 0.0
            cross2_mixture[role_melo_strategies[0][0]] = 1.0  # MOBI_0_100 (MELO)
            cross2_mixture[role_cda_strategies[1][0]] = 1.0   # ZI_100_0 (CDA)
            
            cross2_candidate = SubgameCandidate(
                support=set([role_melo_strategies[0][0], role_cda_strategies[1][0]]),
                restriction=[role_melo_strategies[0][0], role_cda_strategies[1][0]],
                mixture=cross2_mixture
            )
            unconfirmed_candidates.append(cross2_candidate)
            
            if verbose:
                print(f"Added cross combination: MOBI-MELO + ZI-CDA")
    
    # Add pure MELO and pure non-MELO starting points
    if hasattr(game, 'strategy_names') and game.strategy_names:
        # Identify MELO strategies (those containing "MELO" in their name)
        melo_indices = []
        non_melo_indices = []
        
        for idx, strategy_name in enumerate(game.strategy_names):
            if "MELO" in strategy_name or "melo" in strategy_name:
                melo_indices.append(idx)
            else:
                non_melo_indices.append(idx)
        
        if melo_indices:
            if game.is_role_symmetric:
                pure_melo_mixture = create_role_symmetric_mixture(game, game.game.device)
                # Bias each role toward MELO strategies within that role
                global_idx = 0
                for role_strategies in game.strategy_names_per_role:
                    role_melo_indices = []
                    role_non_melo_indices = []
                    
                    for i, strategy in enumerate(role_strategies):
                        global_strategy_idx = global_idx + i
                        if global_strategy_idx in melo_indices:
                            role_melo_indices.append(global_idx + i)
                        else:
                            role_non_melo_indices.append(global_idx + i)
                    
                    if role_melo_indices:
                        # Set this role to favor MELO strategies, but still sum to 1
                        pure_melo_mixture[role_melo_indices] = 0.8 / len(role_melo_indices)
                        if role_non_melo_indices:
                            pure_melo_mixture[role_non_melo_indices] = 0.2 / len(role_non_melo_indices)
                        # Renormalize this role to sum to 1
                        role_start = global_idx
                        role_end = global_idx + len(role_strategies)
                        role_sum = pure_melo_mixture[role_start:role_end].sum()
                        pure_melo_mixture[role_start:role_end] /= role_sum
                    
                    global_idx += len(role_strategies)
            else:
                pure_melo_mixture = torch.zeros(game.num_strategies, device=game.game.device)
                pure_melo_mixture[melo_indices] = 1.0 / len(melo_indices)
            
            pure_melo_candidate = SubgameCandidate(
                support=set(melo_indices),
                restriction=melo_indices,
                mixture=pure_melo_mixture
            )
            unconfirmed_candidates.append(pure_melo_candidate)
            
            if verbose:
                print(f"Added pure MELO starting point with {len(melo_indices)} MELO strategies")
            
            if len(melo_indices) <= restricted_game_size:
                maximal_subgames.add(frozenset(melo_indices))
        
        if non_melo_indices:
            if game.is_role_symmetric:
                pure_non_melo_mixture = create_role_symmetric_mixture(game, game.game.device)
                # Bias each role toward non-MELO strategies within that role
                global_idx = 0
                for role_strategies in game.strategy_names_per_role:
                    role_melo_indices = []
                    role_non_melo_indices = []
                    
                    for i, strategy in enumerate(role_strategies):
                        global_strategy_idx = global_idx + i
                        if global_strategy_idx in melo_indices:
                            role_melo_indices.append(global_idx + i)
                        else:
                            role_non_melo_indices.append(global_idx + i)
                    
                    if role_non_melo_indices:
                        # Set this role to favor non-MELO strategies, but still sum to 1
                        pure_non_melo_mixture[role_non_melo_indices] = 0.8 / len(role_non_melo_indices)
                        if role_melo_indices:
                            pure_non_melo_mixture[role_melo_indices] = 0.2 / len(role_melo_indices)
                        # Renormalize this role to sum to 1
                        role_start = global_idx
                        role_end = global_idx + len(role_strategies)
                        role_sum = pure_non_melo_mixture[role_start:role_end].sum()
                        pure_non_melo_mixture[role_start:role_end] /= role_sum
                    
                    global_idx += len(role_strategies)
            else:
                pure_non_melo_mixture = torch.zeros(game.num_strategies, device=game.game.device)
                pure_non_melo_mixture[non_melo_indices] = 1.0 / len(non_melo_indices)
            
            pure_non_melo_candidate = SubgameCandidate(
                support=set(non_melo_indices),
                restriction=non_melo_indices,
                mixture=pure_non_melo_mixture
            )
            unconfirmed_candidates.append(pure_non_melo_candidate)
            
            if verbose:
                print(f"Added pure non-MELO starting point with {len(non_melo_indices)} non-MELO strategies")
            
            if len(non_melo_indices) <= restricted_game_size:
                maximal_subgames.add(frozenset(non_melo_indices))
    
    if len(game.strategy_names) <= restricted_game_size:
        maximal_subgames.add(frozenset(range(game.num_strategies)))
    
    if verbose:
        print(f"Initialized with {len(maximal_subgames)} maximal subgames")
    
    # Add skewed distributions (80% on one strategy, 20% distributed on others)
    for primary_strategy in range(game.num_strategies):
        if game.num_strategies > 1:
            if game.is_role_symmetric:
                # Create role symmetric skewed mixture
                skewed_mixture = create_role_symmetric_mixture(game, game.game.device)
                
                # Find which role the primary strategy belongs to
                global_idx = 0
                strategy_role_idx = None
                for role_idx, role_strategies in enumerate(game.strategy_names_per_role):
                    if global_idx <= primary_strategy < global_idx + len(role_strategies):
                        strategy_role_idx = role_idx
                        break
                    global_idx += len(role_strategies)
                
                if strategy_role_idx is not None:
                    # Modify the role containing the primary strategy
                    role_start = sum(len(game.strategy_names_per_role[i]) for i in range(strategy_role_idx))
                    role_size = len(game.strategy_names_per_role[strategy_role_idx])
                    
                    if role_size > 1:
                        # Set all strategies in this role to small value, then make primary strategy dominant
                        skewed_mixture[role_start:role_start + role_size] = 0.2 / role_size
                        skewed_mixture[primary_strategy] = 0.8
                        # Renormalize this role to sum to 1
                        role_sum = skewed_mixture[role_start:role_start + role_size].sum()
                        skewed_mixture[role_start:role_start + role_size] /= role_sum
            else:
                skewed_mixture = torch.zeros(game.num_strategies, device=game.game.device)
                skewed_mixture[primary_strategy] = 0.8
                
                # Distribute remaining 20% equally among other strategies
                remaining_mass = 0.2
                other_strategies = [s for s in range(game.num_strategies) if s != primary_strategy]
                mass_per_other = remaining_mass / len(other_strategies)
                
                for other_s in other_strategies:
                    skewed_mixture[other_s] = mass_per_other
            
            # Create candidate
            support = set([s for s in range(game.num_strategies) if skewed_mixture[s] > 0.01])
            skewed_candidate = SubgameCandidate(
                support=support,
                restriction=list(support),
                mixture=skewed_mixture
            )
            unconfirmed_candidates.append(skewed_candidate)
            
            if verbose:
                strategy_name = game.strategy_names[primary_strategy] if hasattr(game, 'strategy_names') else f"Strategy {primary_strategy}"
                print(f"Added 80/20 skewed candidate favoring {strategy_name}")
    
    # Add other structured distributions (70/30, 60/40)
    for primary_strategy in range(game.num_strategies):
        if game.num_strategies > 1:
            for primary_weight in [0.7, 0.6]:
                if game.is_role_symmetric:
                    # Create role symmetric skewed mixture
                    skewed_mixture = create_role_symmetric_mixture(game, game.game.device)
                    
                    # Find which role the primary strategy belongs to
                    global_idx = 0
                    strategy_role_idx = None
                    for role_idx, role_strategies in enumerate(game.strategy_names_per_role):
                        if global_idx <= primary_strategy < global_idx + len(role_strategies):
                            strategy_role_idx = role_idx
                            break
                        global_idx += len(role_strategies)
                    
                    if strategy_role_idx is not None:
                        # Modify the role containing the primary strategy
                        role_start = sum(len(game.strategy_names_per_role[i]) for i in range(strategy_role_idx))
                        role_size = len(game.strategy_names_per_role[strategy_role_idx])
                        
                        if role_size > 1:
                            # Set all strategies in this role to small value, then make primary strategy dominant
                            remaining_weight = 1.0 - primary_weight
                            skewed_mixture[role_start:role_start + role_size] = remaining_weight / role_size
                            skewed_mixture[primary_strategy] = primary_weight
                            # Renormalize this role to sum to 1
                            role_sum = skewed_mixture[role_start:role_start + role_size].sum()
                            skewed_mixture[role_start:role_start + role_size] /= role_sum
                else:
                    skewed_mixture = torch.zeros(game.num_strategies, device=game.game.device)
                    skewed_mixture[primary_strategy] = primary_weight
                    
                    # Distribute remaining mass equally among other strategies
                    remaining_mass = 1.0 - primary_weight
                    other_strategies = [s for s in range(game.num_strategies) if s != primary_strategy]
                    mass_per_other = remaining_mass / len(other_strategies)
                    
                    for other_s in other_strategies:
                        skewed_mixture[other_s] = mass_per_other
                
                # Create candidate
                support = set([s for s in range(game.num_strategies) if skewed_mixture[s] > 0.01])
                skewed_candidate = SubgameCandidate(
                    support=support,
                    restriction=list(support),
                    mixture=skewed_mixture
                )
                unconfirmed_candidates.append(skewed_candidate)
                
                if verbose:
                    strategy_name = game.strategy_names[primary_strategy] if hasattr(game, 'strategy_names') else f"Strategy {primary_strategy}"
                    weight_pct = int(primary_weight * 100)
                    other_pct = int((1.0 - primary_weight) * 100)
                    print(f"Added {weight_pct}/{other_pct} skewed candidate favoring {strategy_name}")
    
    # Add additional diverse random starting points for maximum exploration
    additional_random_starts = max(50, num_random_starts * 2)  # More diverse random starts
    for i in range(additional_random_starts):
        if game.is_role_symmetric:
            # Generate role symmetric random mixture
            rand_mixture = torch.zeros(game.num_strategies, device=game.game.device)
            global_idx = 0
            
            for role_strategies in game.strategy_names_per_role:
                num_role_strats = len(role_strategies)
                if num_role_strats > 0:
                    # Generate different types of random distributions for diversity
                    if i % 4 == 0:
                        # Pure random (uniform distribution)
                        role_random = torch.rand(num_role_strats, device=game.game.device)
                    elif i % 4 == 1:
                        # Dirichlet-like (more extreme distributions)
                        role_random = torch.rand(num_role_strats, device=game.game.device) ** 2
                    elif i % 4 == 2:
                        # Concentrated (favor one strategy heavily)
                        role_random = torch.rand(num_role_strats, device=game.game.device) ** 0.5
                    else:
                        # Uniform-ish (less extreme)
                        role_random = torch.rand(num_role_strats, device=game.game.device) + 0.5
                    
                    # Normalize this role to sum to 1
                    role_random = role_random / role_random.sum()
                    rand_mixture[global_idx:global_idx + num_role_strats] = role_random
                global_idx += num_role_strats
        else:
            # Generate different types of random mixtures for diversity (symmetric games)
            if i % 4 == 0:
                # Pure random (uniform distribution)
                rand_mixture = torch.rand(game.num_strategies, device=game.game.device)
            elif i % 4 == 1:
                # Dirichlet-like (more extreme distributions)
                rand_mixture = torch.rand(game.num_strategies, device=game.game.device) ** 2
            elif i % 4 == 2:
                # Concentrated (favor one strategy heavily)
                rand_mixture = torch.rand(game.num_strategies, device=game.game.device) ** 0.5
            else:
                # Uniform-ish (less extreme)
                rand_mixture = torch.rand(game.num_strategies, device=game.game.device) + 0.5
            
            # Normalize to sum to 1
            rand_mixture = rand_mixture / rand_mixture.sum()
        
        # Apply MELO/non-MELO biasing for some candidates (only if game has strategy names)
        if hasattr(game, 'strategy_names') and game.strategy_names and i < additional_random_starts // 4:
            melo_mask = torch.zeros(game.num_strategies, dtype=torch.bool, device=game.game.device)
            for idx, strategy_name in enumerate(game.strategy_names):
                if "MELO" in strategy_name or "melo" in strategy_name:
                    melo_mask[idx] = True
            
            if game.is_role_symmetric:
                # Apply biasing within each role
                global_idx = 0
                for role_strategies in game.strategy_names_per_role:
                    role_melo_mask = melo_mask[global_idx:global_idx + len(role_strategies)]
                    role_mixture = rand_mixture[global_idx:global_idx + len(role_strategies)]
                    
                    if i % 3 == 0 and role_melo_mask.any():
                        # Bias toward MELO strategies in this role
                        role_mixture[role_melo_mask] *= 5.0
                    elif i % 3 == 1 and (~role_melo_mask).any():
                        # Bias toward non-MELO strategies in this role
                        role_mixture[~role_melo_mask] *= 5.0
                    # i % 3 == 2: Keep original mixture (no bias)
                    
                    # Renormalize this role
                    role_mixture = role_mixture / role_mixture.sum()
                    rand_mixture[global_idx:global_idx + len(role_strategies)] = role_mixture
                    global_idx += len(role_strategies)
            else:
                # Standard biasing for symmetric games
                if i % 3 == 0 and melo_mask.any():
                    # Bias toward MELO strategies
                    rand_mixture[melo_mask] *= 5.0
                elif i % 3 == 1 and (~melo_mask).any():
                    # Bias toward non-MELO strategies
                    rand_mixture[~melo_mask] *= 5.0
                # Renormalize
                rand_mixture = rand_mixture / rand_mixture.sum()
        
        # Create support (strategies with significant probability)
        support = set([s for s in range(game.num_strategies) if rand_mixture[s] > 0.01])
        if len(support) == 0:  
            support = {torch.argmax(rand_mixture).item()}
            
        rand_candidate = SubgameCandidate(
            support=support,
            restriction=list(support),
            mixture=rand_mixture
        )
        unconfirmed_candidates.append(rand_candidate)
        
        # Update maximal subgames
        if len(support) <= restricted_game_size:
            support_set = frozenset(support)
            is_contained = False
            for maximal_set in list(maximal_subgames):
                if support_set.issubset(maximal_set):
                    is_contained = True
                    break
                if maximal_set.issubset(support_set):
                    maximal_subgames.remove(maximal_set)
            
            if not is_contained:
                maximal_subgames.add(support_set)
    
    if verbose:
        total_candidates = len(unconfirmed_candidates)
        print(f"Generated {total_candidates} total candidates (including {num_random_starts + additional_random_starts} random starts)")
        print(f"Updated to {len(maximal_subgames)} maximal subgames")
    
    for iteration in range(num_iters):
        if verbose:
            print(f"QUIESCE iteration {iteration+1}/{num_iters} with {len(unconfirmed_candidates)} candidates")
        
        if not unconfirmed_candidates and deviation_queue.is_empty():
            break
            
        new_unconfirmed = []
        for candidate in unconfirmed_candidates:
            is_eq, _ = await test_candidate(candidate, game, regret_threshold, deviation_queue, restricted_game_size, maximal_subgames, verbose, test_game)
            
            if is_eq:
                is_distinct = True
                for eq_mixture, _ in confirmed_eq:
                    dist = torch.norm(candidate.mixture - eq_mixture, p=1).item()
                    if dist < dist_threshold:
                        is_distinct = False
                        break
                        
                if is_distinct:
                    confirmed_eq.append((candidate.mixture.clone(), candidate.regret))
            else:
                new_unconfirmed.append(candidate)
                
        unconfirmed_candidates = new_unconfirmed
        
        if not deviation_queue.is_empty():
            gain, strategy, base_mixture = deviation_queue.pop()

            if verbose:
                print(f"  Exploring deviation to {game.strategy_names[strategy]} with gain {gain:.6f}")
                
            support = set([i for i, p in enumerate(base_mixture) if p > 0.01])
            
            new_support = support.union({strategy})
            
            restriction = list(new_support)
            restriction_set = frozenset(restriction)
            
            is_contained = False
            for maximal_set in maximal_subgames:
                if restriction_set.issubset(maximal_set):
                    is_contained = True
                    break
            
            if not is_contained:
                for maximal_set in list(maximal_subgames):
                    if maximal_set.issubset(restriction_set):
                        maximal_subgames.remove(maximal_set)
                
                if len(restriction) <= restricted_game_size:
                    maximal_subgames.add(restriction_set)
                    if verbose:
                        print(f"Added new maximal subgame: {restriction}")
            
            # Create restricted game
            restricted_game = game.restrict(restriction)
            
            # Create initial mixture for the restricted game
            if restricted_game.is_role_symmetric:
                init_restricted = create_role_symmetric_mixture(restricted_game, game.game.device)
                # Bias toward the support strategies and add the deviating strategy
                restricted_strategy_to_full = {i: restriction[i] for i in range(len(restriction))}
                
                for i, full_strategy_idx in enumerate(restriction):
                    if full_strategy_idx in support:
                        # Find which role this strategy belongs to in the restricted game
                        global_idx = 0
                        for role_strategies in restricted_game.strategy_names_per_role:
                            if global_idx <= i < global_idx + len(role_strategies):
                                # This strategy is in this role, bias the initialization
                                init_restricted[i] = base_mixture[full_strategy_idx] * 0.8
                                break
                            global_idx += len(role_strategies)
                
                new_idx = restriction.index(strategy)
                init_restricted[new_idx] = 0.2
                
                # Renormalize each role in the restricted game
                global_idx = 0
                for role_strategies in restricted_game.strategy_names_per_role:
                    role_slice = slice(global_idx, global_idx + len(role_strategies))
                    role_sum = init_restricted[role_slice].sum()
                    if role_sum > 1e-10:
                        init_restricted[role_slice] = init_restricted[role_slice] / role_sum
                    else:
                        init_restricted[role_slice] = 1.0 / len(role_strategies)
                    global_idx += len(role_strategies)
            else:
                # Standard symmetric game initialization
                init_restricted = torch.zeros(len(restriction), device=game.game.device)
                for i, s in enumerate(restriction):
                    if s in support:
                        idx = list(support).index(s)
                        init_restricted[i] = base_mixture[s] * 0.8
                
                new_idx = restriction.index(strategy)
                init_restricted[new_idx] = 0.2 
                
                init_restricted = init_restricted / init_restricted.sum()
            
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
            
            # Test regret against the appropriate game (full game if using DPR)
            regret_val = test_game.regret(full_mixture).item()
            
            candidate = SubgameCandidate(
                support=new_support,
                restriction=restriction,
                mixture=full_mixture,
                regret=regret_val
            )
            
            if verbose:
                print(f"  Solver completed in {solver_time:.4f} seconds with regret {regret_val:.6f}")
                
            unconfirmed_candidates.append(candidate)
    
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
    verbose=False,
    full_game=None  # Full game for testing equilibria (when using DPR)
):
    """Synchronous wrapper for quiesce - finds all equilibria of a game using QUIESCE."""
    # Import and apply nest_asyncio to handle nested event loops
    import nest_asyncio
    nest_asyncio.apply()
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        maximal_subgames = set()
        
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
            verbose=verbose,
            full_game=full_game  # Pass through fu_game parameter
        ))
        loop.close()
        
        return equilibria
        
    except Exception as e:
        if verbose:
            print(f"QUIESCE failed: {e}")
        
        try:
            if verbose:
                print("Attempting replicator dynamics on full game as last resort.")
            
            test_game = full_game if full_game is not None else game
            if test_game.is_role_symmetric:
                init_mixture = create_role_symmetric_mixture(test_game, test_game.game.device)
            else:
                init_mixture = torch.ones(test_game.num_strategies, device=test_game.game.device) / test_game.num_strategies
            
            try:
                payoff_matrix = test_game.get_payoff_matrix()
                payoff_matrix = torch.nan_to_num(payoff_matrix, nan=0.0, posinf=10.0, neginf=-10.0)
                payoff_matrix = torch.clamp(payoff_matrix, min=-10.0, max=10.0)
                
                eq, _ = replicator_dynamics(
                    test_game,
                    init_mixture,
                    iters=solver_iters,
                    return_trace=True
                )
            except:
                eq, _ = replicator_dynamics(
                    test_game,
                    init_mixture,
                    iters=solver_iters,
                    return_trace=True
                )
            
            # Calculate regret
            regret_val = test_game.regret(eq).item() if torch.is_tensor(test_game.regret(eq)) else test_game.regret(eq)
            
            if verbose:
                print(f"Replicator dynamics found: {format_mixture(eq, test_game.strategy_names)}")
                print(f"Equilibrium regret: {regret_val:.6f}")
            
            return [(eq, regret_val)]
            
        except Exception as e2:
            if verbose:
                print(f"All equilibrium finding methods failed. Last error: {e2}")
            # Return uniform mixture as a last resort
            test_game = full_game if full_game is not None else None
            if test_game and test_game.is_role_symmetric:
                uniform = create_role_symmetric_mixture(test_game, test_game.game.device)
            else:
                uniform = torch.ones(test_game.num_strategies, device=test_game.game.device) / test_game.num_strategies
            return [(uniform, 1.0)]

async def test_deviations(
    game, 
    mixture, 
    deviation_queue, 
    regret_threshold=1e-6, 
    restricted_game_size=4, 
    maximal_subgames=None,
    verbose=False
):
    num_strategies = game.num_strategies
    
    #if metadata exists and dpr_reduced is True, then we're testing against reduced game
    is_testing_reduced_game = (hasattr(game, 'metadata') and game.metadata and 
                              game.metadata.get('dpr_reduced', False) == True)
    
    if verbose and hasattr(game, 'metadata') and game.metadata:
        if game.metadata.get('dpr_reduced', False):
            print(f"    Testing deviations against REDUCED game ({game.num_players} players)")
        else:
            original_players = game.metadata.get('original_players')
            if original_players and original_players != game.num_players:
                print(f"Testing deviations against FULL game ({game.num_players} players, was reduced from {original_players})")
            else:
                print(f" Testing deviations against FULL game ({game.num_players} players)")
    elif verbose:
        print(f"Testing deviations against game ({game.num_players} players)")
    
    dev_payoffs = game.deviation_payoffs(mixture)
    
    if torch.is_tensor(dev_payoffs) and (torch.isnan(dev_payoffs).any() or torch.isinf(dev_payoffs).any()):
        if verbose:
            print(f"Warning: NaN or Inf in deviation payoffs. Fixing values.")
        dev_payoffs = torch.nan_to_num(dev_payoffs, nan=0.0, posinf=10.0, neginf=-10.0)
        #dev_payoffs = torch.clamp(dev_payoffs, min=-10.0, max=10.0)
        
    # Calculate expected payoff for the mixture - safely
    expected_payoff_values = mixture * dev_payoffs
    if torch.isnan(expected_payoff_values).any() or torch.isinf(expected_payoff_values).any():
        expected_payoff_values = torch.nan_to_num(expected_payoff_values, nan=0.0, posinf=0.0, neginf=0.0)
    
    expected_payoff = expected_payoff_values.sum()
    
    gains = dev_payoffs - expected_payoff
    #gains = torch.clamp(gains, min=-10.0, max=10.0)
    
    has_beneficial = False
    beneficial_strategies = []
    
    for s in range(num_strategies):
        support = set([i for i, p in enumerate(mixture) if p > 0.01])
        
        if s in support:
            continue
        
        gain = gains[s].item() if torch.is_tensor(gains[s]) else gains[s]
        
        if np.isnan(gain) or np.isinf(gain):
            if verbose:
                print(f"    Warning: NaN/Inf gain for strategy {s}. Skipping.")
            continue
        
        if gain > regret_threshold:
            has_beneficial = True
            beneficial_strategies.append((s, gain))
            
            new_support = support.union({s})
            
            if len(new_support) <= restricted_game_size:
                should_explore = True
                if maximal_subgames is not None:
                    new_support_set = frozenset(new_support)
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
                        game_type = "REDUCED" if is_testing_reduced_game else "FULL"
                        print(f"Found beneficial deviation to {game.strategy_names[s]} "
                              f"with gain {gain:.6f} (vs {game_type} game)")
            else:
                if verbose:
                    print(f"    Skipping deviation to {game.strategy_names[s]} as it would exceed restricted game size")
    
    # If no beneficial deviations, this is an equilibrium
    if not has_beneficial:
        if verbose:
            game_type = "REDUCED" if is_testing_reduced_game else "FULL"
            print(f"    Found equilibrium with regret {game.regret(mixture):.6f} (vs {game_type} game): "
                  f"{format_mixture(mixture, game.strategy_names)}")
        return True, []
        
    return False, beneficial_strategies


async def test_candidate(candidate, game, regret_threshold, deviation_queue, restricted_game_size, maximal_subgames=None, verbose=False, full_game: Optional[Game] = None):
    """Test a candidate equilibrium and add beneficial deviations to the queue."""
    mixture = candidate.mixture
    num_strategies = game.num_strategies
    
    # Determine which game to use for testing
    test_game = full_game if full_game is not None else game
    
    # Safety check to ensure we have a true game for testing
    if test_game is None:
        raise ValueError("No valid game available for testing equilibrium candidates")
    
    # Validate mixture for numerical issues before testing
    if torch.isnan(mixture).any() or torch.isinf(mixture).any() or torch.any(mixture < 0):
        if verbose:
            print(f"    Warning: Invalid mixture detected. Fixing before testing.")
        # Fix the mixture
        mixture = torch.nan_to_num(mixture, nan=1.0/num_strategies)
        mixture = torch.clamp(mixture, min=1e-6)
        mixture = mixture / mixture.sum()
        candidate.mixture = mixture
    
    # Calculate regret using the test game (full game if using DPR)
    try:
        regret_val = test_game.regret(mixture)
        
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
            print(f"    Candidate regret (vs {'full' if full_game else 'reduced'} game): {regret_val:.6f}")
            
        # If regret is above threshold, not an equilibrium
        if regret_val > regret_threshold:
            return False, []
            
    except Exception as e:
        if verbose:
            print(f"    Error calculating regret: {e}")
        candidate.regret = 1.0
        return False, []
    
    # Now test for beneficial deviations using the test game
    return await test_deviations(
        game=test_game,  # Use test_game for deviation testing
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
        strategy_names: List https://www.notion.so/MELO-Project-1ec6ce8fce1980b5898ed44329a5c769?pvs=4of strategy names
        threshold: Threshold for including strategies in the output
        
    Returns:
        String representation of the mixture
    """
    if not torch.is_tensor(mixture):
        mixture = torch.tensor(mixture)
    
    significant_idxs = (mixture > threshold).nonzero().flatten().cpu().numpy()
    
    if len(significant_idxs) == 0:
        return "Uniform mixture"
    
    parts = []
    for idx in significant_idxs:
        prob = mixture[idx].item()
        name = strategy_names[idx] if idx < len(strategy_names) else f"Strategy {idx}"
        parts.append(f"{name}:{prob:.4f}")
    
    return ", ".join(parts) 