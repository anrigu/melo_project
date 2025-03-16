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
from dataclasses import dataclass, field
from collections import defaultdict
from marketsim.egta.core.game import Game
from marketsim.math.simplex_operations import simplex_normalize, simplex_projection


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


@dataclass
class SubgameCandidate:
    """
    A candidate subgame mixture and associated data.
    """
    mixture: torch.Tensor
    regret: float = float('inf')
    confirmed: bool = False
    support: Set[int] = field(default_factory=set)
    evaluated: bool = False
    id: int = field(default_factory=lambda: SubgameCandidate._next_id())
    
    _id_counter = 0
    
    @classmethod
    def _next_id(cls):
        cls._id_counter += 1
        return cls._id_counter


class MaximalSubgameCollection:
    """
    Collection of maximal subgames for tracking the QUIESCE state.
    """
    def __init__(self):
        self.subgames = []  # List of sets containing strategy indices
    
    def add(self, support: Set[int]) -> bool:
        """
        Add a new subgame to the collection if it's not contained in another.
        Remove any existing subgames that are contained in this one.
        
        Args:
            support: Set of strategy indices defining the subgame
            
        Returns:
            True if the subgame was added, False otherwise
        """
        for existing in self.subgames:
            if support.issubset(existing):
                return False
        
        self.subgames = [s for s in self.subgames if not s.issubset(support)]
        
        self.subgames.append(support)
        return True
    
    def is_contained(self, support: Set[int]) -> bool:
        """
        Check if a support set is contained in any existing subgame.
        Args:
            support: Set of strategy indices
            
        Returns:
            True if contained in any existing subgame, False otherwise
        """
        return any(support.issubset(existing) for existing in self.subgames)


class DeviationPriorityQueue:
    """
    Priority queue for deviations, ordered by gain.
    """
    def __init__(self):
        self.queue = []  # Heap queue of (negative_gain, id, strategy_idx, mixture)
    
    def push(self, gain: float, strategy_idx: int, mixture: torch.Tensor):
        """
        Push a deviation to the queue.
        
        Args:
            gain: Gain from deviation
            strategy_idx: Strategy index to deviate to
            mixture: Mixture to start from
        """
        # Use negative gain for max-heap behavior with heapq (which is a min-heap)
        item_id = len(self.queue)
        heapq.heappush(self.queue, (-gain, item_id, strategy_idx, mixture))
    
    def pop(self):
        """
        Pop the highest-gain deviation from the queue.
        
        Returns:
            Tuple of (gain, strategy_idx, mixture) or None if queue is empty
        """
        if not self.queue:
            return None
        
        neg_gain, _, strategy_idx, mixture = heapq.heappop(self.queue)
        return (-neg_gain, strategy_idx, mixture)
    
    def __len__(self):
        return len(self.queue)


async def quiesce(
    game: Game, 
    num_iters: int = 10,
    regret_threshold: float = 1e-4,
    dist_threshold: float = 1e-3,
    restricted_game_size: int = 4,
    solver: str = 'replicator',
    solver_iters: int = 1000,
    verbose: bool = True
) -> List[Tuple[torch.Tensor, float]]:
    """
    find all equilibria of a game using the QUIESCE algorithm.
    This implementation follows the formal algorithm as outlined in
    Erik Brinkman's work. 
    
    Args:
        game: Game to analyze
        num_iters: Maximum number of QUIESCE iterations
        regret_threshold: Regret threshold for considering a mixture an equilibrium
        dist_threshold: Distance threshold for considering mixtures distinct
        restricted_game_size: Maximum size of restricted games to explore
        solver: Solver to use ('replicator', 'fictitious_play', or 'gain_descent')
        solver_iters: Number of iterations for the solver
        verbose: Whether to print progress
        
    Returns:
        List of (mixture, regret) tuples, sorted by regret
    """
    device = game.game.device
    num_strategies = game.num_strategies
    
    if solver == 'replicator':
        solver_fn = lambda mix: replicator_dynamics(game, mix, iters=solver_iters)
    elif solver == 'fictitious_play':
        solver_fn = lambda mix: fictitious_play(game, mix, iters=solver_iters)
    elif solver == 'gain_descent':
        solver_fn = lambda mix: gain_descent(game, mix, iters=solver_iters)
    else:
        raise ValueError(f"Unknown solver: {solver}")
    
    # Initialize data structures
    maximal_subgames = MaximalSubgameCollection()
    unconfirmed_candidates = []  # List of SubgameCandidate objects
    confirmed_equilibria = []    # List of (mixture, regret) tuples
    deviation_queue = DeviationPriorityQueue()
    
    # Initialize with singleton strategy subgames
    for s in range(num_strategies):
        support = {s}
        if maximal_subgames.add(support):
            # Create pure strategy mixture
            mixture = torch.zeros(num_strategies, device=device)
            mixture[s] = 1.0
            
            candidate = SubgameCandidate(mixture=mixture, support=support)
            unconfirmed_candidates.append(candidate)

    async def test_unconfirmed_candidates():
        """
        Test unconfirmed candidates to see if they are equilibria.
        """
        nonlocal unconfirmed_candidates
        
        # Get all unconfirmed candidates
        candidates_to_test = [c for c in unconfirmed_candidates if not c.evaluated]
        if not candidates_to_test:
            return
        
        if verbose:
            print(f"  Testing {len(candidates_to_test)} unconfirmed candidates")
        
        # Convert to batch for faster processing
        mixtures = torch.stack([c.mixture for c in candidates_to_test])
        
        # Solve equilibria for all candidates
        solved_mixtures = solver_fn(mixtures.t()).t()
        
        # Calculate regrets
        regrets = regret(game, solved_mixtures.t())
        
        # Process results
        for i, candidate in enumerate(candidates_to_test):
            solved_mix = solved_mixtures[i]
            reg = regrets[i].item()
            
            # Update candidate info
            candidate.evaluated = True
            candidate.mixture = solved_mix
            candidate.regret = reg
            candidate.support = set(torch.where(solved_mix > 1e-4)[0].cpu().numpy().tolist())
            
            if verbose:
                print(f"    Candidate {i+1} regret: {reg:.6f}")
            
            # If regret is below threshold, test for beneficial deviations
            if reg < regret_threshold:
                is_beneficial = await test_deviations(candidate)
                
                if not is_beneficial:
                    # No beneficial deviations - confirmed equilibrium
                    is_duplicate = False
                    for mix, _ in confirmed_equilibria:
                        if torch.max(torch.abs(mix - solved_mix)) < dist_threshold:
                            is_duplicate = True
                            break
                    
                    if not is_duplicate:
                        candidate.confirmed = True
                        confirmed_equilibria.append((solved_mix, reg))
                        
                        if verbose:
                            strat_str = ", ".join([
                                f"{game.strategy_names[s]}: {solved_mix[s].item():.4f}" 
                                for s in range(num_strategies) 
                                if solved_mix[s].item() > 0.01
                            ])
                            print(f"    Found equilibrium with regret {reg:.6f}: {strat_str}")
        
        # Remove tested candidates
        unconfirmed_candidates = [c for c in unconfirmed_candidates if not c.evaluated]
    
    async def test_deviations(candidate: SubgameCandidate) -> bool:
        """
        Test if there are beneficial deviations from a candidate mixture.
        
        Args:
            candidate: Candidate to test
            
        Returns:
            True if there are beneficial deviations, False otherwise
        """
        mixture = candidate.mixture
        
        # Get deviation payoffs
        dev_payoffs = game.deviation_payoffs(mixture)
        
        # Calculate expected payoff for the mixture
        expected_payoff = (mixture * dev_payoffs).sum()
        
        # Calculate gains from deviation
        gains = dev_payoffs - expected_payoff
        
        # Find beneficial deviations
        has_beneficial = False
        beneficial_strategies = []
        
        for s in range(num_strategies):
            # Skip strategies already in support
            if s in candidate.support:
                continue
            
            gain = gains[s].item()
            
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
        
        # If there are beneficial deviations and this is a pure strategy,
        # also add mixed strategy candidates for exploration
        if has_beneficial and len(candidate.support) == 1:
            # Get the pure strategy index
            pure_strat_idx = next(iter(candidate.support))
            
            for s, gain in beneficial_strategies:
                # Create a mixed strategy between the current pure strategy and the beneficial deviation
                for alpha in [0.25, 0.5, 0.75]:
                    new_mixture = torch.zeros_like(mixture)
                    new_mixture[pure_strat_idx] = 1.0 - alpha
                    new_mixture[s] = alpha
                    
                    # Create a new candidate
                    new_support = {pure_strat_idx, s}
                    new_candidate = SubgameCandidate(mixture=new_mixture, support=new_support)
                    unconfirmed_candidates.append(new_candidate)
                    
                    if verbose:
                        print(f"    Added mixed strategy candidate: "
                              f"{game.strategy_names[pure_strat_idx]}: {1.0-alpha:.2f}, "
                              f"{game.strategy_names[s]}: {alpha:.2f}")
        
        # If this is the first time we're finding beneficial deviations,
        # create a fully mixed strategy candidate
        if has_beneficial and len(confirmed_equilibria) > 0 and len(beneficial_strategies) > 0:
            # Create a uniform mixture over all strategies
            uniform_mix = torch.ones(num_strategies, device=device) / num_strategies
            uniform_support = set(range(num_strategies))
            uniform_candidate = SubgameCandidate(mixture=uniform_mix, support=uniform_support)
            unconfirmed_candidates.append(uniform_candidate)
            
            if verbose:
                print(f"    Added uniform mixture candidate over all strategies")
                
        return has_beneficial
    
    def explore_next_subgame():
        """
        Explore a new subgame by selecting the highest-gain deviation.
        """
        if len(deviation_queue) == 0:
            return
        
        # Get next deviation
        gain, strategy_idx, mixture = deviation_queue.pop()
        
        if verbose:
            print(f"  Exploring new subgame with strategy {game.strategy_names[strategy_idx]}, "
                  f"gain: {gain:.6f}")
        
        # Get current support
        current_support = set(torch.where(mixture > 1e-8)[0].cpu().numpy().tolist())
        
        # Add deviating strategy
        new_support = current_support.union({strategy_idx})
        
        # Check if this is a new maximal subgame
        if not maximal_subgames.is_contained(new_support) and maximal_subgames.add(new_support):
            # Create mixture biased toward new strategy
            new_mixture = mixture.clone()
            new_mixture *= 0.5  # Reduce weight of existing strategies
            new_mixture[strategy_idx] = 0.5  # Add weight to new strategy
            
            # Create candidate
            candidate = SubgameCandidate(mixture=new_mixture, support=new_support)
            unconfirmed_candidates.append(candidate)
            
            if verbose:
                print(f"    Added new candidate with support: {[game.strategy_names[s] for s in new_support]}")
    
    # Main QUIESCE loop
    iteration = 0
    while iteration < num_iters:
        if verbose:
            print(f"QUIESCE iteration {iteration+1}/{num_iters}")
            print(f"  Unconfirmed candidates: {len(unconfirmed_candidates)}")
            print(f"  Confirmed equilibria: {len(confirmed_equilibria)}")
            print(f"  Pending deviations: {len(deviation_queue)}")
        
        # Step 1: Test unconfirmed candidates
        if unconfirmed_candidates:
            await test_unconfirmed_candidates()
        
        # Step 2: If no unconfirmed candidates, explore new subgame from deviation queue
        elif len(deviation_queue) > 0:
            explore_next_subgame()
        
        # Step 3: If no deviations to explore, we're done
        else:
            if verbose:
                print("  No more candidates or deviations to explore. Done.")
            break
        
        iteration += 1
    
    # Sort confirmed equilibria by regret
    confirmed_equilibria.sort(key=lambda x: x[1])
    
    return confirmed_equilibria


def quiesce_sync(
    game: Game, 
    num_iters: int = 10,
    regret_threshold: float = 1e-4,
    dist_threshold: float = 1e-3,
    restricted_game_size: int = 4,
    solver: str = 'replicator',
    solver_iters: int = 1000,
    verbose: bool = True
) -> List[Tuple[torch.Tensor, float]]:
    """
    Synchronous wrapper for the asynchronous quiesce function.
    Args:
        Same as quiesce
        
    Returns:
        List of (mixture, regret) tuples, sorted by regret
    """
    try:
        import nest_asyncio
        nest_asyncio.apply()
    except ImportError:
        print("Warning: nest_asyncio not found. If running in a Jupyter notebook, you may need to install it: pip install nest_asyncio")
    
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(
        quiesce(
            game=game,
            num_iters=num_iters,
            regret_threshold=regret_threshold,
            dist_threshold=dist_threshold,
            restricted_game_size=restricted_game_size,
            solver=solver,
            solver_iters=solver_iters,
            verbose=verbose
        )
    ) 