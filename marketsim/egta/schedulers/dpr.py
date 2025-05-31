"""
Deviation Preserving Reduction (DPR) scheduler for EGTA.
Based on the original implementation from quiesce-master.
"""
import itertools
import random
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional, Set, Union
import torch
from marketsim.egta.core.game import Game
from marketsim.egta.schedulers.base import Scheduler


class DPRScheduler(Scheduler):
    """
    Deviation Preserving Reduction scheduler for EGTA.
    DPR reduces the number of profiles that need to be simulated
    by only requesting profiles needed to compute approximate equilibria.
    """
    
    def __init__(self, 
                strategies: List[str], 
                num_players: int, 
                subgame_size: int = 3,
                batch_size: int = 10,
                reduction_size: Optional[int] = None,
                seed: Optional[int] = None):
        """
        Initialize a DPR scheduler.
        
        Args:
            strategies: List of strategy names
            num_players: Number of players in the full game (N)
            subgame_size: Size of subgames to explore
            batch_size: Number of profiles to return per batch
            reduction_size: Number of players in the reduced game (n)
                If None, uses the full num_players (no reduction)
            seed: Random seed
        """
        self.strategies = strategies
        self.num_players = num_players
        self.subgame_size = min(subgame_size, len(strategies))
        self.batch_size = batch_size
        self.rand = random.Random(seed)
        self.game = None
        
        self.reduction_size = reduction_size if reduction_size is not None else num_players
        
        # Calculate the DPR scaling factor: (N-1)/(n-1)
        if self.reduction_size < self.num_players:
            self.scaling_factor = (self.num_players - 1) / (self.reduction_size - 1)
        else: 
            self.scaling_factor = 1.0

            
        self.scheduled_profiles: Set[Tuple[str, ...]] = set()
        self.requested_subgames: List[Set[str]] = []
        
        self._initialize_with_uniform_subgame()
    
    def _initialize_with_uniform_subgame(self):
        """Initialize with a uniform subgame."""
        # Choose strategies uniformly at random
        initial_strategies = set(self.rand.sample(self.strategies, self.subgame_size))
        self.requested_subgames.append(initial_strategies)
    
    def _generate_profiles_for_subgame(self, subgame: Set[str]) -> List[List[str]]:
        """
        Generate all profiles for a subgame.
        
        Args:
            subgame: Set of strategies in the subgame
            
        Returns:
            List of profiles
        """
        subgame_list = list(subgame)
        
        # Generate all possible distributions of players among strategies
        # Use the reduction_size instead of full num_players to reduce profile count
        profiles = []
        for counts in self._distribute_players(len(subgame_list), self.reduction_size):
            profile = []
            for i, count in enumerate(counts):
                profile.extend([subgame_list[i]] * count)
            profiles.append(profile)
        
        return profiles
    
    def _distribute_players(self, num_strategies: int, num_players: int) -> List[List[int]]:
        """
        Generate all ways to distribute players among strategies.
        
        Args:
            num_strategies: Number of strategies
            num_players: Number of players
            
        Returns:
            List of distributions, each a list of counts
        """
        if num_strategies == 1:
            return [[num_players]]
        
        result = []
        for i in range(num_players + 1):
            for sub_dist in self._distribute_players(num_strategies - 1, num_players - i):
                result.append([i] + sub_dist)
        
        return result
    
    def _select_equilibrium_candidates(self, game: Game, max_candidates: int = 10) -> List[np.ndarray]:
        """
        Select candidate equilibria from the game.
        
        Args:
            game: Game with existing data
            max_candidates: Maximum number of candidates to return
            
        Returns:
            List of candidate equilibrium mixtures
        """
        from marketsim.egta.solvers.equilibria import replicator_dynamics
        
        device = game.game.device
        strategy_mapping = {name: i for i, name in enumerate(game.strategy_names)}
        
        candidates = []
        
        for subgame in self.requested_subgames:
            subgame_indices = [strategy_mapping[s] for s in subgame if s in strategy_mapping]
            
            if len(subgame_indices) < len(subgame):
                continue
            
            # Create initial mixture over just this subgame
            mixture = np.zeros(len(game.strategy_names))
            mixture[subgame_indices] = 1.0 / len(subgame_indices)
            
           
            mixture_tensor = torch.tensor(mixture, dtype=torch.float32, device=device)
            
            eq_mixture = replicator_dynamics(game, mixture_tensor, iters=1000)
            
            candidates.append(eq_mixture.cpu().numpy())
        
        if not candidates:
            mixture = np.ones(len(game.strategy_names)) / len(game.strategy_names)
            candidates.append(mixture)
        
        if len(candidates) > max_candidates:
            candidates = self.rand.sample(candidates, max_candidates)
        
        return candidates
    
    def scale_payoffs(self, payoffs: torch.Tensor) -> torch.Tensor:
        """
        Apply DPR scaling to convert reduced game payoffs to full game payoffs.
        Uses the (N-1)/(n-1) scaling factor.
        
        Args:
            payoffs: Payoffs from the reduced game
        
        Returns:
            Scaled payoffs for the full game
        """
        if self.scaling_factor == 1.0:
            return payoffs
            
        return payoffs * self.scaling_factor
    
    def _select_deviating_strategies(self, game: Game, mixture: np.ndarray, num_deviations: int = 2) -> Set[str]:
        """
        Select strategies with highest deviation payoff.
        
        Args:
            game: Game with existing data
            mixture: Mixture to analyze
            num_deviations: Number of deviating strategies to select
            
        Returns:
            Set of strategy names
        """
        
        device = game.game.device
        mixture_tensor = torch.tensor(mixture, dtype=torch.float32, device=device)
        
        payoffs = game.deviation_payoffs(mixture_tensor)
        scaled_payoffs = self.scale_payoffs(payoffs)
        
        sorted_indices = np.argsort(-scaled_payoffs.cpu().numpy())
        
        deviating_indices = sorted_indices[:num_deviations]
        
        return {game.strategy_names[i] for i in deviating_indices}
    
    def _select_support_strategies(self, game: Game, mixture: np.ndarray, threshold: float = 0.01) -> Set[str]:
        """
        Select strategies that are played with significant probability.
        
        Args:
            game: Game instance
            mixture: Mixture to analyze
            threshold: Probability threshold
            
        Returns:
            Set of strategy names
        """
        support_indices = np.where(mixture > threshold)[0]
        return {game.strategy_names[i] for i in support_indices}
    
    def get_next_batch(self, game: Optional[Game] = None) -> List[List[str]]:
        """
        Get the next batch of profiles to simulate.
        Args:
            game: Optional game with existing data
        Returns:
            List of strategy profiles
        """
        if game is None:
            profiles_to_simulate = []
            for subgame in self.requested_subgames:
                profiles_to_simulate.extend(self._generate_profiles_for_subgame(subgame))
        else:
            self.game = game
            
            candidates = self._select_equilibrium_candidates(game)
            
            new_subgames = []
            for candidate in candidates:
                support_strategies = self._select_support_strategies(game, candidate)
                
                deviating_strategies = self._select_deviating_strategies(game, candidate)
                
                new_subgame = support_strategies.union(deviating_strategies)
                
                if len(new_subgame) > self.subgame_size:
                    new_subgame = set(list(new_subgame)[:self.subgame_size])
                
                new_subgames.append(new_subgame)
            
            self.requested_subgames.extend(new_subgames)
            
            profiles_to_simulate = []
            for subgame in new_subgames:
                profiles_to_simulate.extend(self._generate_profiles_for_subgame(subgame))
        
        new_profiles = []
        for profile in profiles_to_simulate:
            sorted_profile = tuple(sorted(profile))
            
            if sorted_profile not in self.scheduled_profiles:
                new_profiles.append(profile)
                self.scheduled_profiles.add(sorted_profile)
        
        self.rand.shuffle(new_profiles)
        return new_profiles[:self.batch_size]
    
    def update(self, game: Game) -> None:
        """
        Update the scheduler with new game data.
        
        Args:
            game: Game with updated data
        """
        self.game = game
        
    def get_scaling_info(self) -> Dict[str, float]:
        """
        Get information about the current DPR reduction and scaling.
        
        Returns:
            Dictionary with scaling information
        """
        return {
            "full_game_players": self.num_players,
            "reduced_game_players": self.reduction_size,
            "scaling_factor": self.scaling_factor,
            "is_reduced": self.num_players != self.reduction_size
        } 