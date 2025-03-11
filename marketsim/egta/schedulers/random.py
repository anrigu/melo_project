"""
Random scheduler for EGTA.
"""
import random
from typing import Dict, List, Tuple, Any, Optional, Set, Union
from itertools import combinations_with_replacement

from marketsim.egta.core.game import Game
from marketsim.egta.schedulers.base import Scheduler


class RandomScheduler(Scheduler):
    """
    Random scheduler for EGTA.
    Samples profiles uniformly at random from the set of all possible profiles.
    """
    
    def __init__(self, 
                strategies: List[str], 
                num_players: int, 
                batch_size: int = 10,
                seed: Optional[int] = None):
        """
        Initialize a random scheduler.
        
        Args:
            strategies: List of strategy names
            num_players: Number of players
            batch_size: Number of profiles to return per batch
            seed: Random seed
        """
        self.strategies = strategies
        self.num_players = num_players
        self.batch_size = batch_size
        self.rand = random.Random(seed)
        
        # Track profiles we've seen and scheduled
        self.scheduled_profiles: Set[Tuple[str, ...]] = set()
    
    def _sample_random_profile(self) -> List[str]:
        """
        Sample a random profile.
        
        Returns:
            A random profile as a list of strategies
        """
        # Sample a distribution over strategies
        profile = []
        for _ in range(self.num_players):
            profile.append(self.rand.choice(self.strategies))
        return profile
    
    def get_next_batch(self, game: Optional[Game] = None) -> List[List[str]]:
        """
        Get the next batch of profiles to simulate.
        
        Args:
            game: Optional game with existing data (not used for random scheduler)
            
        Returns:
            List of strategy profiles
        """
        new_profiles = []
        
        # Keep sampling until we have a full batch
        while len(new_profiles) < self.batch_size:
            profile = self._sample_random_profile()
            
            # Sort profile for consistent representation
            sorted_profile = tuple(sorted(profile))
            
            if sorted_profile not in self.scheduled_profiles:
                new_profiles.append(profile)
                self.scheduled_profiles.add(sorted_profile)
        
        return new_profiles
    
    def update(self, game: Game) -> None:
        """
        Update the scheduler with new game data.
        No-op for random scheduler.
        
        Args:
            game: Game with updated data
        """
        pass 