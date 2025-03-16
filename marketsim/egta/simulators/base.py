"""
Base simulator interface for EGTA.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional


class Simulator(ABC):
    """
    Abstract base class for simulators.
    Simulators are responsible for evaluating strategy profiles and returning payoffs.
    """
    
    @abstractmethod
    def get_num_players(self) -> int:
        """
        Get the number of players in the game.
        
        Returns:
            Number of players
        """
        pass
    
    @abstractmethod
    def get_strategies(self) -> List[str]:
        """
        Get the list of available strategies.
        
        Returns:
            List of strategy names
        """
        pass
    
    @abstractmethod
    def simulate_profile(self, profile: List[str]) -> List[Tuple[int, str, float]]:
        """
        Simulate a single strategy profile and return payoffs.
        
        Args:
            profile: List of strategies, one for each player
            
        Returns:
            List of (player_id, strategy, payoff) tuples
        """
        pass
    
    def simulate_profiles(self, profiles: List[List[str]]) -> List[List[Tuple[int, str, float]]]:
        """
        Simulate multiple strategy profiles and return payoffs.
        Default implementation calls simulate_profile for each profile.
        
        Args:
            profiles: List of strategy profiles, each a list of strategies
            
        Returns:
            List of lists of (player_id, strategy, payoff) tuples
        """
        return [self.simulate_profile(profile) for profile in profiles]   
    


        