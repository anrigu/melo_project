"""
Base scheduler interface for EGTA.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional, Set

from marketsim.egta.core.game import Game
from marketsim.egta.simulators.base import Simulator


class Scheduler(ABC):
    """
    Abstract base class for strategy profile schedulers.
    Schedulers are responsible for deciding which profiles to simulate next.
    """
    
    @abstractmethod
    def get_next_batch(self, game: Optional[Game] = None) -> List[List[str]]:
        """
        Get the next batch of profiles to simulate.
        
        Args:
            game: Optional game with existing data
            
        Returns:
            List of strategy profiles
        """
        pass
    
    @abstractmethod
    def update(self, game: Game) -> None:
        """
        Update the scheduler with new game data.
        
        Args:
            game: Game with updated data
        """
        pass