import torch
import numpy as np
import subprocess
import json
import logging
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)

class MeloScheduler:
    """
    scheduler to connect Quiesce algorithm/EGTA with MELO simulator.
    
    this class handles:
    1. Converting game profiles to simulator parameters
    2. Running the simulator with those parameters
    3. Processing results back into game payoffs
    4. Returning Game Analysis information at each iteration
    """

    def __init__(self, 
                 strategy_names: List[str],
                 num_players: int,
                 simulator_path: str,
                 simulator_config: Dict[str, Any] = None,
                 cache_results: bool = True):
        """
        initialize the MELO scheduler.
        
        parameters:
        strategy_names : List[str]
            Names of strategies (e.g., "S_0.1", "S_0.2", etc.) 
        num_players : int
            Number of players in the game 
        simulator_path : str 
            Path to the MELO simulator executable
        simulator_config : Dict[str, Any] 
            Configuration parameters for the simulator
        cache_results : bool
            Whether to cache simulation results
        """
        self.strategy_names = strategy_names
        self.num_players = num_players
        self.simulator_path = simulator_path
        self.simulator_config = simulator_config or {}
        self.cache_results = cache_results
        self.results_cache = {}
        
        # Number of actions = number of strategies
        self.num_actions = len(strategy_names) 

    


