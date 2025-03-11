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

    

    async def sample_payoffs(self, profiles: torch.Tensor) -> torch.Tensor:
        '''
        sample payoffs for a given profiles by running the M-ELO MarketSim Game

        parameters:
        profiles : torch.Tensor
            Tensor of profiles to sample payoffs for  
        returns:
        torch.Tensor : Payoffs for each profile and strategy
        '''

        payoffs = torch.zeros(self.num_actions, profiles.shape[1], device=profiles.device)

        for i in range(profiles.shape[1]):
            profile = profiles[:, i]
            profile_tuple = tuple(profile.cpu().numpy())

            #check cache first
            if self.cache_results and profile_tuple in self.results_cache:
                payoffs[:, i] = self.results_cache[profile_tuple]
                continue

            #convert profile to simulator format
            sim_input = self._profile_to_simulator_input(profile)

            #Run simulator 
            sim_result = await self._run_simulator(sim_input)

            #process Results 
            strat_payoffs = self._process_simulator_results(sim_result)

            #store results 
            payoffs[:, i] = torch.tensor(strat_payoffs, device=profiles.device)

            #cache if needed
            if self.cache_results:
                self.results_cache[profile_tuple] = payoffs[:, i]
                
        return payoffs
    
    def _profile_to_simulator_input(self, profile: torch.Tensor) -> Dict[str, Any]:
        '''
        convert profile to simulator format
        '''
        counts = profile.cpu().numpy().astype(int)

        # this create a configuration where each strategy is used by the specified count
        agents = []
        for strat_idx, count in enumerate(counts):
            if count > 0:
                strat_name = self.strategy_names[strat_idx]
                for _ in range(count):
                    agents.append({"strategy": strat_name})
        
        #combine with base simulator config
        sim_input = {
            **self.simulator_config,
            "agents": agents,
            "num_iterations": 100  # adjust as needed idk maybe 10k usually?
        }
    
        return sim_input
    
    async def _run_simulator(self, sim_input: Dict[str, Any]) -> Dict[str, Any]:
        '''
        run the M-ELO and CDA simulator with the given input
        '''
        input_file = "sim_input.json"
        output_file = "sim_output.json"

        with open(input_file, 'w') as f:
            json.dump(sim_input, f)
        
        # run simulator as subprocess
        # this could be replaced with direct Python API calls if available

        cmd = [self.simulator_path, "--input", input_file, "--output", output_file]
        process = await subprocess.create_subprocess_exec(*cmd)
        await process.wait()
        
        # Read results
        with open(output_file, 'r') as f:
            results = json.load(f)
            
        return results
    
    def _process_simulator_results(self, sim_result: Dict[str, Any]) -> List[float]:
        """Process simulator results into strategy payoffs"""
        # Extract payoffs for each strategy
        strategy_payoffs = [0.0] * self.num_actions
        
        #parse the payoffs from simulation results
        #this will depend on the exact format of your simulator output
        for agent in sim_result.get("agents", []):
            strategy_name = agent.get("strategy")
            payoff = agent.get("payoff", 0.0)
            
            #find the strategy index
            if strategy_name in self.strategy_names:
                strat_idx = self.strategy_names.index(strategy_name)
                strategy_payoffs[strat_idx] = payoff
        
        return strategy_payoffs
    
    





