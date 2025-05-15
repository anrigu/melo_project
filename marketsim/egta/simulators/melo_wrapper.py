"""
MELO simulator interface for EGTA.
"""
import random
import numpy as np
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Any, Optional, Union
from tqdm import tqdm
from marketsim.egta.simulators.base import Simulator
from marketsim.simulator.melo_simulator import MELOSimulatorSampledArrival
from marketsim.agent.zero_intelligence_agent import ZIAgent
from marketsim.agent.melo_agent import MeloAgent
from marketsim.fourheap.constants import BUY, SELL


class MeloSimulator(Simulator):
    """
    Interface to the MELO simulator for EGTA.
    Shock, holding period, add an hbl
    15 - 8
    10 - 4
    graph payoffs 
    (N+1 - k)! / 15!
    """
    
    def __init__(self, 
                num_strategic: int = 10,
                sim_time: int = 10000,
                num_assets: int = 1,
                lam: float = 6e-3,
                mean: float = 1e6,
                r: float = 0.05,
                shock_var: float = 1e4,
                q_max: int = 15,
                pv_var: float = 5e6,
                shade: Optional[List[float]] = None,
                eta: float = 0.5,
                lam_r: Optional[float] = None,
                holding_period: int = 10,
                lam_melo: float = 1e-3,
                # Add these parameters
                num_zi: int = 15,
                num_hbl: int = 0,
                # num_melo: int = 10,
                reps: int = 50):
        """
        Starting points of rd
        5-10 arrivals 

        Initialize the MELO simulator interface.
        
        Args:
            num_players: Number of strategic agents
            sim_time: Simulation time
            num_assets: Number of assets
            lam: Arrival rate
            mean: Mean fundamental value
            r: Mean reversion rate
            shock_var: Shock variance
            q_max: Maximum quantity
            pv_var: Private value variance
            shade: Shade parameters
            eta: Eta parameter
            lam_r: Arrival rate for regular traders
            holding_period: Holding period
            lam_melo: Arrival rate for MELO traders
            num_zi: Number of zero intelligence agents
            num_hbl: Number of HBL agents
            reps: Number of simulation repetitions
        """
        self.num_strategic = num_strategic
        self.sim_time = sim_time
        self.num_assets = num_assets
        self.lam = lam
        self.mean = mean
        self.r = r
        self.shock_var = shock_var
        self.q_max = q_max
        self.pv_var = pv_var
        self.shade = shade or [10, 30]
        self.eta = eta
        self.lam_r = lam_r or lam
        self.holding_period = holding_period
        self.lam_melo = lam_melo
        self.num_zi = num_zi
        self.num_hbl = num_hbl
        # self.num_melo = num_melo
        self.reps = reps
        self.order_quantity = 5  # Fixed order quantity of 5 for MOBI traders as specified in the paper
        
        # Define strategies as allocation proportions between CDA and MELO
        # Format: "MELO_X_Y" where X is percentage in CDA and Y is percentage in MELO
        self.strategies = [
            "MELO_100_0",   # 100% CDA, 0% MELO
            #"MELO_75_25",   # 75% CDA, 25% MELO
            #"MELO_50_50",   # 50% CDA, 50% MELO
            #"MELO_25_75",   # 25% CDA, 75% MELO
            "MELO_0_100",   # 0% CDA, 100% MELO
        ]
        
        # Define strategy parameters - proportion of trades in each market
        self.strategy_params = {
            "MELO_100_0": {"cda_proportion": 1.0, "melo_proportion": 0.0},
            #"MELO_75_25": {"cda_proportion": 0.75, "melo_proportion": 0.25},
            #"MELO_50_50": {"cda_proportion": 0.5, "melo_proportion": 0.5},
            #"MELO_25_75": {"cda_proportion": 0.25, "melo_proportion": 0.75},
            "MELO_0_100": {"cda_proportion": 0.0, "melo_proportion": 1.0},
        }
    
    def get_num_players(self) -> int:
        """
        Get the number of players in the game.
        
        Returns:
            Number of players
        """
        return self.num_strategic
    
    def get_strategies(self) -> List[str]:
        """
        Get the list of available strategies.
        
        Returns:
            List of strategy names
        """
        return self.strategies
    
    def simulate_profile(self, profile: List[str]) -> List[Tuple[int, str, float]]:
        """
        Simulate a single strategy profile and return payoffs.
        
        Args:
            profile: List of strategies, one for each player
            
        Returns:
            List of (player_id, strategy, payoff) tuples
        """
        # Count strategy occurrences
        strategy_counts = Counter(profile)
        
        # Prepare result container
        all_results = []
        
        # Run multiple repetitions
        for rep_idx in tqdm(range(self.reps)):
            # try:
            # Create a background population
            num_background = self.num_zi + self.num_hbl
        
            # Initialize simulator
            sim = MELOSimulatorSampledArrival(
                num_background_agents=num_background,
                sim_time=self.sim_time,
                num_zi=self.num_zi,
                num_hbl=self.num_hbl,
                num_strategic=self.num_strategic,
                num_assets=self.num_assets,
                lam=self.lam,
                mean=self.mean,
                r=self.r,
                shock_var=self.shock_var,
                q_max=self.q_max,
                pv_var=self.pv_var,
                shade=self.shade,
                eta=self.eta,
                lam_r=self.lam_r,
                holding_period=self.holding_period,
                lam_melo=self.lam_melo,
                strategies = self.strategies,
                strategy_counts = strategy_counts,
                strategy_params = self.strategy_params
            )
            
            # Run the simulation
            sim.run()
            
            # Get final values and profits
            sim_results = sim.end_sim()
            values = sim_results[0]  # First element is the dictionary of agent values
            
            # Collect results for this repetition
            results = []
            player_id = 0
            for strategy in profile:
                # Get the agent ID for this player
                agent_id = num_background + player_id
                
                # Calculate total payoff (combining CDA and MELO profits based on allocation)
                # params = self.strategy_params[strategy]
                # cda_proportion = params["cda_proportion"]
                # melo_proportion = params["melo_proportion"]
                
                # Total payoff is weighted sum of payoffs from both mechanisms
                # EDIT: For now, since the proportions are {1,0}, we don't need the weighted sum
                # total_payoff = (values[agent_id] * cda_proportion) + (melo_profits[agent_id] * melo_proportion)
                if agent_id in values:
                    total_payoff = float(values[agent_id])
                else:
                    print(f"Warning: Agent {agent_id} not found in values. Using default payoff.")
                    total_payoff = 0.0
                
                results.append((player_id, strategy, total_payoff))
                player_id += 1
            
            all_results.append(results)
            # except Exception as e:
            #     print(f"Warning: Simulation repetition {rep_idx+1} failed with error: {e}")
                # Skip this repetition
        
        # Aggregate results across repetitions (with safety checks)
        aggregated_results = []
        
        if not all_results:
            # No successful simulations, return default values
            print("Warning: All simulation repetitions failed! Returning default payoffs of 0.")
            for player_id, strategy in enumerate(profile):
                aggregated_results.append((player_id, strategy, 0.0))
            return aggregated_results
        
        for player_id in range(len(profile)):
            strategy = profile[player_id]
            
            # Get payoffs, filtering out any potential non-numeric values
            payoffs = []
            for rep in all_results:
                try:
                    if player_id < len(rep):
                        payoff = rep[player_id][2]
                        if isinstance(payoff, (int, float)) and not np.isnan(payoff) and not np.isinf(payoff):
                            payoffs.append(float(payoff))
                except (IndexError, TypeError) as e:
                    # Skip this result
                    pass
            
            # Calculate average payoff with a fallback
            if payoffs:
                avg_payoff = sum(payoffs) / len(payoffs)
            else:
                print(f"Warning: No valid payoffs for player {player_id} with strategy {strategy}. Using default value of 0.")
                avg_payoff = 0.0
                
            aggregated_results.append((player_id, strategy, avg_payoff))
        
        return aggregated_results
    
    def simulate_profiles(self, profiles: List[List[str]]) -> List[List[Tuple[int, str, float]]]:
        """
        Simulate multiple strategy profiles and return payoffs.
        
        Args:
            profiles: List of strategy profiles, each a list of strategies
            
        Returns:
            List of lists of (player_id, strategy, payoff) tuples
        """
        results = []
        for i, profile in enumerate(profiles):
            print(f"Simulating profile {i+1}/{len(profiles)}: {profile}")
            profile_results = self.simulate_profile(profile)
            results.append(profile_results)
        return results 