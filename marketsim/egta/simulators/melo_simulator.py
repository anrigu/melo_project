"""
MELO simulator interface for EGTA.
"""
import random
import numpy as np
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Any, Optional, Union

from marketsim.egta.simulators.base import Simulator
from marketsim.simulator.melo_simulator import MELOSimulatorSampledArrival
from marketsim.agent.zero_intelligence_agent import ZIAgent
from marketsim.agent.melo_agent import MeloAgent
from marketsim.fourheap.constants import BUY, SELL


class MeloSimulator(Simulator):
    """
    Interface to the MELO simulator for EGTA.
    """
    
    def __init__(self, 
                num_players: int = 10,
                sim_time: int = 1000,
                num_assets: int = 1,
                lam: float = 0.1,
                mean: float = 100,
                r: float = 0.05,
                shock_var: float = 10,
                q_max: int = 10,
                pv_var: float = 5e6,
                shade: Optional[List[float]] = None,
                eta: float = 0.5,
                lam_r: Optional[float] = None,
                holding_period: int = 1,
                lam_melo: float = 0.1,
                reps: int = 5):
        """
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
            reps: Number of simulation repetitions
        """
        self.num_players = num_players
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
        self.reps = reps
        self.order_quantity = 5  # Fixed order quantity as specified
        
        # Define strategies as allocation proportions between CDA and MELO
        # Format: "MELO_X_Y" where X is percentage in CDA and Y is percentage in MELO
        self.strategies = [
            "MELO_100_0",   # 100% CDA, 0% MELO
            "MELO_75_25",   # 75% CDA, 25% MELO
            "MELO_50_50",   # 50% CDA, 50% MELO
            "MELO_25_75",   # 25% CDA, 75% MELO
            "MELO_0_100",   # 0% CDA, 100% MELO
        ]
        
        # Define strategy parameters - proportion of trades in each market
        self.strategy_params = {
            "MELO_100_0": {"cda_proportion": 1.0, "melo_proportion": 0.0},
            "MELO_75_25": {"cda_proportion": 0.75, "melo_proportion": 0.25},
            "MELO_50_50": {"cda_proportion": 0.5, "melo_proportion": 0.5},
            "MELO_25_75": {"cda_proportion": 0.25, "melo_proportion": 0.75},
            "MELO_0_100": {"cda_proportion": 0.0, "melo_proportion": 1.0},
        }
    
    def get_num_players(self) -> int:
        """
        Get the number of players in the game.
        
        Returns:
            Number of players
        """
        return self.num_players
    
    def get_strategies(self) -> List[str]:
        """
        Get the list of available strategies.
        
        Returns:
            List of strategy names
        """
        return self.strategies
    
    def _create_agent(self, agent_id: int, strategy: str, market, meloMarket):
        """
        Create a MELO agent with the specified market allocation strategy.
        
        Args:
            agent_id: Agent ID
            strategy: Strategy name (market allocation)
            market: Market instance
            meloMarket: MELO market instance
            
        Returns:
            Agent instance
        """
        params = self.strategy_params[strategy]
        
        # Create a MELO agent with specific allocation between markets
        return MeloAgent(
            agent_id=agent_id,
            market=market,
            meloMarket=meloMarket,
            q_max=self.q_max,
            pv_var=self.pv_var,
            cda_proportion=params["cda_proportion"],
            melo_proportion=params["melo_proportion"],
            order_quantity=self.order_quantity
        )
    
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
        for rep in range(self.reps):
            # Create a background population
            num_background = self.num_players // 2  # Half the agents are background
            
            # Initialize simulator
            sim = MELOSimulatorSampledArrival(
                num_background_agents=num_background,
                sim_time=self.sim_time,
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
                lam_melo=self.lam_melo
            )
            
            # Replace some agents with strategic agents
            strategic_agent_id = num_background + 1  # Start after background agents
            for strategy in self.strategies:
                count = strategy_counts[strategy]
                for _ in range(count):
                    # Create the agent with the specified market allocation strategy
                    agent = self._create_agent(
                        agent_id=strategic_agent_id,
                        strategy=strategy,
                        market=sim.market,
                        meloMarket=sim.meloMarket
                    )
                    
                    # Replace the agent in the simulator
                    sim.agents[strategic_agent_id] = agent
                    strategic_agent_id += 1
            
            # Run the simulation
            sim.run()
            
            # Get final values and profits
            values, melo_profits = sim.end_sim()
            
            # Collect results for this repetition
            results = []
            player_id = 0
            for strategy in profile:
                # Get the agent ID for this player
                agent_id = num_background + 1 + player_id
                
                # Calculate total payoff (combining CDA and MELO profits based on allocation)
                params = self.strategy_params[strategy]
                cda_proportion = params["cda_proportion"]
                melo_proportion = params["melo_proportion"]
                
                # Total payoff is weighted sum of payoffs from both mechanisms
                total_payoff = (values[agent_id] * cda_proportion) + (melo_profits[agent_id] * melo_proportion)
                
                results.append((player_id, strategy, total_payoff))
                player_id += 1
            
            all_results.append(results)
        
        # Aggregate results across repetitions
        aggregated_results = []
        for player_id in range(len(profile)):
            strategy = profile[player_id]
            avg_payoff = np.mean([rep[player_id][2] for rep in all_results])
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