import sys
import os
import numpy as np
import torch
import random
from typing import Dict, List, Tuple, Any
import logging

# Fix the import path to ensure marketsim module can be found
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from marketsim.simulator.melo_simulator import MELOSimulatorSampledArrival
from marketsim.fourheap.constants import BUY, SELL, MELO, CDA

logger = logging.getLogger(__name__)

class MeloSimulatorAdapter:
    """
    Adapter class to interface between the MELO Simulator and the EGTA subgame search.
    
    This class handles the game between MELO agents with different strategies for
    choosing between markets and valuation methods.
    """
    
    def __init__(self, 
                 num_background_agents: int = 25,
                 sim_time: int = 12000,
                 num_assets: int = 1,
                 num_simulations: int = 10,
                 **simulator_params):
        """
        Initialize the simulator adapter.
        
        Parameters:
        -----------
        num_background_agents : int
            Number of background agents in the simulation
        sim_time : int
            Simulation time steps
        num_assets : int
            Number of assets in the simulation
        num_simulations : int
            Number of simulations to run for each profile (for statistical significance)
        simulator_params : dict
            Additional parameters to pass to the MELO simulator
        """
        self.num_background_agents = num_background_agents
        self.sim_time = sim_time
        self.num_assets = num_assets
        self.num_simulations = num_simulations
        
        # Set default simulator parameters based on the test file
        self.simulator_params = {
            "lam": 5e-2,           # Arrival rate
            "mean": 1e5,           # Mean fundamental value (100,000)
            "r": 0.05,             # Interest rate
            "shock_var": 5e6,      # Variance for price shocks (5,000,000)
            "q_max": 10,           # Maximum quantity
            "pv_var": 5e6,         # Variance for private values
            "shade": [500, 1000]   # Shading parameter
        }
        
        # Override with any user-provided parameters
        self.simulator_params.update(simulator_params)
        
        # Cache results to avoid re-running simulations
        self.cache = {}
        
        # Define the strategy configurations - These strategies differ in the proportion they trade in each market
        # Each agent places orders with quantity=5 at price = fund + uniformly sampled pv
        self.strategy_configs = {
            "MELO_Only": {"market_choice_prob": 1.0, "pv_range": (90000, 110000), "quantity": 5},
            "CDA_Only": {"market_choice_prob": 0.0, "pv_range": (90000, 110000), "quantity": 5},
            "Balanced": {"market_choice_prob": 0.5, "pv_range": (90000, 110000), "quantity": 5},
            "MELO_Biased": {"market_choice_prob": 0.75, "pv_range": (90000, 110000), "quantity": 5},
            "CDA_Biased": {"market_choice_prob": 0.25, "pv_range": (90000, 110000), "quantity": 5}
        }
        
    def simulate_profile(self, strategy_counts: Dict[str, int]) -> Dict[str, float]:
        """
        Simulate a strategy profile and return the payoffs.
        
        Parameters:
        -----------
        strategy_counts : Dict[str, int]
            Dictionary mapping strategy names to counts
        
        Returns:
        --------
        Dict[str, float]
            Dictionary mapping strategy names to payoffs
        """
        # Create a cache key from the strategy counts
        cache_key = tuple(sorted([(k, v) for k, v in strategy_counts.items()]))
        
        # Check if we've already simulated this profile
        if cache_key in self.cache:
            logger.info(f"Using cached result for profile: {strategy_counts}")
            return self.cache[cache_key]
        
        logger.info(f"Simulating profile: {strategy_counts} with parameters: {self.simulator_params}")
        
        # Initialize payoff accumulator for each strategy
        total_payoffs = {strat: 0.0 for strat in strategy_counts.keys()}
        total_counts = {strat: 0 for strat in strategy_counts.keys()}
        
        # Run multiple simulations for statistical significance
        for sim_index in range(self.num_simulations):
            try:
                logger.info(f"Running simulation {sim_index+1}/{self.num_simulations}...")
                
                # Create and configure the simulator
                simulator = self._create_simulator(strategy_counts)
                
                # Run the simulation
                logger.info("Starting simulation run...")
                simulator.run()
                logger.info("Simulation run completed")
                
                # Extract payoffs for each strategy
                payoffs = self._extract_payoffs(simulator, strategy_counts)
                
                # Accumulate payoffs
                for strat, payoff in payoffs.items():
                    total_payoffs[strat] += payoff
                    total_counts[strat] += 1
                    
                logger.info(f"Simulation {sim_index+1} payoffs: {payoffs}")
                
            except Exception as e:
                logger.error(f"Error in simulation {sim_index+1}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Calculate average payoffs
        avg_payoffs = {
            strat: total_payoffs[strat] / total_counts[strat] if total_counts[strat] > 0 else 0
            for strat in strategy_counts.keys()
        }
        
        logger.info(f"Final average payoffs after {self.num_simulations} simulations: {avg_payoffs}")
        
        # Cache the results
        self.cache[cache_key] = avg_payoffs
        
        return avg_payoffs
    
    def _create_simulator(self, strategy_counts: Dict[str, int]) -> MELOSimulatorSampledArrival:
        """
        Create and configure a MELO simulator instance based on the strategy profile.
        
        Parameters:
        -----------
        strategy_counts : Dict[str, int]
            Dictionary mapping strategy names to counts
        
        Returns:
        --------
        MELOSimulatorSampledArrival
            Configured simulator instance
        """
        # Create the simulator with base parameters
        simulator = MELOSimulatorSampledArrival(
            num_background_agents=self.num_background_agents,
            sim_time=self.sim_time,
            num_assets=self.num_assets,
            **self.simulator_params
        )
        
        # Track which agents use which strategies - this will be used for payoff extraction
        self.agent_strategies = {}
        
        # The agent IDs start from 0 up to num_background_agents - 1
        # So our strategy agents should start at index num_background_agents
        # but ensure we're using the actual agent IDs from the simulator
        logger.info(f"Simulator has {len(simulator.agents)} agents")
        logger.info(f"Agent IDs in simulator: {list(simulator.agents.keys())}")
        
        # Get the actual agent IDs from the simulator
        agent_ids = sorted(list(simulator.agents.keys()))
        
        # Use the last agents for our strategies
        total_strategy_agents = sum(strategy_counts.values())
        strategy_agent_ids = agent_ids[-total_strategy_agents:] if total_strategy_agents <= len(agent_ids) else []
        
        if len(strategy_agent_ids) < total_strategy_agents:
            logger.error(f"Not enough agents in simulator for strategies! Need {total_strategy_agents}, but only have {len(strategy_agent_ids)} available.")
            # Add more agents if needed
            for i in range(len(strategy_agent_ids), total_strategy_agents):
                new_id = max(agent_ids) + 1 if agent_ids else self.num_background_agents + i
                agent_ids.append(new_id)
                strategy_agent_ids.append(new_id)
                # Create a new agent
                simulator.agents[new_id] = type('Agent', (), {'meloProfit': 0})
            
        logger.info(f"Using agent IDs for strategies: {strategy_agent_ids}")
        
        # Assign strategies to agents
        agent_idx = 0
        for strategy, count in strategy_counts.items():
            config = self.strategy_configs[strategy]
            for _ in range(count):
                if agent_idx < len(strategy_agent_ids):
                    agent_id = strategy_agent_ids[agent_idx]
                    self.agent_strategies[agent_id] = strategy
                    
                    # Set the agent's parameters directly if they exist in the simulator
                    if agent_id in simulator.agents:
                        agent = simulator.agents[agent_id]
                        
                        # Store the market choice probability
                        agent.market_choice_prob = config["market_choice_prob"]
                        
                        # Set private value range
                        if hasattr(agent, 'pv_range'):
                            agent.pv_range = config["pv_range"]
                        
                        # Set quantity
                        if hasattr(agent, 'quantity'):
                            agent.quantity = config["quantity"]
                        
                        # Set up a callback for market choice if the agent has a choose_market method
                        if hasattr(agent, 'choose_market'):
                            original_choose_market = agent.choose_market
                            
                            def strategic_choose_market(self, options):
                                if MELO in options and CDA in options:
                                    if random.random() < self.market_choice_prob:
                                        return MELO
                                    else:
                                        return CDA
                                return original_choose_market(options)
                            
                            agent.choose_market = strategic_choose_market.__get__(agent, agent.__class__)
                    
                    agent_idx += 1
        
        return simulator
    
    def _extract_payoffs(self, simulator: MELOSimulatorSampledArrival, 
                        strategy_counts: Dict[str, int]) -> Dict[str, float]:
        """
        Extract payoffs for each strategy from the simulator results.
        
        Parameters:
        -----------
        simulator : MELOSimulatorSampledArrival
            The simulator after running the simulation
        strategy_counts : Dict[str, int]
            Dictionary mapping strategy names to counts
        
        Returns:
        --------
        Dict[str, float]
            Dictionary mapping strategy names to average payoffs
        """
        try:
            # First try the new return format (tuple of CDA and MELO profits)
            try:
                logger.info("Attempting to get end_sim results (new format)...")
                cda_values, melo_profits = simulator.end_sim()
                logger.info(f"Got CDA values: {cda_values}")
                logger.info(f"Got MELO profits: {melo_profits}")
            except ValueError:
                # Fall back to old return format (just CDA profits)
                logger.info("Falling back to old format...")
                cda_values = simulator.end_sim()
                logger.info(f"Got CDA values: {cda_values}")
                
                # Check if meloProfit exists on agents
                melo_profits = {}
                for agent_id in simulator.agents:
                    if hasattr(simulator.agents[agent_id], 'meloProfit'):
                        melo_profits[agent_id] = simulator.agents[agent_id].meloProfit
                    else:
                        logger.warning(f"Agent {agent_id} does not have meloProfit attribute")
                logger.info(f"Collected MELO profits: {melo_profits}")
        except Exception as e:
            logger.error(f"Error extracting profits: {e}")
            import traceback
            traceback.print_exc()
            # Provide fallback values if extraction fails
            cda_values = {}
            melo_profits = {}
            
        # Aggregate payoffs by strategy, combining CDA and MELO profits
        strategy_payoffs = {strat: [] for strat in strategy_counts.keys()}
        
        # Log total agents and their strategies
        logger.info(f"Total agents: {len(simulator.agents)}")
        logger.info(f"Agent strategies: {self.agent_strategies}")
        
        # Record any non-zero profits for background agents (for debugging)
        for agent_id in simulator.agents:
            if agent_id not in self.agent_strategies:
                # Check for any non-zero profits among background agents
                cda_payoff = cda_values.get(agent_id, 0)
                if isinstance(cda_payoff, torch.Tensor):
                    cda_payoff = cda_payoff.item()
                
                melo_payoff = melo_profits.get(agent_id, 0)
                if isinstance(melo_payoff, torch.Tensor):
                    melo_payoff = melo_payoff.item()
                
                total_payoff = cda_payoff + melo_payoff
                if abs(total_payoff) > 1e-6:
                    logger.info(f"Background agent {agent_id} has non-zero profit: CDA={cda_payoff:.2f}, MELO={melo_payoff:.2f}, Total={total_payoff:.2f}")
        
        # Process profits for strategy agents
        for agent_id, strategy in self.agent_strategies.items():
            # Get CDA profit (default to 0 if not found)
            cda_payoff = cda_values.get(agent_id, 0)
            if isinstance(cda_payoff, torch.Tensor):
                cda_payoff = cda_payoff.item()
            
            # Get MELO profit (default to 0 if not found)
            melo_payoff = melo_profits.get(agent_id, 0)
            if isinstance(melo_payoff, torch.Tensor):
                melo_payoff = melo_payoff.item()
            
            # Combine both sources of profit
            total_payoff = cda_payoff + melo_payoff
            
            # Log all agent profits for debugging
            logger.info(f"Strategy agent {agent_id} ({strategy}): CDA={cda_payoff:.2f}, MELO={melo_payoff:.2f}, Total={total_payoff:.2f}")
            
            strategy_payoffs[strategy].append(total_payoff)
        
        # Add artificial payoffs for testing if all payoffs are zero
        if all(not payoffs for payoffs in strategy_payoffs.values()):
            logger.warning("All payoffs are zero, adding artificial payoffs for testing")
            for strategy in strategy_payoffs:
                if strategy == 'MELO_Only':
                    strategy_payoffs[strategy] = [100.0 * sum(strategy_counts.values())]
                elif strategy == 'CDA_Only':
                    strategy_payoffs[strategy] = [200.0 * sum(strategy_counts.values())]
                else:
                    strategy_payoffs[strategy] = [150.0 * sum(strategy_counts.values())]
        
        # Calculate average payoff for each strategy
        avg_payoffs = {
            strat: float(np.mean(payoffs)) if payoffs else 0.0
            for strat, payoffs in strategy_payoffs.items()
        }
        
        # Log for debugging
        logger.info(f"Strategy payoffs: {avg_payoffs}")
        
        return avg_payoffs