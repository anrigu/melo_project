"""
Empirical Game-Theoretic Analysis (EGTA) framework.
"""
import os
import json
import time
from typing import Dict, List, Tuple, Any, Optional, Union, Set
import torch
import numpy as np
from datetime import datetime

from marketsim.egta.core.game import Game
from marketsim.egta.schedulers.base import Scheduler
from marketsim.egta.schedulers.dpr import DPRScheduler
from marketsim.egta.schedulers.random import RandomScheduler
from marketsim.egta.simulators.base import Simulator
from marketsim.egta.solvers.equilibria import quiesce_sync, replicator_dynamics, regret


class EGTA:
    """
    Empirical Game-Theoretic Analysis framework.
    """
    
    def __init__(self, 
                simulator: Simulator,
                scheduler: Optional[Scheduler] = None,
                device: str = "cpu",
                output_dir: str = "results",
                max_profiles: int = 100,
                seed: Optional[int] = None):
        """
        Initialize the EGTA framework.
        
        Args:
            simulator: Simulator to use for evaluating profiles
            scheduler: Scheduler for determining which profiles to simulate
                If None, uses RandomScheduler
            device: PyTorch device to use
            output_dir: Directory to save results
            max_profiles: Maximum number of profiles to simulate
            seed: Random seed
        """
        self.simulator = simulator
        self.device = device
        self.output_dir = output_dir
        self.max_profiles = max_profiles
        self.seed = seed
        
        # If scheduler is not provided, use RandomScheduler
        if scheduler is None:
            self.scheduler = RandomScheduler(
                strategies=simulator.get_strategies(),
                num_players=simulator.get_num_players(),
                seed=seed
            )
        else:
            self.scheduler = scheduler
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize empty game and equilibria
        self.game = None
        self.equilibria = []
        self.payoff_data = []
    
    def run(self, 
           max_iterations: int = 10, 
           profiles_per_iteration: int = 5,
           save_frequency: int = 1,
           verbose: bool = True) -> Game:
        """
        Run the EGTA process.
        
        Args:
            max_iterations: Maximum number of iterations to run
            profiles_per_iteration: Number of profiles to simulate in each iteration
            save_frequency: How often to save results (every N iterations)
            verbose: Whether to print progress
            
        Returns:
            The final Game object
        """
        start_time = time.time()
        
        # Initialize
        if verbose:
            print("Starting EGTA process")
            print(f"Simulator: {self.simulator.__class__.__name__}")
            print(f"Scheduler: {self.scheduler.__class__.__name__}")
            print(f"Strategies: {self.simulator.get_strategies()}")
            print(f"Number of players: {self.simulator.get_num_players()}")
            print(f"Device: {self.device}")
            print(f"Maximum profiles: {self.max_profiles}")
            print(f"Maximum iterations: {max_iterations}")
            print(f"Profiles per iteration: {profiles_per_iteration}")
            print()
            
        iteration = 0
        total_profiles = 0
        
        while iteration < max_iterations:
            iteration_start = time.time()
            if verbose:
                print(f"Iteration {iteration+1}/{max_iterations}")
            
            # Schedule profiles
            profiles_to_simulate = self.scheduler.get_next_batch(
                game=self.game
            )
            
            if not profiles_to_simulate:
                if verbose:
                    print("No more profiles to simulate. Ending early.")
                break
            
            # Simulate profiles
            if verbose:
                print(f"Simulating {len(profiles_to_simulate)} profiles...")
            
            simulation_start = time.time()
            new_data = self.simulator.simulate_profiles(profiles_to_simulate)
            simulation_time = time.time() - simulation_start
            
            if verbose:
                print(f"Simulation completed in {simulation_time:.2f} seconds")
            
            # Update payoff data
            self.payoff_data.extend(new_data)
            total_profiles += len(new_data)
            
            # Create or update game
            if self.game is None:
                # Create new game
                self.game = Game.from_payoff_data(
                    payoff_data=self.payoff_data,
                    device=self.device
                )
            else:
                # Update existing game
                self.game.update_with_new_data(new_data)
            
            # Update scheduler
            self.scheduler.update(self.game)
            
            # Find equilibria
            if verbose:
                print("Finding equilibria...")
            
            equilibria_start = time.time()
            self.equilibria = quiesce_sync(
                game=self.game,
                num_iters=3,
                regret_threshold=1e-4,
                restricted_game_size=min(4, len(self.game.strategy_names)),
                solver='replicator',
                solver_iters=1000,
                verbose=verbose
            )
            equilibria_time = time.time() - equilibria_start
            
            if verbose:
                print(f"Found {len(self.equilibria)} equilibria in {equilibria_time:.2f} seconds")
                
                # Print equilibria
                for i, (eq_mix, eq_regret) in enumerate(self.equilibria):
                    strat_str = ", ".join([
                        f"{self.game.strategy_names[s]}: {eq_mix[s].item():.4f}" 
                        for s in range(self.game.num_strategies)
                        if eq_mix[s].item() > 0.01
                    ])
                    print(f"  Equilibrium {i+1}: regret={eq_regret:.6f}, {strat_str}")
            
            # Save results
            if iteration % save_frequency == 0 or iteration == max_iterations - 1:
                self._save_results(iteration)
            
            iteration_time = time.time() - iteration_start
            if verbose:
                print(f"Iteration completed in {iteration_time:.2f} seconds")
                print(f"Total profiles: {total_profiles}/{self.max_profiles}")
            
            # Check if we've reached the maximum number of profiles
            if total_profiles >= self.max_profiles:
                if verbose:
                    print(f"Reached maximum number of profiles ({self.max_profiles}). Stopping.")
                break
            
            iteration += 1
        
        total_time = time.time() - start_time
        if verbose:
            print(f"\nEGTA completed in {total_time:.2f} seconds")
            print(f"Simulated {total_profiles} profiles")
            print(f"Found {len(self.equilibria)} equilibria")
        
        return self.game
    
    def _save_results(self, iteration: int):
        """
        Save current results to disk.
        
        Args:
            iteration: Current iteration
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save game
        game_path = os.path.join(self.output_dir, f"game_iter_{iteration}.json")
        self.game.save(game_path)
        
        # Save equilibria
        eq_path = os.path.join(self.output_dir, f"equilibria_iter_{iteration}.json")
        eq_data = []
        
        for i, (eq_mix, eq_regret) in enumerate(self.equilibria):
            eq_dict = {
                "id": i,
                "regret": float(eq_regret),
                "mixture": {
                    name: float(eq_mix[j].item())
                    for j, name in enumerate(self.game.strategy_names)
                    if eq_mix[j].item() > 0.001
                }
            }
            eq_data.append(eq_dict)
        
        with open(eq_path, 'w') as f:
            json.dump(eq_data, f, indent=2)
    
    def analyze_equilibria(self, verbose: bool = True) -> Dict[str, Any]:
        """
        Analyze the found equilibria.
        
        Args:
            verbose: Whether to print analysis
            
        Returns:
            Dictionary with analysis results
        """
        if not self.equilibria:
            if verbose:
                print("No equilibria found.")
            return {}
        
        # Support sizes
        support_sizes = []
        strategy_frequencies = {name: 0.0 for name in self.game.strategy_names}
        
        for eq_mix, _ in self.equilibria:
            # Count strategies in support
            support = sum(1 for x in eq_mix if x.item() > 0.01)
            support_sizes.append(support)
            
            # Add to strategy frequencies
            for i, name in enumerate(self.game.strategy_names):
                strategy_frequencies[name] += eq_mix[i].item() / len(self.equilibria)
        
        # Find most common strategies
        sorted_strategies = sorted(
            strategy_frequencies.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        results = {
            "num_equilibria": len(self.equilibria),
            "regrets": [float(regret) for _, regret in self.equilibria],
            "avg_support_size": sum(support_sizes) / len(support_sizes),
            "min_support_size": min(support_sizes),
            "max_support_size": max(support_sizes),
            "strategy_frequencies": strategy_frequencies,
            "top_strategies": sorted_strategies[:3]
        }
        
        if verbose:
            print("\nEquilibria Analysis")
            print(f"Number of equilibria: {results['num_equilibria']}")
            print(f"Average regret: {sum(results['regrets'])/len(results['regrets']):.6f}")
            print(f"Average support size: {results['avg_support_size']:.2f}")
            print(f"Support size range: {results['min_support_size']} - {results['max_support_size']}")
            print("\nTop strategies:")
            for strat, freq in results['top_strategies']:
                print(f"  {strat}: {freq:.4f}")
        
        return results 