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
from marketsim.egta.schedulers.random import RandomScheduler
from marketsim.egta.schedulers.dpr import DPRScheduler
from marketsim.egta.simulators.base import Simulator
from marketsim.egta.solvers.equilibria import quiesce, quiesce_sync, replicator_dynamics, regret
from marketsim.game.symmetric_game import SymmetricGame


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
        
        # Game state
        self.game = None
        self.payoff_data = []
        self.simulated_profiles = set()
        self.equilibria = []
    
    def run(self, 
           max_iterations: int = 10, 
           profiles_per_iteration: int = 10,
           save_frequency: int = 1,
           verbose: bool = True,
           quiesce_kwargs: Optional[Dict] = None) -> Game:
        """
        Run the EGTA process.
        
        Args:
            max_iterations: Maximum number of iterations
            profiles_per_iteration: Number of profiles to simulate per iteration
            save_frequency: How often to save results (in iterations)
            verbose: Whether to print progress
            quiesce_kwargs: Optional dictionary of parameters to pass to quiesce_sync
            
        Returns:
            The final game
        """
        total_profiles = 0
        start_time = time.time()
        
        # Set default quiesce parameters if not provided
        if quiesce_kwargs is None:
            quiesce_kwargs = {
                'num_iters': 100,
                'num_random_starts': 10,
                'regret_threshold': 1e-3,
                'dist_threshold': 1e-2,
                'solver': 'replicator',
                'solver_iters': 5000
            }
        else:
            # Ensure all required parameters are present
            default_quiesce_kwargs = {
                'num_iters': 100,
                'num_random_starts': 60,
                'regret_threshold': 1e-2,
                'dist_threshold': 1e-2,
                'solver': 'replicator',
                'solver_iters': 5000
            }
            # Fill in any missing parameters with defaults
            for key, value in default_quiesce_kwargs.items():
                if key not in quiesce_kwargs:
                    quiesce_kwargs[key] = value
        
        for iteration in range(max_iterations):
            iteration_start = time.time()
            
            if verbose:
                print(f"\nIteration {iteration+1}/{max_iterations}")
            
            # FIXED SECTION: Always generate profiles for the full game, not reduced game
            if isinstance(self.scheduler, DPRScheduler):
                original_reduction_size = self.scheduler.reduction_size
                self.scheduler.reduction_size = self.scheduler.num_players
                profiles_to_simulate = self.scheduler.get_next_batch(self.game)[:profiles_per_iteration]
                self.scheduler.reduction_size = original_reduction_size
            else:
                profiles_to_simulate = self.scheduler.get_next_batch(self.game)[:profiles_per_iteration]
            
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
                
                # Print payoff data for debugging
                print("\nPayoff data from simulation:")
                for profile_data in new_data:
                    # Extract all strategies in this profile
                    all_strategies = [strat for _, strat, _ in profile_data]
                    # Count occurrences of each strategy
                    strategy_counts = {}
                    for strat in set(all_strategies):
                        strategy_counts[strat] = all_strategies.count(strat)
                    # Format as a distribution string
                    profile_dist = ", ".join([f"{strat}:{count}" for strat, count in strategy_counts.items()])
                    
                    payoffs = [float(payoff) for _, _, payoff in profile_data]
                    avg_payoff = sum(payoffs) / len(payoffs) if payoffs else 0
                    print(f"  Profile: [{profile_dist}], Avg Payoff: {avg_payoff:.4f}, Payoffs: {payoffs}")
                print()
            
            self.payoff_data.extend(new_data)
            total_profiles += len(new_data)
            
            if self.game is None:
                self.game = Game.from_payoff_data( #builds full game
                    payoff_data=self.payoff_data,
                    device=self.device
                )
            else:
                self.game.update_with_new_data(new_data)
            
            self.scheduler.update(self.game)
            
            if verbose:
                print("Finding equilibria...")
            
            equilibria_start = time.time()
            
            # Check if we're using DPR and create reduced game if so
            if isinstance(self.scheduler, DPRScheduler):
                if verbose:
                    print(f"Using DPR: Creating reduced game with {self.scheduler.reduction_size} players...")
                reduced_game = self._create_reduced_game(self.game, self.scheduler.reduction_size)
                
                if verbose:
                    print(f"Solving equilibria on reduced game (scaling factor: {self.scheduler.scaling_factor:.2f})")
            else:
                reduced_game = self.game
            
            # Always use quiesce_sync for equilibrium finding
            try:
                self.equilibria = quiesce_sync(
                    game=reduced_game,  # Use reduced game for equilibrium finding
                    num_iters=quiesce_kwargs['num_iters'],
                    num_random_starts=quiesce_kwargs['num_random_starts'],
                    regret_threshold=quiesce_kwargs['regret_threshold'],
                    dist_threshold=quiesce_kwargs['dist_threshold'],
                    solver=quiesce_kwargs['solver'],
                    solver_iters=quiesce_kwargs['solver_iters'],
                    verbose=verbose
                )
                
                # Print payoff matrix for debugging if it's a 2-strategy game
                if verbose and reduced_game.num_strategies == 2:
                    payoff_matrix = reduced_game.get_payoff_matrix()
                    print("\nReduced Game Payoff Matrix:")
                    for i in range(2):
                        print(f"  {reduced_game.strategy_names[i]}: [{payoff_matrix[i, 0].item():.4f}, {payoff_matrix[i, 1].item():.4f}]")
                    print()
                    
            except Exception as e:
                print(f"Error in equilibrium finding: {e}")
                # Fallback to direct replicator dynamics
                mixture = torch.ones(reduced_game.num_strategies, device=self.device) / reduced_game.num_strategies
                eq_mixture = replicator_dynamics(reduced_game, mixture, iters=5000)
                eq_regret = regret(reduced_game, eq_mixture)
                
                # Handle NaN regret
                if torch.is_tensor(eq_regret) and torch.isnan(eq_regret).any():
                    eq_regret = torch.tensor(0.01, device=self.device)
                if not torch.is_tensor(eq_regret) and (np.isnan(eq_regret) or np.isinf(eq_regret)):
                    eq_regret = 0.01
                    
                self.equilibria = [(eq_mixture, eq_regret)]
            
            equilibria_time = time.time() - equilibria_start
            
            if verbose:
                print(f"Found {len(self.equilibria)} equilibria in {equilibria_time:.2f} seconds")
                
                # Print equilibria
                for i, (eq_mix, eq_regret) in enumerate(self.equilibria):
                    # Skip equilibria with NaN regret
                    if torch.is_tensor(eq_regret) and torch.isnan(eq_regret).any():
                        continue
                    if not torch.is_tensor(eq_regret) and (np.isnan(eq_regret) or np.isinf(eq_regret)):
                        continue
                        
                    strat_str = ", ".join([
                        f"{self.game.strategy_names[s]}: {eq_mix[s].item():.4f}" 
                        for s in range(self.game.num_strategies)
                        if eq_mix[s].item() > 0.01
                    ])
                    print(f"  Equilibrium {i+1}: regret={float(eq_regret):.6f}, {strat_str}")
                    
                    # Show the expected payoff for this equilibrium
                    try:
                        # Is this a DPR reduced game?
                        is_dpr = isinstance(self.scheduler, DPRScheduler)
                        
                        # Calculate expected payoff based on the appropriate game
                        if is_dpr:
                            # Show both reduced and full game payoffs
                            reduced_payoffs = reduced_game.deviation_payoffs(eq_mix)
                            reduced_exp_payoff = (eq_mix * reduced_payoffs).sum().item()
                            
                            # Calculate full game expected payoff by scaling
                            scaling_factor = self.scheduler.scaling_factor
                            full_exp_payoff = reduced_exp_payoff * scaling_factor
                            
                            print(f"    Expected Payoff (Reduced Game): {reduced_exp_payoff:.4f}")
                            print(f"    Expected Payoff (Full Game): {full_exp_payoff:.4f}")
                        else:
                            # Regular game
                            payoffs = self.game.deviation_payoffs(eq_mix)
                            exp_payoff = (eq_mix * payoffs).sum().item()
                            
                            # Denormalize if necessary
                            if 'payoff_mean' in self.game.metadata and 'payoff_std' in self.game.metadata:
                                payoff_mean = self.game.metadata['payoff_mean']
                                payoff_std = self.game.metadata['payoff_std']
                                denorm_payoff = exp_payoff * payoff_std + payoff_mean
                                print(f"    Expected Payoff: {denorm_payoff:.4f}")
                            else:
                                print(f"    Expected Payoff: {exp_payoff:.4f}")
                    except Exception as e:
                        print(f"    Error calculating expected payoff: {e}")
            
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
            support = sum(1 for x in eq_mix if x.item() > 0.001)
            support_sizes.append(support)
            
            for i, name in enumerate(self.game.strategy_names):
                strategy_frequencies[name] += eq_mix[i].item() / len(self.equilibria)
        
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

    def _create_reduced_game(self, full_game, reduction_size):
        """
        Create a reduced game with fewer players for DPR.
        
        Args:
            full_game: The full game with original number of players
            reduction_size: Number of players in the reduced game
            
        Returns:
            Reduced game with rescaled payoffs
        """
        if not isinstance(self.scheduler, DPRScheduler):
            return full_game
            
        scaling_factor = (full_game.num_players - 1) / (reduction_size - 1)
        
        # Create a new SymmetricGame with reduced player count
        reduced_sym_game = SymmetricGame(
            num_players=reduction_size,
            num_actions=full_game.game.num_actions,
            config_table=full_game.game.config_table.clone(),
            payoff_table=full_game.game.payoff_table.clone() / scaling_factor,  # Rescale payoffs
            strategy_names=full_game.game.strategy_names,
            device=full_game.game.device,
            offset=full_game.game.offset,
            scale=full_game.game.scale
        )
        
        # Create metadata for the reduced game
        metadata = {**full_game.metadata} if full_game.metadata else {}
        metadata['dpr_reduced'] = True
        metadata['original_players'] = full_game.num_players
        metadata['reduced_players'] = reduction_size
        metadata['scaling_factor'] = scaling_factor
        
        return Game(reduced_sym_game, metadata) 