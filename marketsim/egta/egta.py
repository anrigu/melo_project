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
import matplotlib.pyplot as plt
from collections import defaultdict
import re

from marketsim.egta.core.game import Game
from marketsim.egta.schedulers.base import Scheduler
from marketsim.egta.schedulers.random import RandomScheduler
from marketsim.egta.schedulers.dpr import DPRScheduler
from marketsim.egta.simulators.base import Simulator
from marketsim.egta.solvers.equilibria import quiesce, quiesce_sync, replicator_dynamics, regret
from marketsim.game.symmetric_game import SymmetricGame


class PayoffAnalyzer:
    """
    A class to analyze and visualize payoff data for different strategy compositions.
    """
    
    def __init__(self, payoff_data=None):
        """
        Initialize the analyzer with payoff data.
        
        Args:
            payoff_data (dict): Dictionary with keys as strategy profiles and values as 
                               lists of payoffs [strategy1_payoffs, strategy2_payoffs]
        """
        self.payoff_data = payoff_data or {}
        self.processed_data = []
    
    def parse_strategy_key(self, key):
        """
        Parse strategy composition from key string.
        
        Args:
            key (str): Key string like "[MELO_0_100:5, MELO_100_0:26]"
            
        Returns:
            tuple: (strategy1_count, strategy2_count, strategy1_name, strategy2_name)
        """
        # Default strategy names    
        strategy1_name = "Strategy 1"
        strategy2_name = "Strategy 2"
        
        # Try to extract strategy names and counts
        pattern = r'\[([^:]+):(\d+)(?:,\s*([^:]+):(\d+))?\]'
        match = re.search(pattern, key)
        
        if match:
            strategy1_name = match.group(1).replace('_', '-')
            count1 = int(match.group(2))
            
            if match.group(3) and match.group(4):
                strategy2_name = match.group(3).replace('_', '-')
                count2 = int(match.group(4))
            else:
                count2 = 0
                
            return count1, count2, strategy1_name, strategy2_name
        
        return 0, 0, strategy1_name, strategy2_name
    
    def process_data(self, sort_by='count1'):
        """
        Process the raw payoff data into a format suitable for visualization.
        
        Args:
            sort_by (str): How to sort the data. Options:
                - 'count1': Sort by first strategy count (ascending)
                - 'count2': Sort by second strategy count (ascending)
                - 'total': Sort by total number of agents (ascending)
                - 'ratio': Sort by ratio of count1/(count1+count2) (ascending)
                - 'composition': Sort by composition for better visualization
        """
        self.processed_data = []
        
        for key, payoffs in self.payoff_data.items():
            count1, count2, name1, name2 = self.parse_strategy_key(key)
            
            # Get payoffs for each strategy
            payoffs1 = payoffs[0] if len(payoffs) > 0 else []
            payoffs2 = payoffs[1] if len(payoffs) > 1 else []
            
            # Calculate statistics
            mean1 = np.mean(payoffs1) if payoffs1 else np.nan
            std_err1 = np.std(payoffs1, ddof=1) / np.sqrt(len(payoffs1)) if payoffs1 else 0
            
            mean2 = np.mean(payoffs2) if payoffs2 else np.nan
            std_err2 = np.std(payoffs2, ddof=1) / np.sqrt(len(payoffs2)) if payoffs2 else 0
            
            total_agents = count1 + count2
            ratio = count1 / total_agents if total_agents > 0 else 0
            
            self.processed_data.append({
                'count1': count1,
                'count2': count2,
                'name1': name1,
                'name2': name2,
                'mean1': mean1,
                'std_err1': std_err1,
                'mean2': mean2,
                'std_err2': std_err2,
                'total_agents': total_agents,
                'ratio': ratio,
                'label': f"{name1}:{count1}, {name2}:{count2}",
                'payoffs1': payoffs1,
                'payoffs2': payoffs2
            })
        
        # Sort based on the specified method
        if sort_by == 'count1':
            self.processed_data.sort(key=lambda x: x['count1'])
        elif sort_by == 'count2':
            self.processed_data.sort(key=lambda x: x['count2'])
        elif sort_by == 'total':
            self.processed_data.sort(key=lambda x: x['total_agents'])
        elif sort_by == 'ratio':
            self.processed_data.sort(key=lambda x: x['ratio'])
        elif sort_by == 'composition':
            # Sort by ratio, but handle edge cases better for visualization
            self.processed_data.sort(key=lambda x: (x['ratio'], x['total_agents']))
        
        return self.processed_data
    
    def plot_payoffs(self, title="Average Payoffs by Strategy Composition", 
                    figsize=(12, 6), show_error_bars=False, colors=None, sort_by='composition'):
        """
        Create a bar plot of average payoffs.
        
        Args:
            title (str): Plot title
            figsize (tuple): Figure size
            show_error_bars (bool): Whether to show error bars
            colors (list): Colors for the bars [color1, color2]
            sort_by (str): How to sort the bars ('count1', 'count2', 'total', 'ratio', 'composition')
        """
        if not self.processed_data:
            self.process_data(sort_by=sort_by)
        
        # Default colors
        if colors is None:
            colors = ["#9ace69", "#44a2f0"]
        
        # Prepare data for plotting
        labels = [d['label'] for d in self.processed_data]
        means1 = [d['mean1'] if not np.isnan(d['mean1']) else 0 for d in self.processed_data]
        means2 = [d['mean2'] if not np.isnan(d['mean2']) else 0 for d in self.processed_data]
        
        if show_error_bars:
            std_errs1 = [d['std_err1'] if not np.isnan(d['mean1']) else 0 for d in self.processed_data]
            std_errs2 = [d['std_err2'] if not np.isnan(d['mean2']) else 0 for d in self.processed_data]
        else:
            std_errs1 = std_errs2 = None
        
        # Create plot
        x = np.arange(len(labels))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get strategy names for legend
        strategy1_label = self.processed_data[0]['name1'] if self.processed_data else "MELO"
        strategy2_label = self.processed_data[0]['name2'] if self.processed_data else "CDA"
        
        rects1 = ax.bar(x - width/2, means1, width, 
                       label=f'{strategy1_label} Strategy',
                       yerr=std_errs1, capsize=5, color=colors[0])
        rects2 = ax.bar(x + width/2, means2, width,
                       label=f'{strategy2_label} Strategy', 
                       yerr=std_errs2, capsize=5, color=colors[1])
        
        # Formatting
        ax.set_ylabel('Average Payoff', fontsize=18)
        ax.set_title(title, fontsize=18)
        ax.set_xlabel('Strategy Profiles', fontsize=18)
        ax.set_xticks(x)
        
        # Truncate labels to show only the counts
        truncated_labels = []
        for label in labels:
            # Remove the square brackets and split by comma
            clean_label = label.strip('[]')
            parts = clean_label.split(', ')
            
            # Initialize counts for both strategies
            melo_count = '0'
            cda_count = '0'
            
            # Process each part
            for part in parts:
                strat, count = part.split(':')
                if 'MELO-0-100' in strat:
                    melo_count = count
                elif 'MELO-100-0' in strat:
                    cda_count = count
            
            # Always show both strategies in the same format
            truncated_labels.append(f"MELO:{melo_count}, CDA:{cda_count}")
        
        # Sort labels based on MELO count first, then CDA count
        def sort_key(label):
            melo, cda = label.split(', ')
            melo_count = int(melo.split(':')[1])
            cda_count = int(cda.split(':')[1])
            return (melo_count, cda_count)
        
        truncated_labels.sort(key=sort_key)
        
        print(truncated_labels)
        ax.set_xticklabels(truncated_labels, rotation=45, ha='right', fontsize=16)
        ax.legend(fontsize=16, loc='lower right')
        ax.tick_params(axis='y', labelsize=15)
        
        plt.tight_layout()
        return fig, ax
    
    def get_summary_stats(self, sort_by='composition'):
        """
        Get summary statistics for all strategy compositions.
        
        Args:
            sort_by (str): How to sort the results
        """
        if not self.processed_data:
            self.process_data(sort_by=sort_by)
        
        summary = []
        for data in self.processed_data:
            summary.append({
                'composition': data['label'],
                'strategy1_count': data['count1'],
                'strategy2_count': data['count2'],
                'total_agents': data['total_agents'],
                'ratio': data['ratio'],
                'strategy1_mean': data['mean1'],
                'strategy1_std_err': data['std_err1'],
                'strategy1_n': len(data['payoffs1']),
                'strategy2_mean': data['mean2'],
                'strategy2_std_err': data['std_err2'],
                'strategy2_n': len(data['payoffs2'])
            })
        
        return summary


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
        
        os.makedirs(output_dir, exist_ok=True)
        
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
                
                # Store data for plotting
                payoff_vectors = {}
                
                for profile_data in new_data:
                    # Extract all strategies in this profile
                    all_strategies = [strat for _, strat, _ in profile_data]
                    # Count occurrences of each strategy
                    strategy_counts = {}
                    for strat in set(all_strategies):
                        strategy_counts[strat] = all_strategies.count(strat)
                    
                    
                    # Sort strategies by MELO_0_100 first, then MELO_100_0
                    sorted_strategies = sorted(strategy_counts.items(), 
                                            key=lambda x: (x[0] != "MELO_0_100", x[0]))
                    
                    # Format as a distribution string
                    profile_dist = ", ".join([f"{strat}:{count}" for strat, count in sorted_strategies])
                    
                    payoffs = [float(payoff) for _, _, payoff in profile_data]
                    # For MELO_0_100, take the last x elements where x is the count
                    melo_count = strategy_counts.get("MELO_0_100", 0)
                    cda_count = strategy_counts.get("MELO_100_0", 0)
                    
                    if melo_count > 0 and cda_count > 0:
                        # Both strategies present
                        melo_payoffs = payoffs[-melo_count:]
                        cda_payoffs = payoffs[:-melo_count]
                    elif melo_count > 0:
                        # Only MELO_0_100
                        melo_payoffs = payoffs
                        cda_payoffs = []
                    else:
                        # Only MELO_100_0
                        melo_payoffs = []
                        cda_payoffs = payoffs
                        
                    formatted_payoffs = [melo_payoffs, cda_payoffs]
                    
                    print(f'"{profile_dist}": {formatted_payoffs}')
                    print()
                    
                    # Store data for plotting
                    payoff_vectors[f"[{profile_dist}]"] = formatted_payoffs
                
                # Create analyzer and process data
                analyzer = PayoffAnalyzer(payoff_vectors)
                
                # Generate plot
                fig, ax = analyzer.plot_payoffs(
                    title="Average Payoffs by Strategy Composition",
                    show_error_bars=True,
                    sort_by='composition'
                )
                
                # Save the plot
                print("GRAPH OUTPUT DIR", self.output_dir)
                plot_path = os.path.join(self.output_dir, f"payoff_plot_{datetime.now().strftime('%Y%m%d_%H%M%S_eq_payoffs_2')}.png")
                fig.savefig(plot_path)
                plt.close()
                
                print(f"\nPayoff plot saved to: {plot_path}")
                
                # Print summary statistics
                print("\nSummary Statistics:")
                print("-" * 80)
                for stat in analyzer.get_summary_stats():
                    print(f"Composition: {stat['composition']}")
                    print(f"  Strategy 1: Mean={stat['strategy1_mean']:.2f}, SE={stat['strategy1_std_err']:.2f}, N={stat['strategy1_n']}")
                    print(f"  Strategy 2: Mean={stat['strategy2_mean']:.2f}, SE={stat['strategy2_std_err']:.2f}, N={stat['strategy2_n']}")
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
            else: #TODO remove for role symmetry
                reduced_game = self.game
            
            try:
                self.equilibria = quiesce_sync(
                    game=reduced_game,  # Use reduced game for equilibrium finding
                    num_iters=quiesce_kwargs['num_iters'],
                    num_random_starts=quiesce_kwargs['num_random_starts'],
                    regret_threshold=quiesce_kwargs['regret_threshold'],
                    dist_threshold=quiesce_kwargs['dist_threshold'],
                    solver=quiesce_kwargs['solver'],
                    solver_iters=quiesce_kwargs['solver_iters'],
                    verbose=verbose,
                    full_game=self.game if isinstance(self.scheduler, DPRScheduler) else None  # Pass full game for DPR for eq checking
                )
                
                if verbose and reduced_game.num_strategies == 2:
                    payoff_matrix = reduced_game.get_payoff_matrix()
                    print("\nReduced Game Payoff Matrix:")
                    for i in range(2):
                        print(f"  {reduced_game.strategy_names[i]}: [{payoff_matrix[i, 0].item():.4f}, {payoff_matrix[i, 1].item():.4f}]")
                    print()

                    
            except Exception as e:
                print(f"Error in equilibrium finding: {e}")
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
                
                for i, (eq_mix, eq_regret) in enumerate(self.equilibria):
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
                    
                    try:
                        is_dpr = isinstance(self.scheduler, DPRScheduler)
                        
                        if is_dpr:
                            reduced_payoffs = reduced_game.deviation_payoffs(eq_mix)
                            reduced_exp_payoff = (eq_mix * reduced_payoffs).sum().item()
                            
                            scaling_factor = self.scheduler.scaling_factor
                            full_exp_payoff = reduced_exp_payoff * scaling_factor
                            
                            print(f"Expected Payoff (Reduced Game): {reduced_exp_payoff:.4f}")
                            print(f"Expected Payoff (Full Game): {full_exp_payoff:.4f}")
                        else:
                            payoffs = self.game.deviation_payoffs(eq_mix)
                            exp_payoff = (eq_mix * payoffs).sum().item()
                            
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
        
        support_sizes = []
        strategy_frequencies = {name: 0.0 for name in self.game.strategy_names}
        
        for eq_mix, _ in self.equilibria:
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
    
  