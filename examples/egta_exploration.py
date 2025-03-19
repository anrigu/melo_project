#!/usr/bin/env python
# coding: utf-8

"""
EGTA Framework Command Line Script

This script runs Empirical Game-Theoretic Analysis (EGTA) to explore 
strategic interactions between agents in markets.

Example usage:
    python egta_exploration.py --output_dir results/my_experiment --num_strategic 15 --sim_time 100
"""

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import sys
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
from tqdm import tqdm
import time
from datetime import datetime

# Fix path to include the project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"Added {project_root} to Python path")

try:
    import marketsim
    print("Successfully imported marketsim module")
except ImportError as e:
    print(f"Error importing marketsim: {e}")
    print("Please make sure marketsim is installed or the path is correctly set")
    sys.exit(1)

from marketsim.egta.egta import EGTA
from marketsim.egta.simulators.melo_wrapper import MeloSimulator
from marketsim.egta.schedulers.dpr import DPRScheduler
from marketsim.egta.schedulers.random import RandomScheduler
from marketsim.egta.utils.visualization import (
    plot_equilibria, 
    plot_strategy_frequency, 
    plot_payoff_matrix,
    plot_regret_landscape,
    create_visualization_report
)
from marketsim.egta.core.game import Game
from marketsim.egta.solvers.equilibria import replicator_dynamics, quiesce, regret

# Set up plotting style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['font.size'] = 12


class CustomMeloSimulator(MeloSimulator):
    """Extended MeloSimulator with custom strategy space."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # more fine-grained strategy space
        self.strategies = [
            "MELO_100_0",   # 100% CDA, 0% MELO
            "MELO_90_10",   # 90% CDA, 10% MELO
            "MELO_80_20",   # 80% CDA, 20% MELO
            "MELO_70_30",   # 70% CDA, 30% MELO
            "MELO_60_40",   # 60% CDA, 40% MELO
            "MELO_50_50",   # 50% CDA, 50% MELO
            "MELO_40_60",   # 40% CDA, 60% MELO
            "MELO_30_70",   # 30% CDA, 70% MELO
            "MELO_20_80",   # 20% CDA, 80% MELO
            "MELO_10_90",   # 10% CDA, 90% MELO
            "MELO_0_100",   # 0% CDA, 100% MELO
        ]
        
        # strategy parameters
        self.strategy_params = {}
        for strategy in self.strategies:
            # Parse the strategy name to get proportions
            parts = strategy.split('_')
            cda_prop = int(parts[1]) / 100
            melo_prop = int(parts[2]) / 100
            self.strategy_params[strategy] = {
                "cda_proportion": cda_prop,
                "melo_proportion": melo_prop
            }


def setup_simulator(args):
    """Create a simulator with the specified parameters."""
    if args.custom_strategy_space:
        simulator = CustomMeloSimulator(
            num_strategic=args.num_strategic,
            sim_time=args.sim_time,
            lam=args.lam,
            mean=args.mean,
            r=args.r,
            q_max=args.q_max,
            holding_period=args.holding_period,
            reps=args.reps,
            num_zi=args.num_zi
        )
    else:
        simulator = MeloSimulator(
            num_strategic=args.num_strategic,
            sim_time=args.sim_time,
            lam=args.lam,
            mean=args.mean,
            r=args.r,
            q_max=args.q_max,
            holding_period=args.holding_period,
            reps=args.reps,
            num_zi=args.num_zi
        )
    
    return simulator


def setup_scheduler(args, strategies, num_players):
    """Create a scheduler with the specified parameters."""
    # Set batch size to match profiles_per_iteration to ensure we get enough profiles
    effective_batch_size = max(args.batch_size, args.profiles_per_iteration)
    
    if args.scheduler_type == 'dpr':
        scheduler = DPRScheduler(
            strategies=strategies,
            num_players=num_players,
            reduction_size=args.reduction_size,
            subgame_size=min(args.subgame_size, len(strategies)),
            batch_size=effective_batch_size,
            seed=args.seed
        )
    else:
        scheduler = RandomScheduler(
            strategies=strategies,
            num_players=num_players,
            batch_size=effective_batch_size,
            seed=args.seed
        )
    
    return scheduler


def run_egta_experiment(args):
    """Run a single EGTA experiment with the specified parameters."""
    print("Setting up simulator...")
    simulator = setup_simulator(args)
    
    # Get available strategies
    strategies = simulator.get_strategies()
    print(f"Available strategies: {strategies}")
    
    # Print strategy parameters
    print("\nStrategy parameters:")
    for strategy in strategies:
        params = simulator.strategy_params[strategy]
        print(f"  {strategy}: CDA={params['cda_proportion']:.2f}, MELO={params['melo_proportion']:.2f}")
    
    print("Setting up scheduler...")
    scheduler = setup_scheduler(args, strategies, simulator.get_num_players())
    
    if args.scheduler_type == 'dpr':
        print(f"DPR Scaling Info: {scheduler.get_scaling_info()}")
    
    # Create EGTA framework
    print("Creating EGTA framework...")
    egta = EGTA(
        simulator=simulator,
        scheduler=scheduler,
        device=args.device,
        output_dir=args.output_dir,
        max_profiles=args.max_profiles,
        seed=args.seed
    )
    
    # Run EGTA
    print(f"Starting EGTA experiment...")
    print(f"Analyzing market allocation strategies: {strategies}")
    
    start_time = time.time()
    game = egta.run(
        max_iterations=args.max_iterations,
        profiles_per_iteration=args.profiles_per_iteration,
        save_frequency=args.save_frequency,
        verbose=args.verbose,
        quiesce_kwargs={
            'num_random_starts': args.num_random_starts,  # Add more random starts to find multiple equilibria
            'regret_threshold': args.regret_threshold,
            'dist_threshold': args.dist_threshold,
            'solver_iters': args.solver_iters
        }
    )
    run_time = time.time() - start_time
    
    print(f"\nEGTA experiment completed in {run_time:.2f} seconds!")
    
    # Analyze equilibria
    print("Analyzing equilibria...")
    analysis = egta.analyze_equilibria(verbose=args.verbose)
    
    # Save analysis
    analysis_file = os.path.join(args.output_dir, 'analysis.json')
    with open(analysis_file, 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"Analysis saved to {analysis_file}")
    
    # Print all equilibria in detail
    print(f"\nFound {len(egta.equilibria)} equilibria:")
    for i, (mixture, regret) in enumerate(egta.equilibria):
        print(f"\nEquilibrium {i+1} (regret: {regret:.6f}):")
        # Sort strategies by their probability in this equilibrium
        strat_probs = [(strat, float(prob)) for strat, prob in zip(game.strategy_names, mixture)]
        strat_probs.sort(key=lambda x: x[1], reverse=True)
        # Print all strategies with non-zero probability
        for strat, prob in strat_probs:
            if prob > 0.001:  # Only show strategies with meaningful probability
                print(f"  {strat}: {prob:.4f}")
    
    # Print top strategies
    print("\nTop strategies by frequency across all equilibria:")
    for strategy, freq in analysis['top_strategies']:
        print(f"  {strategy}: {freq:.4f}")
    
    # Create visualizations if requested
    if args.create_visualizations:
        create_visualizations(args, egta, game)
    
    return egta, game, analysis


def create_visualizations(args, egta, game):
    """Create and save visualizations."""
    print("Creating visualizations...")
    vis_dir = os.path.join(args.output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Strategy frequencies plot
    strategy_freqs = pd.Series(egta.analyze_equilibria(verbose=False)['strategy_frequencies'])
    plt.figure(figsize=(12, 8))
    strategy_freqs.sort_values(ascending=False).plot(kind='bar')
    plt.title('Strategy Frequencies Across Equilibria')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'strategy_frequencies.png'), dpi=300)
    plt.close()
    
    # Create a detailed visualization of all equilibria
    num_eq = len(egta.equilibria)
    strategies = game.strategy_names
    
    # Create a DataFrame with all equilibria
    eq_data = []
    for i, (mixture, regret) in enumerate(egta.equilibria):
        eq_dict = {strat: float(prob) for strat, prob in zip(strategies, mixture)}
        eq_dict['Equilibrium'] = f"Eq {i+1} (r={regret:.4f})"
        eq_data.append(eq_dict)
    
    if eq_data:
        eq_df = pd.DataFrame(eq_data)
        eq_df.set_index('Equilibrium', inplace=True)
        
        # Plot heatmap of all equilibria
        plt.figure(figsize=(14, 10))
        sns.heatmap(eq_df.T, annot=True, cmap='viridis', fmt='.3f', linewidths=.5)
        plt.title('All Equilibria Comparison')
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, 'all_equilibria_heatmap.png'), dpi=300)
        plt.close()
    
    try:
        # Plot equilibria
        plot_equilibria(
            game=game,
            equilibria=egta.equilibria,
            output_file=os.path.join(vis_dir, 'equilibria.png'),
            show=False
        )
        
        # Plot strategy frequency
        plot_strategy_frequency(
            equilibria=egta.equilibria,
            strategy_names=game.strategy_names,
            output_file=os.path.join(vis_dir, 'strategy_frequency.png'),
            show=False
        )
        
        # Plot payoff matrix
        plot_payoff_matrix(
            game=game,
            output_file=os.path.join(vis_dir, 'payoff_matrix.png'),
            show=False
        )
        
        # Plot regret landscapes for pairs of strategies (if not too many)
        if len(game.strategy_names) <= 5 and args.plot_regret_landscapes:
            for i in range(len(game.strategy_names)):
                for j in range(i+1, len(game.strategy_names)):
                    plot_regret_landscape(
                        game=game,
                        strategies=[game.strategy_names[i], game.strategy_names[j]],
                        resolution=50,
                        output_file=os.path.join(vis_dir, f'regret_landscape_{i}_{j}.png'),
                        show=False
                    )
        
        # Create comprehensive visualization report
        if args.create_report:
            report_path = create_visualization_report(
                game=game,
                equilibria=egta.equilibria,
                output_dir=vis_dir
            )
            print(f"Visualization report saved to {report_path}")
    except Exception as e:
        print(f"Error creating some visualizations: {e}")


def run_parameter_sweep(args):
    """Run a parameter sweep across holding periods and lambda values."""
    if not args.run_parameter_sweep:
        return
    
    print("\nRunning parameter sweep...")
    sweep_dir = os.path.join(args.output_dir, 'parameter_sweep')
    os.makedirs(sweep_dir, exist_ok=True)
    
    # Define parameter grid
    holding_periods = [int(h) for h in args.sweep_holding_periods.split(',')]
    lambdas = [float(l) for l in args.sweep_lambdas.split(',')]
    
    print(f"Parameter grid: holding_periods={holding_periods}, lambdas={lambdas}")
    
    # Function to run a single experiment
    def run_experiment(holding_period, lam):
        """Run an EGTA experiment with specific parameters."""
        exp_dir = f"{sweep_dir}/h{holding_period}_lam{lam}"
        os.makedirs(exp_dir, exist_ok=True)
        
        # Create simulator with specific parameters
        sim_args = argparse.Namespace(**vars(args))
        sim_args.holding_period = holding_period
        sim_args.lam = lam
        sim_args.output_dir = exp_dir
        sim_args.max_iterations = args.sweep_max_iterations
        sim_args.profiles_per_iteration = args.sweep_profiles_per_iteration
        sim_args.verbose = False
        sim_args.create_visualizations = False
        
        egta, game, analysis = run_egta_experiment(sim_args)
        
        return {
            'holding_period': holding_period,
            'lam': lam,
            'equilibria': [(mix.tolist(), float(reg)) for mix, reg in egta.equilibria],
            'strategy_frequencies': analysis['strategy_frequencies'],
            'top_strategies': analysis['top_strategies']
        }
    
    # Run experiments
    results = []
    for h in holding_periods:
        for lam in lambdas:
            print(f"\nRunning sweep experiment with holding_period={h}, lambda={lam}")
            result = run_experiment(h, lam)
            results.append(result)
            print(f"Completed experiment. Found {len(result['equilibria'])} equilibria.")
    
    # Save all results
    with open(os.path.join(sweep_dir, 'parameter_sweep_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create visualization
    sweep_df = pd.DataFrame(results)
    
    def get_top_strategy(row):
        if row['top_strategies']:
            return row['top_strategies'][0][0]
        return None
    
    sweep_df['top_strategy'] = sweep_df.apply(get_top_strategy, axis=1)
    
    pivot = pd.pivot_table(
        sweep_df,
        values='top_strategy',
        index='holding_period',
        columns='lam',
        aggfunc=lambda x: x
    )
    
    print("Top strategy by holding period and arrival rate:")
    print(pivot)
    
    # Create parameter sweep heatmap
    plt.figure(figsize=(15, 10))
    fig, axes = plt.subplots(len(holding_periods), len(lambdas), figsize=(15, 10))
    fig.suptitle('Strategy Frequencies by Market Conditions', fontsize=16)
    
    for i, h in enumerate(holding_periods):
        if len(holding_periods) > 1:
            axes[i, 0].set_ylabel(f'Holding Period = {h}')
        else:
            axes[0].set_ylabel(f'Holding Period = {h}')
    
    for j, lam in enumerate(lambdas):
        if len(lambdas) > 1:
            axes[0, j].set_title(f'λ = {lam}')
        else:
            axes[0].set_title(f'λ = {lam}')
    
    for i, h in enumerate(holding_periods):
        for j, lam in enumerate(lambdas):
            result = next((r for r in results if r['holding_period'] == h and r['lam'] == lam), None)
            if result:
                freqs = pd.Series(result['strategy_frequencies'])
                if len(holding_periods) > 1 and len(lambdas) > 1:
                    ax = axes[i, j]
                elif len(holding_periods) > 1:
                    ax = axes[i]
                elif len(lambdas) > 1:
                    ax = axes[j]
                else:
                    ax = axes
                freqs.plot(kind='bar', ax=ax)
                ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
                ax.set_ylim(0, 1)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(sweep_dir, 'parameter_sweep_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Parameter sweep results saved to {sweep_dir}")


def run_holding_period_sweep(args):
    """Run a sweep across different holding periods."""
    if not args.run_holding_sweep:
        return
    
    print("\nRunning holding period sweep...")
    sweep_dir = os.path.join(args.output_dir, 'holding_sweep')
    os.makedirs(sweep_dir, exist_ok=True)
    
    holding_periods = [int(h) for h in args.sweep_holding_periods.split(',')]
    
    sweep_results = []
    for holding_period in tqdm(holding_periods):
        # Create experiment directory
        exp_dir = f"{sweep_dir}/holding_{holding_period}"
        os.makedirs(exp_dir, exist_ok=True)
        
        # Create simulator with specific parameters
        sim_args = argparse.Namespace(**vars(args))
        sim_args.holding_period = holding_period
        sim_args.output_dir = exp_dir
        sim_args.max_iterations = args.sweep_max_iterations
        sim_args.profiles_per_iteration = args.sweep_profiles_per_iteration
        sim_args.verbose = False
        sim_args.create_visualizations = False
        sim_args.custom_strategy_space = True  # Use fine-grained strategies
        
        egta, game, analysis = run_egta_experiment(sim_args)
        
        # Extract CDA proportions from top strategies
        cda_proportions = []
        for eq_mix, _ in egta.equilibria:
            eq_strat_freqs = dict(zip(game.strategy_names, eq_mix.tolist()))
            weighted_cda_prop = 0
            for strat, freq in eq_strat_freqs.items():
                if freq > 0.01:  # Only count strategies with significant probability
                    cda_prop = egta.simulator.strategy_params[strat]['cda_proportion']
                    weighted_cda_prop += cda_prop * freq
            cda_proportions.append(weighted_cda_prop)
        
        # Calculate average CDA proportion
        avg_cda_prop = np.mean(cda_proportions) if cda_proportions else None
        
        # Save results
        result = {
            'holding_period': holding_period,
            'avg_cda_proportion': avg_cda_prop,
            'cda_proportions': cda_proportions,
            'num_equilibria': len(egta.equilibria),
            'top_strategies': analysis['top_strategies']
        }
        sweep_results.append(result)
    
    # Save results
    with open(os.path.join(sweep_dir, 'holding_period_sweep_results.json'), 'w') as f:
        json.dump(sweep_results, f, indent=2)
    
    # Create a dataframe from results
    holding_df = pd.DataFrame(sweep_results)
    
    # Plot the relationship between holding period and CDA proportion
    plt.figure(figsize=(10, 6))
    plt.plot(holding_df['holding_period'], holding_df['avg_cda_proportion'], 'o-', linewidth=2, markersize=8)
    plt.xlabel('MELO Holding Period')
    plt.ylabel('Average CDA Proportion in Equilibrium')
    plt.title('Optimal CDA Allocation vs. MELO Holding Period')
    plt.grid(True)
    plt.ylim(0, 1)
    plt.savefig(os.path.join(sweep_dir, 'optimal_allocation_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save the dataframe as CSV
    holding_df.to_csv(os.path.join(sweep_dir, 'holding_period_sweep_results.csv'), index=False)
    
    print(f"Holding period sweep results saved to {sweep_dir}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run EGTA experiments for market allocation strategies.')
    
    # Output directory
    parser.add_argument('--output_dir', type=str, default='results/egta_run',
                        help='Directory to save results (default: results/egta_run)')
    
    # Simulator parameters
    parser.add_argument('--num_strategic', type=int, default=15,
                        help='Number of strategic agents (default: 15)')
    parser.add_argument('--sim_time', type=int, default=100,
                        help='Simulation time horizon (default: 100)')
    parser.add_argument('--lam', type=float, default=0.01,
                        help='Arrival rate (default: 0.01)')
    parser.add_argument('--mean', type=float, default=100,
                        help='Mean fundamental value (default: 100)')
    parser.add_argument('--r', type=float, default=0.05,
                        help='Mean reversion rate (default: 0.05)')
    parser.add_argument('--q_max', type=int, default=10,
                        help='Maximum inventory (default: 10)')
    parser.add_argument('--holding_period', type=int, default=1,
                        help='MELO holding period (default: 1)')
    parser.add_argument('--reps', type=int, default=100,
                        help='Number of simulation repetitions per profile (default: 100)')
    parser.add_argument('--num_zi', type=int, default=15,
                        help='Number of ZI agents (default: 15)')
    parser.add_argument('--custom_strategy_space', action='store_true',
                        help='Use fine-grained strategy space (default: False)')
    
    # Scheduler parameters
    parser.add_argument('--scheduler_type', type=str, choices=['dpr', 'random'], default='dpr',
                        help='Type of scheduler to use (default: dpr)')
    parser.add_argument('--reduction_size', type=int, default=8,
                        help='DPR reduction size (default: 8)')
    parser.add_argument('--subgame_size', type=int, default=2,
                        help='Subgame size for DPR (default: 2)')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='Batch size for scheduler (default: 10)')
    
    # EGTA run parameters
    parser.add_argument('--max_profiles', type=int, default=1000,
                        help='Maximum number of profiles to simulate (default: 1000)')
    parser.add_argument('--max_iterations', type=int, default=100,
                        help='Maximum number of iterations (default: 100)')
    parser.add_argument('--profiles_per_iteration', type=int, default=1000,
                        help='Number of profiles to simulate per iteration (default: 1000)')
    parser.add_argument('--save_frequency', type=int, default=1,
                        help='How often to save results (in iterations) (default: 1)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='PyTorch device to use (default: cpu)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print verbose output (default: False)')
    
    # Equilibrium finding parameters
    parser.add_argument('--num_random_starts', type=int, default=20,
                        help='Number of random starting points for equilibrium finding (default: 20)')
    parser.add_argument('--regret_threshold', type=float, default=1e-3,
                        help='Regret threshold for equilibrium finding (default: 1e-3)')
    parser.add_argument('--dist_threshold', type=float, default=1e-2,
                        help='Distance threshold for distinguishing equilibria (default: 1e-2)')
    parser.add_argument('--solver_iters', type=int, default=10000,
                        help='Number of iterations for solver (default: 10000)')
    
    # Visualization parameters
    parser.add_argument('--create_visualizations', action='store_true',
                        help='Create visualizations (default: False)')
    parser.add_argument('--plot_regret_landscapes', action='store_true',
                        help='Plot regret landscapes (default: False)')
    parser.add_argument('--create_report', action='store_true',
                        help='Create visualization report (default: False)')
    
    # Parameter sweep parameters
    parser.add_argument('--run_parameter_sweep', action='store_true',
                        help='Run parameter sweep (default: False)')
    parser.add_argument('--sweep_holding_periods', type=str, default='1,5,10',
                        help='Comma-separated list of holding periods for sweep (default: 1,5,10)')
    parser.add_argument('--sweep_lambdas', type=str, default='0.05,0.1,0.2',
                        help='Comma-separated list of lambda values for sweep (default: 0.05,0.1,0.2)')
    parser.add_argument('--sweep_max_iterations', type=int, default=3,
                        help='Maximum iterations for parameter sweep experiments (default: 3)')
    parser.add_argument('--sweep_profiles_per_iteration', type=int, default=3,
                        help='Profiles per iteration for parameter sweep (default: 3)')
    
    # Holding period sweep parameters
    parser.add_argument('--run_holding_sweep', action='store_true',
                        help='Run holding period sweep (default: False)')
    
    return parser.parse_args()


def main():
    """Main function to run the EGTA experiment."""
    args = parse_arguments()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save run parameters
    with open(os.path.join(args.output_dir, 'parameters.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    print(f"Starting EGTA run with output to: {args.output_dir}")
    
    # Run main EGTA experiment
    egta, game, analysis = run_egta_experiment(args)
    
    # Run parameter sweep if requested
    run_parameter_sweep(args)
    
    # Run holding period sweep if requested
    run_holding_period_sweep(args)
    
    print(f"\nAll experiments completed. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()

