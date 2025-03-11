"""
Example script for running EGTA with the MELO simulator.

This script allows you to simulate strategic interactions between MELO agents 
deciding how to allocate their trading between the traditional CDA market
and the MELO mechanism.
"""
import argparse
import os
import time
import json
import torch

from marketsim.egta.egta import EGTA
from marketsim.egta.simulators.melo_simulator import MeloSimulator
from marketsim.egta.schedulers.dpr import DPRScheduler
from marketsim.egta.schedulers.random import RandomScheduler
from marketsim.egta.utils.visualization import create_visualization_report


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run EGTA with MELO simulator to analyze allocation decisions between CDA and MELO markets'
    )
    
    # EGTA parameters
    parser.add_argument('--max_iterations', type=int, default=10, 
                       help='Maximum number of iterations of the EGTA process')
    parser.add_argument('--profiles_per_iteration', type=int, default=5, 
                       help='Number of strategy profiles to simulate in each iteration')
    parser.add_argument('--max_profiles', type=int, default=100, 
                       help='Maximum total number of profiles to simulate')
    parser.add_argument('--output_dir', type=str, default='results/egta', 
                       help='Directory to save simulation results and visualizations')
    parser.add_argument('--scheduler', type=str, default='dpr', choices=['random', 'dpr'], 
                       help='Profile scheduler type: random (uniform sampling) or dpr (Deviation-Preserving Reduction)')
    parser.add_argument('--device', type=str, default='cpu', 
                       help='PyTorch device for computations (cpu or cuda)')
    parser.add_argument('--seed', type=int, default=42, 
                       help='Random seed for reproducibility')
    parser.add_argument('--visualize', action='store_true', 
                       help='Generate visualization report with plots of equilibria and strategy frequencies')
    
    # Simulator parameters
    parser.add_argument('--num_players', type=int, default=10, 
                       help='Number of strategic MELO agents')
    parser.add_argument('--sim_time', type=int, default=1000, 
                       help='Simulation time horizon')
    parser.add_argument('--lam', type=float, default=0.1, 
                       help='Arrival rate of market events')
    parser.add_argument('--mean', type=float, default=100, 
                       help='Mean of the fundamental value process')
    parser.add_argument('--r', type=float, default=0.05, 
                       help='Mean reversion rate of the fundamental value process')
    parser.add_argument('--q_max', type=int, default=10, 
                       help='Maximum inventory quantity for agents')
    parser.add_argument('--holding_period', type=int, default=1, 
                       help='Holding period for MELO mechanism')
    parser.add_argument('--reps', type=int, default=3, 
                       help='Number of simulation repetitions for each profile (higher = more precise payoff estimates)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save arguments
    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Create MELO simulator
    simulator = MeloSimulator(
        num_players=args.num_players,
        sim_time=args.sim_time,
        lam=args.lam,
        mean=args.mean,
        r=args.r,
        q_max=args.q_max,
        holding_period=args.holding_period,
        reps=args.reps
    )
    
    # Get strategies from simulator
    strategies = simulator.get_strategies()
    
    # Create scheduler
    if args.scheduler == 'random':
        scheduler = RandomScheduler(
            strategies=strategies,
            num_players=args.num_players,
            batch_size=args.profiles_per_iteration,
            seed=args.seed
        )
    else:  # 'dpr'
        scheduler = DPRScheduler(
            strategies=strategies,
            num_players=args.num_players,
            subgame_size=min(4, len(strategies)),
            batch_size=args.profiles_per_iteration,
            seed=args.seed
        )
    
    # Create EGTA framework
    egta = EGTA(
        simulator=simulator,
        scheduler=scheduler,
        device=args.device,
        output_dir=args.output_dir,
        max_profiles=args.max_profiles,
        seed=args.seed
    )
    
    # Run EGTA
    print(f"Starting EGTA with {args.scheduler} scheduler")
    print(f"Analyzing market allocation strategies: {strategies}")
    print("Strategies represent different allocations between CDA and MELO markets")
    
    start_time = time.time()
    game = egta.run(
        max_iterations=args.max_iterations,
        profiles_per_iteration=args.profiles_per_iteration,
        save_frequency=1,
        verbose=True
    )
    total_time = time.time() - start_time
    
    print(f"\nEGTA completed in {total_time:.2f} seconds")
    
    # Analyze equilibria
    print("\nAnalyzing equilibria...")
    analysis = egta.analyze_equilibria(verbose=True)
    
    # Save analysis
    with open(os.path.join(args.output_dir, 'analysis.json'), 'w') as f:
        json.dump(analysis, f, indent=2)
    
    # Create visualization report if requested
    if args.visualize:
        print("\nCreating visualization report...")
        vis_dir = os.path.join(args.output_dir, 'visualizations')
        report_path = create_visualization_report(
            game=game,
            equilibria=egta.equilibria,
            output_dir=vis_dir
        )
        print(f"Visualization report saved to {report_path}")
        print(f"Open this HTML file in a browser to view the visualization report")
    
    print(f"\nResults saved to {args.output_dir}")


if __name__ == '__main__':
    main() 