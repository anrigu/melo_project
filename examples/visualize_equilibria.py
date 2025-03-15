#!/usr/bin/env python
"""
Example script demonstrating how to visualize equilibria and strategy traces
in the EGTA framework.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import json
from marketsim.egta.core.game import Game
from marketsim.egta.solvers.equilibria import replicator_dynamics, fictitious_play, gain_descent, quiesce
from marketsim.egta.visualization import (
    plot_equilibrium_payoffs, 
    plot_strategy_traces, 
    run_solver_with_trace,
    compare_solvers,
    visualize_quiesce_results
)

def create_rock_paper_scissors_game(device="cpu"):
    """Create a simple Rock-Paper-Scissors game for demonstration."""
    payoff_data = []
    strategy_names = ["Rock", "Paper", "Scissors"]
    
    # All Rock
    payoff_data.append([
        (0, "Rock", 0),
        (1, "Rock", 0)
    ])
    
    # All Paper
    payoff_data.append([
        (0, "Paper", 0),
        (1, "Paper", 0)
    ])
    
    # All Scissors
    payoff_data.append([
        (0, "Scissors", 0),
        (1, "Scissors", 0)
    ])
    
    # Rock vs Paper
    payoff_data.append([
        (0, "Rock", -1),
        (1, "Paper", 1)
    ])
    
    # Rock vs Scissors
    payoff_data.append([
        (0, "Rock", 1),
        (1, "Scissors", -1)
    ])
    
    # Paper vs Scissors
    payoff_data.append([
        (0, "Paper", -1),
        (1, "Scissors", 1)
    ])
    
    return Game.from_payoff_data(payoff_data, strategy_names, device=device)

def create_prisoners_dilemma_game(device="cpu"):
    """Create a simple Prisoner's Dilemma game for demonstration."""
    payoff_data = []
    strategy_names = ["Cooperate", "Defect"]
    
    # Both cooperate
    payoff_data.append([
        (0, "Cooperate", -1),
        (1, "Cooperate", -1)
    ])
    
    # Both defect
    payoff_data.append([
        (0, "Defect", -3),
        (1, "Defect", -3)
    ])
    
    # Cooperate vs Defect
    payoff_data.append([
        (0, "Cooperate", -5),
        (1, "Defect", 0)
    ])
    
    return Game.from_payoff_data(payoff_data, strategy_names, device=device)

def create_melo_cda_game(device="cpu"):
    """Create a simplified MELO vs CDA strategy game."""
    payoff_data = []
    strategy_names = ["MELO_0.0", "MELO_0.5", "MELO_1.0", "CDA"]
    
    # All MELO_0.0
    payoff_data.append([
        (0, "MELO_0.0", 3.2),
        (1, "MELO_0.0", 3.2),
        (2, "MELO_0.0", 3.2)
    ])
    
    # All MELO_0.5
    payoff_data.append([
        (0, "MELO_0.5", 3.5),
        (1, "MELO_0.5", 3.5),
        (2, "MELO_0.5", 3.5)
    ])
    
    # All MELO_1.0
    payoff_data.append([
        (0, "MELO_1.0", 2.8),
        (1, "MELO_1.0", 2.8),
        (2, "MELO_1.0", 2.8)
    ])
    
    # All CDA
    payoff_data.append([
        (0, "CDA", 2.5),
        (1, "CDA", 2.5),
        (2, "CDA", 2.5)
    ])
    
    # MELO_0.0 and MELO_0.5
    payoff_data.append([
        (0, "MELO_0.0", 3.3),
        (1, "MELO_0.5", 3.6),
        (2, "MELO_0.5", 3.6)
    ])
    
    # MELO_0.0 and MELO_1.0
    payoff_data.append([
        (0, "MELO_0.0", 3.1),
        (1, "MELO_1.0", 3.0),
        (2, "MELO_1.0", 3.0)
    ])
    
    # MELO_0.0 and CDA
    payoff_data.append([
        (0, "MELO_0.0", 3.4),
        (1, "CDA", 2.3),
        (2, "CDA", 2.3)
    ])
    
    # MELO_0.5 and MELO_1.0
    payoff_data.append([
        (0, "MELO_0.5", 3.4),
        (1, "MELO_1.0", 2.9),
        (2, "MELO_1.0", 2.9)
    ])
    
    # MELO_0.5 and CDA
    payoff_data.append([
        (0, "MELO_0.5", 3.7),
        (1, "CDA", 2.2),
        (2, "CDA", 2.2)
    ])
    
    # MELO_1.0 and CDA
    payoff_data.append([
        (0, "MELO_1.0", 3.0),
        (1, "CDA", 2.1),
        (2, "CDA", 2.1)
    ])
    
    # Mixed profiles with 3 strategies
    # MELO_0.0, MELO_0.5, MELO_1.0
    payoff_data.append([
        (0, "MELO_0.0", 3.2),
        (1, "MELO_0.5", 3.5),
        (2, "MELO_1.0", 2.9)
    ])
    
    # MELO_0.0, MELO_0.5, CDA
    payoff_data.append([
        (0, "MELO_0.0", 3.3),
        (1, "MELO_0.5", 3.6),
        (2, "CDA", 2.2)
    ])
    
    # MELO_0.0, MELO_1.0, CDA
    payoff_data.append([
        (0, "MELO_0.0", 3.2),
        (1, "MELO_1.0", 2.9),
        (2, "CDA", 2.1)
    ])
    
    # MELO_0.5, MELO_1.0, CDA
    payoff_data.append([
        (0, "MELO_0.5", 3.5),
        (1, "MELO_1.0", 2.8),
        (2, "CDA", 2.1)
    ])
    
    # All strategies
    payoff_data.append([
        (0, "MELO_0.0", 3.2),
        (1, "MELO_0.5", 3.5),
        (2, "MELO_1.0", 2.8)
    ])
    
    metadata = {
        "name": "MELO vs CDA Example Game",
        "description": "A simplified game demonstrating MELO vs CDA strategy selection",
        "num_agents": 3
    }
    
    return Game.from_payoff_data(payoff_data, strategy_names, device=device, metadata=metadata)

def visualize_equilibrium_solver(game, solver_name, iters=1000):
    """Run a single equilibrium solver and visualize the results."""
    print(f"\nRunning {solver_name} solver...")
    
    # Initialize with uniform mixture
    num_strategies = game.num_strategies
    device = game.game.device
    uniform_mix = torch.ones(num_strategies, device=device) / num_strategies
    
    # Run solver with trace
    final_mix, trace, regret = run_solver_with_trace(game, solver_name, uniform_mix, iters=iters)
    
    # Print results
    print(f"Final mixture after {len(trace)-1} iterations:")
    for s, prob in enumerate(final_mix):
        if prob > 0.001:
            print(f"  {game.strategy_names[s]}: {prob.item():.4f}")
    print(f"Regret: {regret:.6f}")
    
    # Plot strategy traces
    plot_strategy_traces(game, trace, f"{solver_name} - Strategy Traces")
    
    # Calculate payoffs at equilibrium
    payoffs = game.deviation_payoffs(final_mix).cpu().numpy()
    print("\nExpected payoffs at equilibrium:")
    for s, payoff in enumerate(payoffs):
        print(f"  {game.strategy_names[s]}: {payoff:.4f}")
    
    # Create mock equilibria list for plot_equilibrium_payoffs
    equilibria = [(final_mix, regret)]
    plot_equilibrium_payoffs(game, equilibria)
    
    return final_mix, trace, regret

def visualize_quiesce(game, num_iters=5, num_random_starts=10, solver='replicator'):
    """Run QUIESCE algorithm and visualize the results."""
    print(f"\nRunning QUIESCE algorithm with {solver} solver...")
    
    # Run QUIESCE
    equilibria = quiesce(
        game, 
        num_iters=num_iters, 
        num_random_starts=num_random_starts, 
        solver=solver, 
        verbose=True
    )
    
    # Visualize results
    visualize_quiesce_results(game, equilibria)
    
    return equilibria

def main():
    parser = argparse.ArgumentParser(description='Visualize equilibria in EGTA framework')
    parser.add_argument('--game', type=str, default='melo_cda', 
                        choices=['rps', 'pd', 'melo_cda'],
                        help='Game to analyze')
    parser.add_argument('--solver', type=str, default='replicator',
                        choices=['replicator', 'fictitious_play', 'gain_descent', 'all', 'quiesce'],
                        help='Equilibrium solver to use')
    parser.add_argument('--iters', type=int, default=100,
                        help='Number of solver iterations')
    parser.add_argument('--device', type=str, default='cpu',
                        help='PyTorch device')
    parser.add_argument('--output_dir', type=str, default='visualization_results',
                        help='Directory to save results')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create game
    if args.game == 'rps':
        game = create_rock_paper_scissors_game(args.device)
        print("Analyzing Rock-Paper-Scissors game")
    elif args.game == 'pd':
        game = create_prisoners_dilemma_game(args.device)
        print("Analyzing Prisoner's Dilemma game")
    else:  # melo_cda
        game = create_melo_cda_game(args.device)
        print("Analyzing MELO vs CDA game")
    
    # Run solver(s)
    if args.solver == 'all':
        # Compare all solvers
        results = compare_solvers(game, iters=args.iters)
        
        # Save results to JSON
        results_dict = {}
        for solver_name, result in results.items():
            if 'final_mixture' in result:
                results_dict[solver_name] = {
                    'final_mixture': result['final_mixture'].tolist(),
                    'regret': result['regret'],
                    'iterations': result['iterations']
                }
        
        with open(os.path.join(args.output_dir, 'comparison_results.json'), 'w') as f:
            json.dump(results_dict, f, indent=2)
    
    elif args.solver == 'quiesce':
        # Run QUIESCE
        equilibria = visualize_quiesce(game, num_iters=5, solver='replicator')
        
        # Save results to JSON
        equilibria_dict = []
        for i, (mix, regret) in enumerate(equilibria):
            equilibria_dict.append({
                'id': i,
                'mixture': mix.tolist(),
                'regret': regret,
                'strategy_names': game.strategy_names
            })
        
        with open(os.path.join(args.output_dir, 'quiesce_results.json'), 'w') as f:
            json.dump(equilibria_dict, f, indent=2)
    
    else:
        # Run single solver
        final_mix, trace, regret = visualize_equilibrium_solver(game, args.solver, args.iters)
        
        # Save results to JSON
        result_dict = {
            'final_mixture': final_mix.tolist(),
            'regret': float(regret),
            'iterations': len(trace) - 1,
            'strategy_names': game.strategy_names
        }
        
        with open(os.path.join(args.output_dir, f'{args.solver}_results.json'), 'w') as f:
            json.dump(result_dict, f, indent=2)

if __name__ == '__main__':
    main() 