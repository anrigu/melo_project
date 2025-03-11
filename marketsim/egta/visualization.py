"""
Visualization utilities for EGTA framework.
This module provides functions for visualizing game equilibria and solver traces.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
import seaborn as sns
from marketsim.egta.core.game import Game


def plot_equilibrium_payoffs(game: Game, equilibria: List[Tuple[torch.Tensor, float]]):
    """
    Plot the expected payoffs for each strategy at equilibrium.
    
    Args:
        game: The game to analyze
        equilibria: List of (mixture, regret) tuples from equilibrium solvers
    """
    if not equilibria:
        print("No equilibria to display")
        return
    
    n_equilibria = len(equilibria)
    strategy_names = game.strategy_names
    n_strategies = len(strategy_names)
    
    # Calculate expected payoffs for each equilibrium
    payoffs_at_eq = []
    for eq_mix, eq_regret in equilibria:
        # Get deviation payoffs (expected payoff of each strategy against the mixture)
        dev_payoffs = game.deviation_payoffs(eq_mix).cpu().numpy()
        payoffs_at_eq.append(dev_payoffs)
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create x-positions for the bars
    x = np.arange(n_strategies)
    width = 0.8 / n_equilibria  # Width of the bars
    
    # Plot bars for each equilibrium
    for i, (eq_mix, eq_regret) in enumerate(equilibria):
        eq_mix_np = eq_mix.cpu().numpy()
        payoffs = payoffs_at_eq[i]
        
        # Offset bars for each equilibrium
        offset = (i - n_equilibria / 2 + 0.5) * width
        
        # Create bars with height based on payoffs
        bars = ax.bar(x + offset, payoffs, width, alpha=0.7,
                      label=f'Eq {i+1} (regret: {eq_regret:.4f})')
        
        # Add strategy proportions as text on each bar
        for j, bar in enumerate(bars):
            if eq_mix_np[j] > 0.01:  # Only show text for strategies with non-zero probability
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                       f'{eq_mix_np[j]:.2f}', ha='center', va='bottom', 
                       rotation=90, fontsize=8)
    
    # Add labels and legend
    ax.set_xlabel('Strategy')
    ax.set_ylabel('Expected Payoff')
    ax.set_title('Expected Payoffs at Equilibria')
    ax.set_xticks(x)
    ax.set_xticklabels(strategy_names, rotation=45, ha='right')
    ax.legend()
    
    plt.tight_layout()
    plt.show()
    return fig


def plot_strategy_traces(game: Game, traces: List[torch.Tensor], title: str = 'Strategy Traces'):
    """
    Plot how strategy probabilities change during equilibrium solving.
    
    Args:
        game: The game being analyzed
        traces: List of mixture tensors representing the solver's path
        title: Title for the plot
    """
    if not traces:
        print("No traces to display")
        return
    
    # Convert traces to numpy
    trace_np = torch.stack(traces).cpu().numpy()
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot each strategy's probability over iterations
    for s in range(game.num_strategies):
        ax.plot(trace_np[:, s], label=game.strategy_names[s], linewidth=2)
    
    # Add labels and legend
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Strategy Probability')
    ax.set_title(title)
    ax.set_ylim(-0.05, 1.05)  # Give a little padding
    ax.legend()
    
    plt.tight_layout()
    plt.show()
    return fig


def run_solver_with_trace(game: Game, 
                         solver_name: str, 
                         mixture: torch.Tensor, 
                         **solver_kwargs) -> Tuple[torch.Tensor, List[torch.Tensor], float]:
    """
    Run a solver and return the result along with the trace of mixtures.
    
    Args:
        game: Game to analyze
        solver_name: Name of the solver ('replicator', 'fictitious_play', or 'gain_descent')
        mixture: Initial mixture
        **solver_kwargs: Additional arguments for the solver
        
    Returns:
        Tuple of (final_mixture, trace, regret)
    """
    from marketsim.egta.solvers.equilibria import replicator_dynamics, fictitious_play, gain_descent, regret
    
    # Make sure trace is returned
    solver_kwargs['return_trace'] = True
    
    # Run the appropriate solver
    if solver_name == 'replicator':
        result, trace = replicator_dynamics(game, mixture, **solver_kwargs)
    elif solver_name == 'fictitious_play':
        result, trace = fictitious_play(game, mixture, **solver_kwargs)
    elif solver_name == 'gain_descent':
        result, trace = gain_descent(game, mixture, **solver_kwargs)
    else:
        raise ValueError(f"Unknown solver: {solver_name}")
    
    # Calculate regret
    final_regret = regret(game, result).item()
    
    return result, trace, final_regret


def compare_solvers(game: Game, 
                   solvers: List[str] = ['replicator', 'fictitious_play', 'gain_descent'],
                   iters: int = 1000) -> Dict:
    """
    Compare different equilibrium solvers on the same game.
    
    Args:
        game: Game to analyze
        solvers: List of solver names to compare
        iters: Number of iterations for each solver
        
    Returns:
        Dictionary with solver results
    """
    device = game.game.device
    num_strategies = game.num_strategies
    
    # Start from uniform mixture
    uniform_mix = torch.ones(num_strategies, device=device) / num_strategies
    
    results = {}
    for solver_name in solvers:
        print(f"Running {solver_name}...")
        final_mix, trace, final_regret = run_solver_with_trace(
            game, solver_name, uniform_mix, iters=iters
        )
        
        results[solver_name] = {
            'final_mixture': final_mix,
            'trace': trace,
            'regret': final_regret,
            'iterations': len(trace) - 1  # Subtract 1 because first point is initial mixture
        }
        
        print(f"  Result: regret={final_regret:.6f}, iterations={len(trace)-1}")
        
        # Plot the trace
        fig = plot_strategy_traces(game, trace, f'{solver_name} - Strategy Traces')
        results[solver_name]['trace_figure'] = fig
    
    # Plot comparison of convergence
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for solver_name in solvers:
        regrets = []
        for t in range(len(results[solver_name]['trace'])):
            mix = results[solver_name]['trace'][t]
            regrets.append(game.regret(mix).item())
        
        ax.semilogy(regrets, label=solver_name)
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Regret (log scale)')
    ax.set_title('Solver Convergence Comparison')
    ax.legend()
    
    plt.tight_layout()
    plt.show()
    results['convergence_figure'] = fig
    
    return results


def visualize_quiesce_results(game: Game, equilibria: List[Tuple[torch.Tensor, float]]):
    """
    Visualize the results of QUIESCE algorithm.
    
    Args:
        game: Game being analyzed
        equilibria: List of (mixture, regret) tuples from QUIESCE
    """
    if not equilibria:
        print("No equilibria found")
        return
    
    # Display equilibria in a table
    strategy_names = game.strategy_names
    print(f"Found {len(equilibria)} equilibria:")
    
    for i, (eq_mix, eq_regret) in enumerate(equilibria):
        print(f"\nEquilibrium {i+1} (regret: {eq_regret:.6f}):")
        eq_np = eq_mix.cpu().numpy()
        
        # Print strategy probabilities
        for s, (name, prob) in enumerate(zip(strategy_names, eq_np)):
            if prob > 0.001:  # Only show strategies with non-zero probability
                print(f"  {name}: {prob:.4f}")
    
    # Plot the payoffs at equilibria
    plot_equilibrium_payoffs(game, equilibria)
    
    # If there are multiple equilibria, create a heatmap showing their similarities
    if len(equilibria) > 1:
        # Create similarity matrix
        n_eq = len(equilibria)
        similarity = np.zeros((n_eq, n_eq))
        
        for i in range(n_eq):
            for j in range(n_eq):
                mix_i = equilibria[i][0].cpu().numpy()
                mix_j = equilibria[j][0].cpu().numpy()
                # Use cosine similarity
                similarity[i, j] = np.dot(mix_i, mix_j) / (np.linalg.norm(mix_i) * np.linalg.norm(mix_j))
        
        # Plot heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(similarity, annot=True, fmt=".2f", cmap="YlGnBu",
                   xticklabels=[f"Eq {i+1}" for i in range(n_eq)],
                   yticklabels=[f"Eq {i+1}" for i in range(n_eq)])
        plt.title("Equilibria Similarity (Cosine)")
        plt.tight_layout()
        plt.show() 