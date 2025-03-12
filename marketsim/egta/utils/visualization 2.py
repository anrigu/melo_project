"""
Visualization utilities for EGTA.
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional, Union
import torch

from marketsim.egta.core.game import Game


def plot_regret_landscape(game: Game, strategies: List[str], resolution: int = 100, 
                        output_file: Optional[str] = None, show: bool = True):
    """
    Plot the regret landscape for a 2D projection of the strategy space.
    
    Args:
        game: Game to analyze
        strategies: List of two strategy names to plot
        resolution: Grid resolution
        output_file: File to save the plot (optional)
        show: Whether to show the plot
    """
    if len(strategies) != 2:
        raise ValueError("Must specify exactly 2 strategies for visualization")
    
    # Get strategy indices
    strategy_indices = [game.strategy_names.index(s) for s in strategies]
    
    # Create a grid of mixtures
    x = np.linspace(0, 1, resolution)
    y = np.linspace(0, 1, resolution)
    X, Y = np.meshgrid(x, y)
    
    # Calculate regret at each grid point
    regrets = np.zeros((resolution, resolution))
    for i in range(resolution):
        for j in range(resolution):
            # Create mixture over the two strategies
            mix = np.zeros(game.num_strategies)
            mix[strategy_indices[0]] = X[i, j]
            mix[strategy_indices[1]] = Y[i, j]
            
            # Fill in the rest uniformly
            remaining = 1.0 - mix[strategy_indices[0]] - mix[strategy_indices[1]]
            if remaining > 0:
                other_indices = [i for i in range(game.num_strategies) if i not in strategy_indices]
                for idx in other_indices:
                    mix[idx] = remaining / len(other_indices)
            
            # Skip invalid mixtures
            if np.sum(mix) < 0.99 or np.sum(mix) > 1.01 or np.min(mix) < 0:
                regrets[i, j] = np.nan
                continue
            
            # Calculate regret
            mix_tensor = torch.tensor(mix, dtype=torch.float32, device=game.game.device)
            regrets[i, j] = game.regret(mix_tensor).item()
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    contour = plt.contourf(X, Y, regrets, levels=20, cmap='viridis')
    plt.colorbar(contour, label='Regret')
    plt.xlabel(strategies[0])
    plt.ylabel(strategies[1])
    plt.title('Regret Landscape')
    
    # Mark low-regret regions
    low_regret_mask = regrets < 0.01
    if np.any(low_regret_mask):
        plt.contour(X, Y, low_regret_mask, levels=[0.5], colors='red', linewidths=2)
    
    # Save the plot
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    # Show the plot
    if show:
        plt.show()
    else:
        plt.close()


def plot_equilibria(game: Game, equilibria: List[Tuple[torch.Tensor, float]], 
                   output_file: Optional[str] = None, show: bool = True):
    """
    Plot the equilibria of a game as a heatmap.
    
    Args:
        game: Game to analyze
        equilibria: List of (mixture, regret) tuples
        output_file: File to save the plot (optional)
        show: Whether to show the plot
    """
    # Extract data
    strategy_names = game.strategy_names
    num_strategies = len(strategy_names)
    num_eq = len(equilibria)
    
    # Create data matrix
    data = np.zeros((num_strategies, num_eq))
    regrets = []
    
    for i, (eq_mix, eq_regret) in enumerate(equilibria):
        data[:, i] = eq_mix.cpu().numpy()
        regrets.append(eq_regret)
    
    # Create the plot
    plt.figure(figsize=(max(8, num_eq * 0.5 + 2), max(6, num_strategies * 0.5 + 2)))
    
    # Add heatmap
    im = plt.imshow(data, cmap='YlOrRd')
    plt.colorbar(im, label='Probability')
    
    # Add labels
    plt.xlabel('Equilibrium')
    plt.ylabel('Strategy')
    plt.title('Equilibrium Mixtures')
    
    # Add strategy names
    plt.yticks(range(num_strategies), strategy_names)
    
    # Add equilibrium numbers and regrets
    eq_labels = [f"Eq {i+1}\n({regrets[i]:.4f})" for i in range(num_eq)]
    plt.xticks(range(num_eq), eq_labels)
    
    # Add values as text
    for i in range(num_strategies):
        for j in range(num_eq):
            if data[i, j] > 0.01:  # Only show significant values
                plt.text(j, i, f"{data[i, j]:.2f}", 
                        ha="center", va="center", 
                        color="black" if data[i, j] < 0.5 else "white")
    
    # Save the plot
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    # Show the plot
    if show:
        plt.show()
    else:
        plt.close()


def plot_payoff_matrix(game: Game, strategy_subset: Optional[List[str]] = None,
                      output_file: Optional[str] = None, show: bool = True):
    """
    Plot the expected payoff matrix for pure strategy profiles.
    
    Args:
        game: Game to analyze
        strategy_subset: Optional subset of strategies to plot
        output_file: File to save the plot (optional)
        show: Whether to show the plot
    """
    # Get strategy names
    strategy_names = game.strategy_names if strategy_subset is None else strategy_subset
    
    # Filter to subset if needed
    if strategy_subset is not None:
        strategy_indices = [game.strategy_names.index(s) for s in strategy_subset]
    else:
        strategy_indices = list(range(len(game.strategy_names)))
    
    # Extract payoff data for pure strategy profiles
    payoff_matrix = np.zeros((len(strategy_indices), len(strategy_indices)))
    
    for i, row_idx in enumerate(strategy_indices):
        for j, col_idx in enumerate(strategy_indices):
            # Create a profile with all row strategy except one col strategy
            profile = np.zeros(game.num_strategies)
            profile[row_idx] = game.num_players - 1
            profile[col_idx] = 1
            
            # Convert to tensor
            profile_tensor = torch.tensor(profile, dtype=torch.float32, device=game.game.device)
            
            # Get payoffs
            try:
                payoff = game.game.pure_payoffs(profile_tensor)[col_idx].item()
                payoff_matrix[i, j] = payoff
            except:
                payoff_matrix[i, j] = np.nan
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    im = plt.imshow(payoff_matrix, cmap='coolwarm')
    plt.colorbar(im, label='Payoff')
    
    # Add labels
    plt.xlabel('Deviating Strategy')
    plt.ylabel('Opponent Strategy')
    plt.title('Payoff Matrix')
    
    # Add strategy names
    plt.xticks(range(len(strategy_indices)), [strategy_names[i] for i in strategy_indices], rotation=45)
    plt.yticks(range(len(strategy_indices)), [strategy_names[i] for i in strategy_indices])
    
    # Add values as text
    for i in range(len(strategy_indices)):
        for j in range(len(strategy_indices)):
            if not np.isnan(payoff_matrix[i, j]):
                plt.text(j, i, f"{payoff_matrix[i, j]:.1f}", 
                        ha="center", va="center", 
                        color="black" if payoff_matrix[i, j] < np.nanmax(payoff_matrix)/2 else "white")
    
    # Save the plot
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    # Show the plot
    if show:
        plt.show()
    else:
        plt.close()


def plot_strategy_frequency(equilibria: List[Tuple[torch.Tensor, float]], 
                           strategy_names: List[str],
                           output_file: Optional[str] = None, 
                           show: bool = True):
    """
    Plot the frequency of strategies across equilibria.
    
    Args:
        equilibria: List of (mixture, regret) tuples
        strategy_names: List of strategy names
        output_file: File to save the plot (optional)
        show: Whether to show the plot
    """
    # Calculate strategy frequencies
    strategy_freqs = np.zeros(len(strategy_names))
    
    for eq_mix, _ in equilibria:
        strategy_freqs += eq_mix.cpu().numpy() / len(equilibria)
    
    # Sort strategies by frequency
    sorted_indices = np.argsort(-strategy_freqs)
    sorted_names = [strategy_names[i] for i in sorted_indices]
    sorted_freqs = strategy_freqs[sorted_indices]
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(sorted_names, sorted_freqs)
    
    # Color bars by frequency
    for i, bar in enumerate(bars):
        bar.set_color(plt.cm.viridis(sorted_freqs[i]))
    
    # Add labels
    plt.xlabel('Strategy')
    plt.ylabel('Average Frequency')
    plt.title('Strategy Frequency Across Equilibria')
    
    # Rotate x-axis labels for readability
    plt.xticks(rotation=45, ha='right')
    
    # Add values on top of bars
    for i, v in enumerate(sorted_freqs):
        if v > 0.01:  # Only show significant values
            plt.text(i, v + 0.01, f"{v:.2f}", ha='center')
    
    plt.tight_layout()
    
    # Save the plot
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    # Show the plot
    if show:
        plt.show()
    else:
        plt.close()


def create_visualization_report(game: Game, equilibria: List[Tuple[torch.Tensor, float]], 
                              output_dir: str):
    """
    Create a comprehensive visualization report.
    
    Args:
        game: Game to analyze
        equilibria: List of (mixture, regret) tuples
        output_dir: Directory to save the visualization
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot equilibria
    plot_equilibria(
        game=game,
        equilibria=equilibria,
        output_file=os.path.join(output_dir, 'equilibria.png'),
        show=False
    )
    
    # Plot strategy frequency
    plot_strategy_frequency(
        equilibria=equilibria,
        strategy_names=game.strategy_names,
        output_file=os.path.join(output_dir, 'strategy_frequency.png'),
        show=False
    )
    
    # Plot payoff matrix
    plot_payoff_matrix(
        game=game,
        output_file=os.path.join(output_dir, 'payoff_matrix.png'),
        show=False
    )
    
    # If there are only a few strategies, plot regret landscape for pairs
    if len(game.strategy_names) <= 5:
        for i in range(len(game.strategy_names)):
            for j in range(i+1, len(game.strategy_names)):
                plot_regret_landscape(
                    game=game,
                    strategies=[game.strategy_names[i], game.strategy_names[j]],
                    output_file=os.path.join(output_dir, f'regret_landscape_{i}_{j}.png'),
                    show=False
                )
    
    # Create a simple HTML report
    with open(os.path.join(output_dir, 'report.html'), 'w') as f:
        f.write(f"""
        <html>
        <head>
            <title>EGTA Visualization Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #2c3e50; }}
                h2 {{ color: #3498db; }}
                .figure {{ margin: 20px 0; }}
                .figure img {{ max-width: 100%; border: 1px solid #ddd; }}
            </style>
        </head>
        <body>
            <h1>EGTA Visualization Report</h1>
            
            <h2>Equilibria</h2>
            <div class="figure">
                <img src="equilibria.png" alt="Equilibria">
                <p>Heatmap showing the probability of each strategy in each equilibrium. Regret values are shown below each equilibrium.</p>
            </div>
            
            <h2>Strategy Frequency</h2>
            <div class="figure">
                <img src="strategy_frequency.png" alt="Strategy Frequency">
                <p>Average frequency of each strategy across all equilibria.</p>
            </div>
            
            <h2>Payoff Matrix</h2>
            <div class="figure">
                <img src="payoff_matrix.png" alt="Payoff Matrix">
                <p>Payoff matrix for pure strategy profiles. Each cell shows the payoff when deviating to the column strategy against a field of the row strategy.</p>
            </div>
        """)
        
        # Add regret landscapes if available
        if len(game.strategy_names) <= 5:
            f.write("<h2>Regret Landscapes</h2>\n")
            for i in range(len(game.strategy_names)):
                for j in range(i+1, len(game.strategy_names)):
                    f.write(f"""
                    <div class="figure">
                        <img src="regret_landscape_{i}_{j}.png" alt="Regret Landscape">
                        <p>Regret landscape for {game.strategy_names[i]} vs {game.strategy_names[j]}. Red contours indicate low-regret regions (potential equilibria).</p>
                    </div>
                    """)
        
        f.write("""
        </body>
        </html>
        """)
    
    print(f"Visualization report created in {output_dir}")
    return os.path.join(output_dir, 'report.html') 