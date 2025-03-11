# EGTA Framework for Market Mechanism Analysis

This repository contains an Empirical Game-Theoretic Analysis (EGTA) framework for analyzing market mechanisms, with a focus on comparing MELO (Minimum-cost Equilibrium-outcome LOcal-operations) and CDA (Continuous Double Auction) mechanisms.

## Overview

The EGTA framework provides tools for:

1. Simulating strategic interactions in market environments
2. Computing equilibria in symmetric games
3. Visualizing equilibria and strategy traces
4. Comparing different market mechanisms

## Installation

```bash
git clone <repository_url>
cd <repository_directory>
pip install -r requirements.txt
```

## Directory Structure

- `marketsim/`: Main package directory
  - `egta/`: EGTA framework implementation
    - `core/`: Core functionality for EGTA
    - `solvers/`: Equilibrium solvers
    - `visualization/`: Visualization utilities
  - `game/`: Game representation classes
  - `math/`: Mathematical utilities
  - `simulator/`: Market simulator implementations
- `examples/`: Example scripts
- `tests/`: Unit tests

## Usage

### Basic EGTA Framework Usage

```python
from marketsim.egta.core.game import Game
from marketsim.egta.solvers.equilibria import replicator_dynamics, quiesce

# Create a game from payoff data
game = Game.from_payoff_data(payoff_data, strategy_names)

# Find equilibria using replicator dynamics
eq_mix = replicator_dynamics(game, initial_mixture, iters=1000)

# Compute regret
regret = game.regret(eq_mix)

# Find all equilibria using QUIESCE
equilibria = quiesce(game, num_iters=5, solver='replicator')
```

### Visualizing Equilibria and Strategy Traces

The framework includes visualization utilities to help analyze equilibria and understand solver behavior:

```python
from marketsim.egta.visualization import (
    plot_equilibrium_payoffs, 
    plot_strategy_traces, 
    run_solver_with_trace,
    compare_solvers,
    visualize_quiesce_results
)

# Run a solver and get both the result and trace
final_mix, trace, regret = run_solver_with_trace(game, 'replicator', initial_mixture)

# Plot how strategy probabilities change during solving
plot_strategy_traces(game, trace, "Replicator Dynamics")

# Plot expected payoffs at equilibrium
plot_equilibrium_payoffs(game, [(final_mix, regret)])

# Compare multiple solvers
results = compare_solvers(game, solvers=['replicator', 'fictitious_play', 'gain_descent'])

# Visualize all equilibria found by QUIESCE
visualize_quiesce_results(game, equilibria)
```

### Example Script

We provide an example script `examples/visualize_equilibria.py` that demonstrates the visualization capabilities:

```bash
# Run with default settings (MELO vs CDA game, replicator dynamics)
python examples/visualize_equilibria.py

# Run QUIESCE on a Rock-Paper-Scissors game
python examples/visualize_equilibria.py --game rps --solver quiesce

# Compare all solvers on a Prisoner's Dilemma game
python examples/visualize_equilibria.py --game pd --solver all --iters 200
```

## Key Features

### Equilibrium Solvers

- **Replicator Dynamics**: Evolutionary algorithm for finding Nash equilibria
- **Fictitious Play**: Iterative best-response dynamics
- **Gain Descent**: Gradient-based optimization approach
- **QUIESCE**: QUality Improvement by Exploring Subgame Candidate Equilibria

### Visualization Capabilities

- **Strategy Traces**: View how mixture probabilities change during solver iterations
- **Equilibrium Payoffs**: Compare expected payoffs at equilibrium for different strategies
- **Solver Comparison**: Compare convergence behavior of different equilibrium solvers
- **Equilibrium Similarity**: Analyze similarities between multiple equilibria

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.