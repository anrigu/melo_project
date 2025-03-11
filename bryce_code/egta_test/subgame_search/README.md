# EGTA with Subgame Search for MELO Simulator

This package implements Empirical Game-Theoretic Analysis (EGTA) with subgame search (also called quiesce) for analyzing multi-agent strategies using the MELO simulator.

## Overview

The implementation integrates with existing components in the MELO project:
- `melo_simulator.py` from the `simulator` folder (serves as the payoff oracle)
- `process_data.py` (prepares data and returns a `SymmetricGame`)
- `symmetric_game.py` (the main `SymmetricGame` class)
- `game.py` (abstract game class)
- `reductions/dpr.py` (deviation-preserving reduction of a `SymmetricGame`)
- `utils/eq_computation.py` (contains various equilibrium solvers)
- Support utilities for log-space operations, random sampling, and simplex transformations

## Algorithm

The subgame search algorithm (also known as quiesce) explores equilibria by:

1. Starting from small subgames (usually single-strategy subsets)
2. Maintaining a priority queue (PQ) of subgames to explore
3. For each subgame, ensuring it's "completely evaluated" by sampling profiles from the simulator
4. Solving the empirical subgame for Nash Equilibria
5. Checking if deviations outside the subgame are profitable
6. If profitable deviations exist, expanding the subgame
7. Continuing until finding an equilibrium robust to deviations, or all possible expansions are exhausted

## Components

The implementation consists of the following components:

### `subgame_search.py`

Contains the main `SubgameSearch` class that implements the algorithm. It:
- Manages a priority queue of subgames to explore
- Samples profiles from the simulator as needed
- Constructs and maintains the full empirical game
- Solves for equilibria in subgames
- Checks for profitable deviations
- Expands subgames when beneficial

### `melo_simulator_adapter.py`

Provides an adapter between the EGTA framework and the MELO simulator:
- Converts between EGTA strategy representations and simulator strategy representations
- Handles simulation of strategy profiles
- Extracts payoffs from simulation results
- Provides caching to avoid redundant simulations

### `run_egta.py`

A script that demonstrates how to use the implementation:
- Parses command line arguments
- Sets up the simulator adapter and subgame search
- Runs the algorithm
- Analyzes and prints results
- Optionally applies DPR (Deviation-Preserving Reduction)

### `egta_tutorial.py`

A tutorial script that explains the algorithm step-by-step:
- Shows how to set up the simulator adapter
- Demonstrates how to configure and run the subgame search
- Explains how to analyze the results
- Provides an example of using DPR for games with many players

## Usage

To run the EGTA with subgame search:

```bash
python run_egta.py --num-players 4 --strategies ZI MELO --sim-time 500 --max-iterations 20
```

For more options:

```bash
python run_egta.py --help
```

## Example

Here's a simple example of how to use the implementation:

```python
from subgame_search import SubgameSearch, MeloSimulatorAdapter

# Create the simulator adapter
simulator_adapter = MeloSimulatorAdapter(
    num_background_agents=15,
    sim_time=500,
    num_simulations=5
)

# Define strategies
strategy_names = ['ZI', 'MELO']
num_players = 4

# Create the subgame search instance
subgame_search = SubgameSearch(
    strategy_names=strategy_names,
    simulator=simulator_adapter.simulate_profile,
    num_players=num_players,
    regret_threshold=1e-3
)

# Run the subgame search
equilibria = subgame_search.run(max_iterations=20)

# Print results
for eq_mixture, regret in equilibria:
    print(f"Equilibrium with regret {regret:.6f}:")
    for i, (strat, prob) in enumerate(zip(strategy_names, eq_mixture)):
        print(f"  {strat}: {prob * 100:.2f}%")
```

## Requirements

- Python 3.6+
- PyTorch
- NumPy
- NetworkX (for visualization)
- Matplotlib (for visualization)

## References

This implementation is based on:
- The original quiesce algorithm: [EGTA Online](https://github.com/egtaonline/quiesce)
- Subgame search approach from Wellman et al. "Methods for Empirical Game-Theoretic Analysis" 