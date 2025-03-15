# Empirical Game-Theoretic Analysis (EGTA) Framework

PyTorch-based implementation of EGTA for market simulations, integrating with the MELO simulator.

## Note 
This needs to be rewritten and will change

## Overview

This framework provides tools for:

1. Running empirical game-theoretic analysis on market simulations
2. Finding Nash equilibria in symmetric games
3. Efficiently scheduling profile simulations
4. Analyzing equilibrium behavior

## Key Components

### Core

- `Game`: A wrapper around the SymmetricGame implementation with additional functionality for serialization and analysis

### Solvers

- `replicator_dynamics`: Find equilibria using replicator dynamics
- `fictitious_play`: Find equilibria using fictitious play
- `gain_descent`: Find equilibria using gradient descent on gains
- `quiesce`: Find all equilibria of a game using the QUIESCE algorithm

### Schedulers

- `RandomScheduler`: Samples profiles uniformly at random
- `DPRScheduler`: Implements Deviation-Preserving Reduction to efficiently explore large strategy spaces

### Simulators

- `MeloSimulator`: Interface to the MELO simulator

## Usage

Here's a basic example of using the framework:

```python
from marketsim.egta.egta import EGTA
from marketsim.egta.simulators.melo_simulator import MeloSimulator
from marketsim.egta.schedulers.dpr import DPRScheduler

# Create simulator
simulator = MeloSimulator(
    num_players=10,
    sim_time=1000,
    lam=0.1
)

# Create scheduler
scheduler = DPRScheduler(
    strategies=simulator.get_strategies(),
    num_players=10,
    subgame_size=4
)

# Create EGTA framework
egta = EGTA(
    simulator=simulator,
    scheduler=scheduler,
    output_dir='results/egta'
)

# Run EGTA
game = egta.run(
    max_iterations=10,
    profiles_per_iteration=5,
    verbose=True
)

# Analyze equilibria
analysis = egta.analyze_equilibria(verbose=True)
```

See `examples/run_melo_egta.py` for a complete example.

## Extending the Framework

### Custom Simulators

To create a custom simulator, inherit from `Simulator` and implement:

```python
class MySimulator(Simulator):
    def get_num_players(self) -> int:
        # Return number of players
        
    def get_strategies(self) -> List[str]:
        # Return list of strategy names
        
    def simulate_profile(self, profile: List[str]) -> List[Tuple[int, str, float]]:
        # Simulate a single profile and return payoffs
```

### Custom Schedulers

To create a custom scheduler, inherit from `Scheduler` and implement:

```python
class MyScheduler(Scheduler):
    def get_next_batch(self, game: Optional[Game] = None) -> List[List[str]]:
        # Return next batch of profiles to simulate
        
    def update(self, game: Game) -> None:
        # Update with new game data
```

## Dependencies

- PyTorch
- NumPy
- Matplotlib (for visualization)
- MELO simulator 