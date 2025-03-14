# Implementation Notes: MELO vs CDA Allocation Experiment

## Overview of Changes

Updated the EGTA framework to see howstrategic agents allocate their trading between two market mechanisms (CDA and MELO). The key changes are:

### 1. Updated MeloSimulator Class

- **Strategy Definition**: Changed from agent types to market allocation proportions
  - Strategies now represent different allocations between CDA and MELO markets
  - Format: `MELO_X_Y` where X is percentage in CDA and Y is percentage in MELO
  - Five strategies implemented: 100-0, 75-25, 50-50, 25-75, 0-100

- **Agent Creation**: Modified to create MELO agents with specific allocation parameters
  - Each agent now has `cda_proportion` and `melo_proportion` parameters
  - Fixed order quantity (5) as Anri mentioned

- **Payoff Calculation**: Updated to combine payoffs from both markets based on allocation
  - Total payoff = (CDA payoff × cda_proportion) + (MELO payoff × melo_proportion)

### 2. Updated Example Script (run_melo_egta.py)

- Command line argument descriptions
- Better Logging

### 3. Added Documentation

- README for example experiment
- Added implementation notes for future reference
- Created a test script to verify the simulator works correctly

## Technical Details

### MeloAgent Modifications

The `MeloAgent` class now needs to handle:

1. Trading in both CDA and MELO markets with specific proportions
2. Placing orders with fixed quantity (5)
3. Setting prices as fundamental value + uniformly sampled private value

### Simulation Flow

1. For each profile, the simulator creates agents with different market allocation strategies
2. Agents place orders in both markets according to their allocation proportions
3. The simulation runs and collects payoffs from both markets
4. Payoffs are combined based on each agent's allocation proportions
5. Results are analyzed to find Nash equilibria

## Testing

To test the implementation:

1. Run the test script: `python examples/test_melo_simulator.py`
2. Run a small EGTA experiment: `python examples/run_melo_egta.py --max_iterations 2 --reps 2`

## Next Steps

1. Consider adding more fine-grained allocation strategies (e.g., 10% increments)
2. Add analysis of market liquidity and price efficiency under different allocation mixes

# QUIESCE Implementation Notes

This document outlines the improved implementation of the QUIESCE algorithm in the EGTA framework. The implementation follows the formal algorithm as outlined in "Solving Large Incomplete-Information Games Using Quiesce" (Brinkman et al.).

## Key Improvements

1. **Asynchronous Programming**:
   - The implementation uses asynchronous programming for efficiency
   - Helper functions like `test_unconfirmed_candidates` and `test_deviations` are defined as async functions
   - A synchronous wrapper `quiesce_sync` is provided for ease of use

2. **Explicit Tracking of Game State**:
   - `SubgameCandidate` class to track candidates, their status, and support sets
   - `MaximalSubgameCollection` class to manage subgames and check containment
   - `DeviationPriorityQueue` class to prioritize the exploration of deviations by gain

3. **Separation of Solver and Game View Operations (GVO)**:
   - Clear separation between equilibrium finding and candidate generation
   - Explicit handling of unconfirmed and confirmed candidates
   - Follows the flowchart in the formal algorithm

4. **Adaptive Exploration of Strategy Space**:
   - Priority queue for deviations based on gain
   - Exploration of mixed strategy candidates in addition to pure strategies
   - Consideration of the uniform mixture as a potential equilibrium

## Algorithm Flow

The implementation follows this general flow:

1. **Initialization**:
   - Create singleton strategy subgames
   - Initialize data structures for tracking game state

2. **Main Loop**:
   - Test unconfirmed candidates to see if they are equilibria
   - If no unconfirmed candidates, explore new subgames from the deviation queue
   - If no more candidates or deviations, we're done

3. **Testing Unconfirmed Candidates**:
   - Solve equilibria for all unconfirmed candidates
   - Calculate regrets
   - Test for beneficial deviations
   - If no beneficial deviations, add to confirmed equilibria

4. **Testing Deviations**:
   - Calculate gains from deviation
   - Add beneficial deviations to the priority queue
   - Also add mixed strategy candidates between pure strategies and beneficial deviations

5. **Exploring New Subgames**:
   - Select the highest-gain deviation from the queue
   - Check if it creates a new maximal subgame
   - Add a new candidate if it does

## Comparison with Traditional QUIESCE

This implementation more closely follows the formal QUIESCE algorithm compared to the original implementation in several ways:

1. **Adaptive vs. Fixed Exploration**:
   - Original: Used fixed mixing weights (0.25, 0.5, 0.75) to generate new candidates
   - Improved: Uses a priority queue based on gain to explore the most promising strategies first

2. **Subgame Management**:
   - Original: Did not explicitly track maximal subgames
   - Improved: Maintains a collection of maximal subgames and checks containment

3. **Asynchronous Programming**:
   - Original: Purely synchronous implementation
   - Improved: Uses asynchronous programming for better efficiency

4. **Mixed Strategy Support**:
   - Original: Limited exploration of mixed strategies
   - Improved: Better testing and exploration of mixed strategy candidates

## Usage

The implementation provides two main entry points:

1. `quiesce`: Asynchronous implementation of the QUIESCE algorithm
2. `quiesce_sync`: Synchronous wrapper for ease of use

Example usage:

```python
from marketsim.egta.solvers.equilibria import quiesce_sync

equilibria = quiesce_sync(
    game=game,
    num_iters=10,
    regret_threshold=1e-4,
    dist_threshold=1e-3,
    restricted_game_size=4,
    solver='replicator',
    solver_iters=1000,
    verbose=True
)
```

## Testing

The implementation has been tested with classic game theory examples like Rock-Paper-Scissors, Prisoner's Dilemma, and Stag Hunt. It successfully finds both pure and mixed strategy equilibria in these games.
