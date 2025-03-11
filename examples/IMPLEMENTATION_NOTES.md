# Implementation Notes: MELO vs CDA Allocation Experiment

## Overview of Changes

We've updated the EGTA framework to study how strategic agents allocate their trading between two market mechanisms (CDA and MELO). The key changes are:

### 1. Updated MeloSimulator Class

- **Strategy Definition**: Changed from agent types to market allocation proportions
  - Strategies now represent different allocations between CDA and MELO markets
  - Format: `MELO_X_Y` where X is percentage in CDA and Y is percentage in MELO
  - Five strategies implemented: 100-0, 75-25, 50-50, 25-75, 0-100

- **Agent Creation**: Modified to create MELO agents with specific allocation parameters
  - Each agent now has `cda_proportion` and `melo_proportion` parameters
  - Fixed order quantity (5) as specified in requirements

- **Payoff Calculation**: Updated to combine payoffs from both markets based on allocation
  - Total payoff = (CDA payoff × cda_proportion) + (MELO payoff × melo_proportion)

### 2. Updated Example Script (run_melo_egta.py)

- Improved command line argument descriptions
- Added explanatory comments about strategy meanings
- Enhanced output messages to better explain what the simulation is doing

### 3. Added Documentation

- Created a README specifically for this experiment
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
3. Implement sensitivity analysis for key parameters (holding period, arrival rates) 