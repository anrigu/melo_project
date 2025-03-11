# MELO vs CDA Market Allocation Experiment

This experiment uses Empirical Game-Theoretic Analysis (EGTA) to study how strategic agents allocate their trading between two market mechanisms:

1. **Continuous Double Auction (CDA)**: Traditional order book market mechanism
2. **MELO Mechanism**: Alternative market mechanism with holding period

## Overview

In this simulation, strategic agents need to decide what proportion of their trading to allocate to each market mechanism. The strategies represent different allocation proportions:

- **MELO_100_0**: 100% CDA, 0% MELO
- **MELO_75_25**: 75% CDA, 25% MELO
- **MELO_50_50**: 50% CDA, 50% MELO
- **MELO_25_75**: 25% CDA, 75% MELO
- **MELO_0_100**: 0% CDA, 100% MELO

The EGTA framework runs simulations with various profiles of these strategies and analyzes the resulting Nash equilibria, which represent stable allocation decisions.

## Running the Experiment

To run the experiment with default parameters:

```bash
python examples/run_melo_egta.py --visualize
```

### Important Parameters

- `--num_players`: Number of strategic agents (default: 10)
- `--max_iterations`: Maximum EGTA iterations (default: 10)
- `--reps`: Simulation repetitions per profile (default: 3)
- `--visualize`: Generate visualization report (optional)

For a full list of parameters:

```bash
python examples/run_melo_egta.py --help
```

## Interpreting Results

The EGTA framework analyzes the outcomes to find Nash equilibria - allocation strategies where no agent can benefit by unilaterally changing their allocation.

Results are saved to `results/egta/` (by default) and include:

1. `analysis.json`: Summary of equilibria and strategy frequencies
2. `visualizations/`: Directory with plots and visualizations (if `--visualize` is used)
   - `equilibria.png`: Heatmap showing the probability of each strategy in equilibria
   - `strategy_frequency.png`: Average frequency of each strategy across equilibria
   - `payoff_matrix.png`: Payoff matrix showing returns for different strategy profiles
   - `regret_landscape_*.png`: Regret landscapes showing low-regret regions
   - `report.html`: Interactive HTML report with all visualizations

### Key Metrics

- **Equilibrium Mixtures**: Which allocation strategies are played with what probability in equilibrium
- **Regret Values**: How far each mixture is from a perfect equilibrium (lower is better)
- **Strategy Frequencies**: How common each allocation strategy is across all equilibria

## Example Interpretation

If the equilibrium analysis shows high probability for `MELO_50_50`, it suggests that splitting trading equally between CDA and MELO is strategically optimal.

If `MELO_100_0` dominates, it suggests the traditional CDA mechanism is more advantageous in the current market conditions. 