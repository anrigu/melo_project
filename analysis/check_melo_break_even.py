#!/usr/bin/env python
"""Quick sanity-check plots for MELO break-even horizon.

Expected input: a CSV (or Parquet/Pickle) file produced during the bootstrap
pass that contains *replicate-level* payoffs and (optionally) inventory data.
Mandatory columns
    • replicate   – integer replicate id
    • hp          – holding period (int, in the same units used in the study)
    • strategy    – string strategy name (e.g. 'MELO', 'CDA')
    • payoff      – deviation payoff for that strategy in the replicate
Optional columns
    • inventory   – inventory level at the end of the holding period (or any
                    risk metric you'd like – variance is computed over replicates)

Usage
-----
$ python check_melo_break_even.py path/to/bootstrap_replicates.csv

The script produces two figures in the current directory:
    • delta_payoff_MELO.png   – first-difference of MELO mean payoff vs HP
    • inventory_var_MELO.png  – inventory variance vs HP (if inventory column)
"""
from __future__ import annotations

import sys, os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Ensure local project packages (e.g., `marketsim/`) are importable when the
# script is executed via `python analysis/check_melo_break_even.py`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------
# Parse CLI
# ---------------------------------------------------------------
if len(sys.argv) >= 2:
    in_path = Path(sys.argv[1]).expanduser()
else:
    in_path = None

# ------------------------------------------------------------------
# Option A: load precomputed bootstrap file if provided
# ------------------------------------------------------------------
if in_path is not None:
    if not in_path.exists():
        raise FileNotFoundError(in_path)

    if in_path.suffix == ".csv":
        df = pd.read_csv(in_path)
    elif in_path.suffix == ".parquet":
        df = pd.read_parquet(in_path)
    elif in_path.suffix in {".pkl", ".pickle"}:
        df = pd.read_pickle(in_path)
    else:
        raise ValueError("Unsupported file extension – use .csv, .parquet or .pkl")
else:
    # ------------------------------------------------------------------
    # Option B: no path given – rebuild payoff DataFrame directly from
    #           raw_payoff_data.json files, mimicking plot_payoff_vs_eq.py
    # ------------------------------------------------------------------
    print("No input file provided – scanning raw_payoff_data.json files …")
    import glob, json, torch
    from marketsim.egta.core.game import Game

    # These must match the constants in plot_payoff_vs_eq.py
    SEED_ROOTS = [
        "result_two_role_still_role_symmetric_3",
        "result_two_role_still_role_symmetric_4",
        "result_two_role_still_role_symmetric_5",
        "result_two_role_still_role_symmetric_6",
        "result_two_role_still_role_symmetric_7",
    ]

    # Holding periods we care about (0,20,…,320)
    HP_LIST = list(range(0, 321, 20))

    rows = []
    for hp in HP_LIST:
        # Collect all raw payoff files for this HP across seeds
        raw_files = []
        for root in SEED_ROOTS:
            raw_files.extend(glob.glob(f"{root}/**/holding_period_{hp}/**/raw_payoff_data.json", recursive=True))
            raw_files.extend(glob.glob(f"{root}/**/pilot_egta_run_{hp}_*/**/raw_payoff_data.json", recursive=True))

        # Each file contains list-of-profiles; treat each file as one replicate
        # Compute per-strategy deviation payoffs using marketsim.Game
        for rep_id, rf in enumerate(raw_files):
            try:
                profs = json.load(open(rf))
            except Exception:
                continue

            try:
                game = Game.from_payoff_data(profs, normalize_payoffs=False)
            except Exception:
                continue

            # Try analytic 2x2 NE first
            mix = None
            if all(len(lst) == 2 for lst in game.strategy_names_per_role):
                try:
                    eqs = game.find_nash_equilibrium_2x2()
                    if eqs:
                        mix = eqs[0][0]
                except Exception:
                    mix = None

            if mix is None:
                from marketsim.egta.solvers.equilibria import replicator_dynamics
                init = torch.ones(game.num_strategies) / game.num_strategies
                try:
                    mix = replicator_dynamics(game, init, iters=400, converge_threshold=1e-4, use_multiple_starts=False)
                except Exception:
                    continue

            dev = game.deviation_payoffs(mix)
            idx = 0
            for role, strats in zip(game.role_names, game.strategy_names_per_role):
                for strat in strats:
                    strat_name = strat
                    rows.append({
                        "replicate": rep_id,
                        "hp": hp,
                        "strategy": strat_name,
                        "payoff": float(dev[idx])
                    })
                    idx += 1

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No payoff data could be constructed. Provide an input file instead.")

required_cols = {"replicate", "hp", "strategy", "payoff"}
if not required_cols.issubset(df.columns):
    raise ValueError(f"Input file must contain columns: {', '.join(required_cols)}")

# ---------------------------------------------------------------
# Filter to MELO strategy (case-insensitive contains 'melo')
# ---------------------------------------------------------------
strat_mask = df["strategy"].str.contains("melo", case=False, na=False)
if not strat_mask.any():
    raise RuntimeError("Could not find any rows for a MELO strategy.")

df_melo = df[strat_mask].copy()

# Ensure holding periods are sorted multiples of 20
hp_sorted = sorted(df_melo["hp"].unique())

# ---------------------------------------------------------------
# 1. Mean payoff slope ∆ over HP
# ---------------------------------------------------------------
mean_by_hp = df_melo.groupby("hp")["payoff"].mean().reindex(hp_sorted)

delta = mean_by_hp.diff().iloc[1:]  # first difference
hp_mid = hp_sorted[1:]

plt.figure(figsize=(5, 3))
plt.plot(hp_mid, delta.values, marker="o")
plt.axhline(0, ls="--", c="grey")
plt.axvline(220, ls=":", c="red")
plt.title("Change in MELO payoff per 20-period step")
plt.ylabel("Δ Payoff")
plt.xlabel("Holding period")
plt.tight_layout()
plt.savefig("delta_payoff_MELO.png", dpi=150)
print("Saved delta_payoff_MELO.png")

# ---------------------------------------------------------------
# 2. Payoff variance across replicates as risk proxy
# ---------------------------------------------------------------

var_payoff = df_melo.groupby("hp")["payoff"].var(ddof=1).reindex(hp_sorted)

plt.figure(figsize=(5, 3))
plt.plot(hp_sorted, var_payoff.values, marker="s")
plt.axvline(220, ls=":", c="red")
plt.title("MELO payoff variance vs holding period")
plt.ylabel("Var(payoff)")
plt.xlabel("Holding period")
plt.tight_layout()
plt.savefig("payoff_var_MELO.png", dpi=150)
print("Saved payoff_var_MELO.png") 