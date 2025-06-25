#!/usr/bin/env python3
"""Run Alpha-Rank on a previously saved RoleSymmetricGame directory.

The directory must contain a ``raw_payoff_data.json`` produced by the EGTA
pipeline (see examples/run_mobi_zi_role_symmetric_analysis.py).  Optionally it
may also contain ``experiment_parameters.json`` or ``game_details.json`` but we
only need the raw pay-off rows to rebuild the game.

Usage
-----
python analysis/alpharank_on_saved_rsg.py PATH [PATH ...] \
       --alpha 50 --network

* ``PATH`` can be either a single ``comprehensive_rsg_results_*`` directory or
  a higher-level directory; in the latter case the script searches recursively
  for sub-directories that contain the JSON file and processes each one.

* ``--network`` draws the Markov-chain network (same semantics as in
  ``alpharank_by_hp.py``).
"""
from __future__ import annotations

import argparse, json, os, sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

try:
    from open_spiel.python.egt import alpharank, alpharank_visualizer, utils
except ImportError:
    sys.stderr.write("open_spiel is required; pip install open_spiel\n"); raise

# Re-use helpers from the other script to avoid duplication
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))  # so python can find sibling module
from alpharank_by_hp import rsg_to_alpharank_tables, print_alpharank_results  # type: ignore

# Local import of RoleSymmetricGame (after updating path for project root)
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
from marketsim.game.role_symmetric_game import RoleSymmetricGame

RAW_FILE = "raw_payoff_data.json"

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def load_raw_payoff_data(path: Path):
    with open(path) as f:
        return [[tuple(x) for x in prof] for prof in json.load(f)]  # type: ignore[arg-type]


def build_rsg(raw: List[List[Tuple[str, str, str, float]]]) -> RoleSymmetricGame:
    """Infer meta data from first profile and rebuild an RSG."""
    first = raw[0]
    role_names = sorted({row[1] for row in first})  # e.g. MOBI, ZI
    strategy_names_per_role: Dict[str, List[str]] = {r: [] for r in role_names}
    num_players_per_role = []

    for row in first:
        _, role, strat, _ = row
        if strat not in strategy_names_per_role[role]:
            strategy_names_per_role[role].append(strat)

    for role in role_names:
        strategy_names_per_role[role].sort()
        num_players_per_role.append(sum(1 for r in first if r[1] == role))

    return RoleSymmetricGame.from_payoff_data_rsg(
        payoff_data=raw,
        role_names=role_names,
        num_players_per_role=num_players_per_role,
        strategy_names_per_role=[strategy_names_per_role[r] for r in role_names],
        device="cpu",
        normalize_payoffs=False,
    )


# -----------------------------------------------------------------------------
# Main routine for one directory
# -----------------------------------------------------------------------------

def run_alpharank_on_dir(res_dir: Path, args):
    raw_path = next(res_dir.glob(f"**/{RAW_FILE}"), None)
    if raw_path is None:
        print(f"[SKIP] {res_dir}: no {RAW_FILE} found")
        return

    print("\n======= Alpha-Rank on", res_dir.relative_to(Path.cwd()), "=======")

    raw_data = load_raw_payoff_data(raw_path)
    rsg = build_rsg(raw_data)
    tables = rsg_to_alpharank_tables(rsg)

    if args.sweep:
        pi = alpharank.sweep_pi_vs_alpha(tables, visualize=True)
        rhos = rho_m = None
    else:
        rhos, rho_m, pi, *_ = alpharank.compute(tables, alpha=args.alpha)

    print_alpharank_results(rsg, pi)

    if np.count_nonzero(pi > 1e-6) == 1 and np.isclose(pi.max(), 1.0, atol=1e-4):
        print(">>> Detected PURE ESS (singleton MCC) – no profitable single-role deviations in this saved game.")

    if args.network and not args.sweep:
        if rhos is None or not np.isfinite(rhos).all() or not np.isfinite(rho_m):
            print("[WARN] Non-finite rhos – network plot skipped.")
        else:
            labels = utils.get_strat_profile_labels(tables, utils.check_payoffs_are_hpt(tables))
            try:
                alpharank_visualizer.NetworkPlot(
                    tables, rhos, rho_m, pi, labels, num_top_profiles=args.top_profiles
                ).compute_and_draw_network()
            except ValueError as e:
                print(f"[WARN] Network plot failed: {e}")


# -----------------------------------------------------------------------------
# CLI entry-point
# -----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Alpha-Rank on saved RoleSymmetricGame directories.")
    p.add_argument("paths", type=Path, nargs="+", help="File or directory paths to search for raw_payoff_data.json")
    p.add_argument("--alpha", type=float, default=50, help="Selection intensity for fixed-α run.")
    p.add_argument("--sweep", action="store_true", help="Do an alpha-sweep instead of fixed alpha.")
    p.add_argument("--network", action="store_true", help="Draw Markov-chain network (disabled with --sweep).")
    p.add_argument("--top-profiles", type=int, default=6, help="How many profiles to label in the network plot.")
    args = p.parse_args()

    for root in args.paths:
        if root.is_file():
            run_alpharank_on_dir(root.parent, args)
        else:
            # search depth-first for result dirs containing the raw file
            for sub in root.rglob("*"):
                if sub.is_dir():
                    if (sub / RAW_FILE).exists():
                        run_alpharank_on_dir(sub, args)


if __name__ == "__main__":
    main() 