#!/usr/bin/env python3
"""
solve_equilibria_from_data.py
================================
Utility script to reconstruct a `Game` from a previously saved
`observations.json` file (created by the EGTA framework) and run
`quiesce_sync` to find Nash equilibria **without re-simulating** any
profiles.

Example
-------
$ python scripts/solve_equilibria_from_data.py \
        path/to/egta/run \
        --dist-threshold 1e-2 \
        --regret-threshold 1e-3 \
        --solver replicator \
        --solver-iters 5000

The script prints a human-readable list of equilibria and also writes a
machine-readable `equilibria.json` file next to `observations.json`.
"""
import argparse
import json
import os, sys
from typing import List, Tuple, Any

import torch
# --------------------------------------------------------------------- #
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from marketsim.egta.core.game import Game
from marketsim.egta.solvers.equilibria import quiesce_sync

###############################################################################
# Helper – load observations.json and convert into legacy payoff_data format
###############################################################################

def load_payoff_data(obs_path: str) -> List[List[Tuple[Any, ...]]]:
    """Return *payoff_data* in the format expected by Game.from_payoff_data.

    Each entry is a list[(player_id, role, strategy, payoff)].  We
    reconstruct artificial player ids (p0, p1, …) because they are not
    needed for equilibrium computation.
    """
    with open(obs_path, "r") as fh:
        raw = json.load(fh)

    payoff_data: List[List[Tuple[str, str, str, float]]] = []
    for record in raw:
        profile = record["profile"]  # list[[role, strat], …]
        payoffs = record["payoffs"]  # list[float] – same length
        # Safety: some old dumps store only the *average* payoff once.
        if len(payoffs) == 1 and len(profile) > 1:
            payoffs = payoffs * len(profile)

        row: List[Tuple[str, str, str, float]] = []
        for idx, ((role, strat), pay) in enumerate(zip(profile, payoffs)):
            row.append((f"p{idx}", role, strat, float(pay)))
        payoff_data.append(row)

    return payoff_data

# -------------------------------------------------------------------------
# Optional helper: sanity-check role ↔ strategy naming and filter out rows
# where a strategy appears under a role that does not match its prefix.
# E.g. strategy names starting with "ZI_" should not appear in role "MOBI".
# -------------------------------------------------------------------------

def filter_mislabeled_rows(payoff_data: List[List[Tuple[str,str,str,float]]]) -> List[List[Tuple[str,str,str,float]]]:
    """Drop complete profiles that contain a mismatching (role,strategy) pair.

    Heuristic: if strategy name starts with "MOBI" it should be in role
    "MOBI"; if it starts with "ZI" in role "ZI".  Extend this rule easily
    if you have more roles.
    """
    def pair_ok(role: str, strat: str) -> bool:
        low = strat.lower()
        if low.startswith("mobi"):
            return role.upper().startswith("MOBI")
        if low.startswith("zi"):
            return role.upper().startswith("ZI")
        # default: keep
        return True

    filtered = []
    dropped = 0
    for prof in payoff_data:
        if all(pair_ok(role, strat) for _pid, role, strat, _ in prof):
            filtered.append(prof)
        else:
            dropped += 1

    if dropped:
        print(f"[filter] Dropped {dropped} mis-labelled profiles (kept {len(filtered)})")
    return filtered

###############################################################################
# CLI
###############################################################################

def parse_args():
    p = argparse.ArgumentParser(__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument(
        "results_dir",
        nargs="?",
        default=None,
        help="(Option 1) Directory that contains observations.json or raw_payoff_data.json",
    )
    p.add_argument(
        "--data-file",
        type=str,
        default=None,
        help="(Option 2) Explicit path to observations.json or raw_payoff_data.json",
    )
    p.add_argument("--device", default="cpu", help="Torch device to use")
    p.add_argument("--dist-threshold", type=float, default=1e-2, help="L1 distance threshold for merging equilibria")
    p.add_argument("--regret-threshold", type=float, default=1e-3, help="Regret threshold for equilibrium acceptance")
    p.add_argument("--num-iters", type=int, default=50, help="Max QUIESCE iterations")
    p.add_argument("--num-random-starts", type=int, default=20, help="Additional random initial mixtures")
    p.add_argument("--restricted-game-size", type=int, default=4, help="Max support size for restricted subgames")
    p.add_argument("--solver", choices=["replicator", "fictitious_play", "gain_descent"], default="replicator", help="Inner solver for restricted games")
    p.add_argument("--solver-iters", type=int, default=5000, help="Iterations for the inner solver")
    p.add_argument("--verbose", action="store_true", help="Print progress information")
    p.add_argument("--filter-mislabeled", action="store_true", help="Drop profiles where strategy prefix doesn't match role name")
    return p.parse_args()

###############################################################################
# Main
###############################################################################


def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # Resolve input file path
    # ------------------------------------------------------------------
    if args.data_file is not None:
        data_path = args.data_file
        if not os.path.isfile(data_path):
            raise FileNotFoundError(f"{data_path} does not exist")
    elif args.results_dir is not None:
        # Prefer observations.json, else raw_payoff_data.json
        cand1 = os.path.join(args.results_dir, "observations.json")
        cand2 = os.path.join(args.results_dir, "raw_payoff_data.json")
        if os.path.isfile(cand1):
            data_path = cand1
        elif os.path.isfile(cand2):
            data_path = cand2
        else:
            raise FileNotFoundError(
                f"Neither observations.json nor raw_payoff_data.json found in {args.results_dir}"
            )
    else:
        raise ValueError("Please supply either results_dir or --data-file")

    print(f"Loading data from {data_path} …")

    # ------------------------------------------------------------------
    # Detect file format and build payoff_data list accordingly
    # ------------------------------------------------------------------
    with open(data_path, "r") as fh:
        first_char = fh.read(1)
        fh.seek(0)
        data_json = json.load(fh)

    if not isinstance(data_json, list):
        raise ValueError("Expected JSON list at top level")

    # Heuristic: observations.json has dicts with keys profile/payoffs; raw_payoff_data is list[list[tuple]]
    if data_json and isinstance(data_json[0], dict) and "profile" in data_json[0]:
        payoff_data = load_payoff_data(data_path)
        fmt = "observations.json"
    else:
        payoff_data = data_json  # already in legacy format
        fmt = "raw_payoff_data.json"

    print(f"Detected format: {fmt}  –  {len(payoff_data):,} profiles")

    if args.filter_mislabeled:
        payoff_data = filter_mislabeled_rows(payoff_data)

    print("Building Game object in memory …")
    game = Game.from_payoff_data(payoff_data, device=args.device)
    print(game)

    print("Solving for equilibria via QUIESCE (no simulation) …")
    equilibria = quiesce_sync(
        game=game,
        full_game=game,  # test candidates against the same table
        num_iters=args.num_iters,
        num_random_starts=args.num_random_starts,
        regret_threshold=args.regret_threshold,
        dist_threshold=args.dist_threshold,
        restricted_game_size=args.restricted_game_size,
        solver=args.solver,
        solver_iters=args.solver_iters,
        verbose=args.verbose,
        obs_store=None,  # deterministic mode – skip statistical tests
    )

    if not equilibria:
        print("No equilibria found.")
        return

    print("\nEquilibria:")
    for i, (mix, reg) in enumerate(equilibria, 1):
        supp = [
            f"{name}:{mix[j].item():.3f}"
            for j, name in enumerate(game.strategy_names)
            if mix[j].item() > 0.01
        ]
        print(f"[{i}] regret={reg:.3e}   " + ", ".join(supp))

    # ------------------------------------------------------------------
    # Save machine-readable output for reuse
    # ------------------------------------------------------------------
    out_path = os.path.join(args.results_dir, "equilibria.json")
    out = [
        {
            "mixture": mix.tolist(),
            "regret": float(reg),
        }
        for mix, reg in equilibria
    ]
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nSaved {len(equilibria)} equilibria → {out_path}")


if __name__ == "__main__":
    main() 