#!/usr/bin/env python3
"""Compute Alpha-Rank rankings for each holding-period EGTA run.

Usage
-----
python analysis/alpharank_by_hp.py 2_strat_results/2_strat_q_max_10 --q-max 10

This will iterate over all ``pilot_egta_run_*`` sub-directories, build a
RoleSymmetricGame from each run's ``raw_payoff_data.json`` file, convert that
game into the pay-off tensors expected by OpenSpiel's Alpha-Rank implementation
and finally print the Alpha-Rank stationary distribution for every pure
strategy profile, along with marginal per-role rankings.

Requirements
------------
* open_spiel (``pip install open_spiel``)
* numpy, torch (already required by marketsim)
"""
from __future__ import annotations

import argparse
import json
import sys, os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch


# External deps
try:
    from open_spiel.python.egt import alpharank
    from open_spiel.python.egt import alpharank_visualizer
except ImportError as err:  # pragma: no cover
    sys.stderr.write(
        "\nERROR: open_spiel is required for this script.\n"
        "Install it with `pip install open_spiel`.\n\n"
    )
    raise

# Local imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from marketsim.game.role_symmetric_game import RoleSymmetricGame

# -------------------------------------------------------------
# Helpers to build RoleSymmetricGame
# -------------------------------------------------------------

def load_raw_payoff_data(payoff_json_path: Path) -> List[List[Tuple[str, str, str, float]]]:
    """Load raw payoff data file returned by our simulations.

    The JSON structure is a list over *profiles*, each of which is a list of
    rows of the form ``[player_id, role, strategy_name, payoff]``.
    """
    with open(payoff_json_path, "r") as f:
        profiles = json.load(f)
    # Convert inner lists to tuples for type compatibility
    return [[tuple(x) for x in prof] for prof in profiles]  # type: ignore[arg-type]


# -------------------------------------------------------------
# Conversion to AlphaRank input tensors
# -------------------------------------------------------------

def rsg_to_alpharank_tables(rsg: RoleSymmetricGame) -> List[np.ndarray]:
    """Convert *rsg* into a list of payoff tensors for Alpha-Rank.

    Returns
    -------
    tables : list[np.ndarray]
        ``tables[p]`` has shape ``(S_0, S_1, ..., S_{R-1})`` where
        ``S_r`` is the number of strategies of role ``r``.  Entry ``tables[p][i_0, ..., i_{R-1}]``
        is the expected payoff for population/role *p* when each role ``r``
        deterministically plays its strategy index ``i_r``.
    """
    role_sizes: List[int] = [len(lst) for lst in rsg.strategy_names_per_role]
    n_roles = rsg.num_roles

    # Pre-compute global indices for (role, local_strat_idx)
    global_idx = {
        (r, s): rsg.role_starts[r] + s for r in range(n_roles) for s in range(role_sizes[r])
    }

    # Create empty pay-off tensors
    tables = [np.full(role_sizes, np.nan, dtype=np.float64) for _ in range(n_roles)]

    # Iterate over all pure-profile combinations
    for pure_profile in np.ndindex(*role_sizes):  # tuple of local strategy indices per role
        # Build mixture vector – 1.0 on chosen strat for each role
        mix = torch.zeros(rsg.num_strategies, dtype=torch.float32, device=rsg.device)
        for r_idx, s_local in enumerate(pure_profile):
            mix[global_idx[(r_idx, s_local)]] = 1.0

        # Expected pay-offs for this pure profile
        payoffs_per_role = rsg.mixture_values(mix).cpu().numpy()  # shape (n_roles,)

        # Fill tensors
        for r_idx in range(n_roles):
            tables[r_idx][pure_profile] = payoffs_per_role[r_idx]

    # Alpha-Rank expects *no* NaNs – replace any missing values with 0.
    tables = [np.nan_to_num(t, nan=0.0).astype(np.float64) for t in tables]
    return tables


# -------------------------------------------------------------
# Reporting helpers
# -------------------------------------------------------------

def print_alpharank_results(rsg: RoleSymmetricGame, pi: np.ndarray) -> None:
    """Print flattened Alpha-Rank stationary distribution.

    ``pi`` is a flat vector whose length equals the product of strategy counts
    across roles.  We map each profile to a readable label.
    """
    role_sizes = [len(lst) for lst in rsg.strategy_names_per_role]
    assert pi.size == int(np.prod(role_sizes)), "Size of pi mismatch with strategy profile space."

    print("\nAlpha-Rank stationary distribution over pure profiles:")
    print("----------------------------------------------------")
    for idx, prob in enumerate(pi):
        if prob < 1e-6:
            continue  # skip negligible mass
        local_indices = np.unravel_index(idx, role_sizes)
        prof_label = " | ".join(
            f"{rsg.role_names[r]}={rsg.strategy_names_per_role[r][s]}" for r, s in enumerate(local_indices)
        )
        print(f"{prof_label:<60} : {prob:.4f}")

    # Marginal per-role probabilities
    for r, role_name in enumerate(rsg.role_names):
        marg = np.zeros(role_sizes[r])
        for idx, prob in enumerate(pi):
            if prob == 0.0:
                continue
            local_indices = np.unravel_index(idx, role_sizes)
            marg[local_indices[r]] += prob
        print(f"\nRole {role_name} marginal ranking:")
        ordering = np.argsort(marg)[::-1]
        for rank, s_local in enumerate(ordering, start=1):
            strat = rsg.strategy_names_per_role[r][s_local]
            print(f"  {rank:2d}. {strat:<40}  {marg[s_local]:.4f}")


# -------------------------------------------------------------
# Main driver
# -------------------------------------------------------------

def process_run(run_dir: Path, args: argparse.Namespace) -> None:
    """Compute and print AlphaRank for one EGTA run directory."""
    payoff_files = list(run_dir.glob("**/raw_payoff_data.json"))
    if not payoff_files:
        print(f"[WARN] No raw_payoff_data.json found under {run_dir}")
        return
    payoff_path = payoff_files[0]
    print(f"\n=== Processing {run_dir.name} (payoff file: {payoff_path.relative_to(run_dir)}) ===")

    raw_payoff_data = load_raw_payoff_data(payoff_path)

    # Infer metadata – assume it matches across profiles
    role_names = ["MOBI", "ZI"]
    num_players_per_role = []
    strategy_names_per_role: Dict[str, List[str]] = {"MOBI": [], "ZI": []}

    # Determine counts & strategy sets from first profile
    first_profile = raw_payoff_data[0]
    for row in first_profile:
        _, role, strat, _ = row  # type: ignore[misc]
        if strat not in strategy_names_per_role[role]:
            strategy_names_per_role[role].append(strat)
    # Ensure consistent ordering
    for role in role_names:
        strategy_names_per_role[role].sort()
        # Count players per role in first profile
        num_players_per_role.append(sum(1 for r in first_profile if r[1] == role))

    rsg = RoleSymmetricGame.from_payoff_data_rsg(
        payoff_data=raw_payoff_data,
        role_names=role_names,
        num_players_per_role=num_players_per_role,
        strategy_names_per_role=[strategy_names_per_role[r] for r in role_names],
        device="cpu",
        normalize_payoffs=False,
    )

    tables = rsg_to_alpharank_tables(rsg)

    # Alpha-Rank computation (either fixed alpha or sweep with visualization)
    if args.sweep:
        # sweep_pi_vs_alpha internally calls alpharank_visualizer to draw the figure
        # It returns the stationary distribution π at the final alpha chosen.
        pi = alpharank.sweep_pi_vs_alpha(tables, visualize=True)
        rhos = rho_m = None  # network plot not available from sweep output
    else:
        rhos, rho_m, pi, *_rest = alpharank.compute(tables, alpha=args.alpha)

    print_alpharank_results(rsg, pi)

    # --- ESS detection ---------------------------------------------------
    if np.count_nonzero(pi > 1e-6) == 1 and np.isclose(pi.max(), 1.0, atol=1e-4):
        print(">>> Detected PURE ESS (singleton MCC) – no profitable single-role deviations exist for this market snapshot.")

    # Optional Markov-chain network visualization
    if args.network and not args.sweep:
        # Guard against numerical overflows that leave inf/NaN in rhos
        if rhos is None or not np.isfinite(rhos).all() or not np.isfinite(rho_m):
            print("[WARN] Skipping network plot due to non-finite transition matrix (overflow in AlphaRank).")
        else:
            from open_spiel.python.egt import utils  # local import to avoid unconditional dep

            payoffs_are_hpt = utils.check_payoffs_are_hpt(tables)
            strat_labels = utils.get_strat_profile_labels(tables, payoffs_are_hpt)

            try:
                plotter = alpharank_visualizer.NetworkPlot(
                    tables,
                    rhos,
                    rho_m,
                    pi,
                    strat_labels,
                    num_top_profiles=args.top_profiles,
                )
                plotter.compute_and_draw_network()
            except ValueError as e:
                print(f"[WARN] Network plot failed: {e}")
    elif args.network and args.sweep:
        print("[INFO] --network ignored when --sweep is used (need rhos matrix).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AlphaRank per holding period (pilot_egta_run_* dirs).")
    parser.add_argument("root_dir", type=Path, help="Directory containing pilot_egta_run_* subdirs")
    parser.add_argument("--alpha", type=float, default=1e2, help="Ranking-intensity parameter for AlphaRank.")
    parser.add_argument("--sweep", action="store_true", help="If set, run alpharank.sweep_pi_vs_alpha and show plot instead of fixed alpha computation.")
    parser.add_argument("--network", action="store_true", help="Draw Alpha-Rank Markov-chain network graph (disabled when --sweep).")
    parser.add_argument("--top-profiles", type=int, default=8, help="How many top profiles to display in the network plot legend.")
    args = parser.parse_args()

    if not args.root_dir.exists():
        parser.error(f"{args.root_dir} does not exist.")

    # -------------------------------------------------------------
    # Accept two layouts:
    # 1. *root_dir* contains many pilot_egta_run_* sub-dirs (original)
    # 2. *root_dir* itself IS a pilot_egta_run_* directory that contains
    #    holding_period_* children (newer dumps)
    # -------------------------------------------------------------
    run_dirs = sorted([p for p in args.root_dir.glob("pilot_egta_run_*") if p.is_dir()])

    if not run_dirs:
        # Fallback: treat the root itself as *the* run dir if it matches the naming convention
        if args.root_dir.name.startswith("pilot_egta_run_") and args.root_dir.is_dir():
            run_dirs = [args.root_dir]
        else:
            parser.error("No pilot_egta_run_* directories found under the given root, and the root is not itself a pilot run directory.")

    for run in run_dirs:
        try:
            process_run(run, args)
        except Exception as exc:  # pragma: no cover
            print(f"[ERROR] Failed to process {run}: {exc}")


if __name__ == "__main__":
    main() 