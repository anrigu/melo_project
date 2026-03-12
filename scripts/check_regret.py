#!/usr/bin/env python3
"""check_regret.py

Usage
-----
    python scripts/check_regret.py --hp 140 --mix 1 0 0.097 0.903

The mixture must be supplied **in canonical order**
(MOBI_MELO, MOBI_CDA, ZI_MELO, ZI_CDA).

The helper reuses utilities from ``analysis.plot_bootstrap_compare`` to build
an empirical game for the requested holding period and then reports the regret
of the supplied mixture (after per-role normalisation).
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path
from typing import List

import torch

# ---------------------------------------------------------------------------
# Make the project root importable when the script is executed directly
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# pylint: disable=import-error, wrong-import-position
from analysis.plot_bootstrap_compare import (
    collect_profiles_by_seed,
    build_game_from_profiles,
    IDX_CAN_MAP,
)
# pylint: enable=import-error, wrong-import-position


def _renormalise_per_role(vec: torch.Tensor, role_sizes: List[int]) -> torch.Tensor:
    """Normalise *vec* **in place** so each role sums to 1."""
    start = 0
    for n in role_sizes:
        seg = slice(start, start + n)
        seg_sum = vec[seg].sum()
        if seg_sum > 0:
            vec[seg] /= seg_sum
        start += n
    return vec


def compute_regret(hp: int, mix_can: torch.Tensor) -> float:
    """Return regret of *mix_can* (canonical order) for holding-period *hp*."""
    # --- build empirical game ------------------------------------------------
    profiles = list(itertools.chain.from_iterable(collect_profiles_by_seed(hp).values()))
    game_full = build_game_from_profiles(profiles)
    if game_full is None:
        raise RuntimeError(f"Could not build RoleSymmetricGame for HP={hp}")

 
    keep_idx = [
        s for s in range(game_full.num_strategies)
        if not torch.isnan(game_full.game.rsg_payoff_table[s]).all()
    ]

    # Ensure at least one strategy per role survives; otherwise fall back to
    # the full game.
    role_ok = True
    for r_start, n in zip(game_full.role_starts, [2, 2]):  # 2×2 layout
        if not any(i in keep_idx for i in range(int(r_start), int(r_start + n))):
            role_ok = False
            break

    game = game_full.restrict(keep_idx) if role_ok else game_full

    # --- map canonical mixture into (possibly restricted) game order ---------
    mix_vec: List[float] = []
    for role, strats in zip(game.role_names, game.strategy_names_per_role):
        for strat in strats:
            can_idx = IDX_CAN_MAP.get((role, strat))
            if can_idx is None:
                # Strategy unknown in canonical mapping → zero mass
                mix_vec.append(0.0)
            else:
                mix_vec.append(float(mix_can[can_idx]))

    mix = torch.tensor(mix_vec, dtype=torch.float32)

    # --- per-role normalisation & regret -------------------------------------
    role_sizes = [len(s) for s in game.strategy_names_per_role]
    _renormalise_per_role(mix, role_sizes)

    # Fill any remaining NaNs in payoff table with zeros to avoid explosions
    payoffs = game.game.rsg_payoff_table
    if torch.isnan(payoffs).any():
        payoffs[torch.isnan(payoffs)] = 0.0

    return float(game.regret(mix))


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute regret of a canonical-order mixture for a given HP.")
    ap.add_argument("--hp", type=int, required=True, help="Holding period value (e.g. 140)")
    ap.add_argument(
        "--mix",
        type=float,
        nargs=4,
        metavar=("MOBI_MELO", "MOBI_CDA", "ZI_MELO", "ZI_CDA"),
        required=True,
        help="Mixture probabilities in canonical order",
    )
    ap.add_argument("--debug", action="store_true", help="Print detailed game/mix diagnostics")
    ap.add_argument("--dpr", action="store_true", help="Use DPR reduced game for regret (matches plot_bootstrap_compare block)")
    args = ap.parse_args()

    hp_val: int = args.hp
    mix_can = torch.tensor(args.mix, dtype=torch.float32)

    if args.debug:
        profiles = list(itertools.chain.from_iterable(collect_profiles_by_seed(hp_val).values()))
        game_full = build_game_from_profiles(profiles)
        if game_full is None:
            raise RuntimeError("Failed to build game.")
        print("\nStrategies present in game:")
        for role, strats in zip(game_full.role_names, game_full.strategy_names_per_role):
            print(f"  {role}: {', '.join(strats)}")

        # Map mixture onto *full* game order (without restriction) and compute regret
        mix_vec_full = []
        for role, strats in zip(game_full.role_names, game_full.strategy_names_per_role):
            for strat in strats:
                mix_vec_full.append(float(mix_can[IDX_CAN_MAP.get((role, strat), 0.0)]))
        mix_full = torch.tensor(mix_vec_full, dtype=torch.float32)
        _renormalise_per_role(mix_full, [len(s) for s in game_full.strategy_names_per_role])
        reg_full = game_full.regret(mix_full)

        print(f"Regret in *full* game (no restriction) = {reg_full:.6f}")

    if args.dpr:
        try:
            from gameanalysis import paygame, regret as ga_regret  # type: ignore
            from gameanalysis.reduction import deviation_preserving  # type: ignore
        except ImportError as exc:
            raise RuntimeError("gameanalysis package is required for --dpr mode; install it first") from exc

        # ---------------- Build full RSG and convert to paygame ----------------
        profiles_all = list(itertools.chain.from_iterable(collect_profiles_by_seed(hp_val).values()))
        game_full_ms = build_game_from_profiles(profiles_all)
        if game_full_ms is None:
            raise RuntimeError("Could not construct RoleSymmetricGame for DPR mode")

        role_names = game_full_ms.role_names
        num_players_full = [int(x) for x in game_full_ms.num_players_per_role]
        strat_names_per_role = game_full_ms.strategy_names_per_role

        cfg_arr = game_full_ms.game.rsg_config_table.cpu().numpy().astype(int)
        pay_arr = game_full_ms.game.rsg_payoff_table.cpu().numpy().T  # shape (num_profiles, num_strats)

        # Zero out payoffs for zero-count strategies in each profile (required by paygame)
        pay_arr[cfg_arr == 0] = 0.0

        ga_full = paygame.game_names(
            role_names,
            num_players_full,
            strat_names_per_role,
            cfg_arr,
            pay_arr,
        )

        red_players = [4] * len(role_names)
        ga_red = deviation_preserving.reduce_game(ga_full, red_players)

        # Map canonical mix to ga_red order
        mix_vec = []
        for role, strats in zip(ga_red.role_names, ga_red.strat_names):  # type: ignore[attr-defined]
            for strat in strats:
                mix_vec.append(float(mix_can[IDX_CAN_MAP[(role, strat)]]))

        # renormalise per role
        idx = 0
        for sz in ga_red.num_role_strats:  # type: ignore[attr-defined]
            seg = slice(idx, idx + sz)
            seg_sum = sum(mix_vec[seg]) if isinstance(seg, slice) else mix_vec[seg]
            if seg_sum > 0:
                for k in range(seg.start, seg.stop):
                    mix_vec[k] /= seg_sum
            idx += sz

        reg_dpr = ga_regret.mixture_regret(ga_red, mix_vec)
        print(f"HP {hp_val} | regret (DPR) = {reg_dpr:.9f} for mix {args.mix}")
    else:
        # Primary output (restricted RoleSymmetricGame)
        regret_val = compute_regret(hp_val, mix_can)
        print(f"HP {hp_val} | regret (restricted) = {regret_val:.9f} for mix {args.mix}")


if __name__ == "__main__":
    main() 