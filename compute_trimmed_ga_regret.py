#!/usr/bin/env python3
"""Compute GameAnalysis regret for a trimmed MELO corner game.

This script focuses on the canonical profile (28, 0; 30, 10) and its
one-player deviations, builds a small paygame containing just those
four profiles, and reports the pure-strategy regret using the
gameanalysis library.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


CANONICAL_ORDER: List[Tuple[str, str]] = [
    ("MOBI", "MOBI_0_100_shade0_0"),
    ("MOBI", "MOBI_100_0_shade250_500"),
    ("ZI", "ZI_0_100_shade250_500"),
    ("ZI", "ZI_100_0_shade250_500"),
]

ROLE_INDICES = {
    "MOBI": [0, 1],
    "ZI": [2, 3],
}

SEED_ROOTS: List[str] = [
    "result_two_role_still_role_symmetric_3",
    "result_two_role_still_role_symmetric_4",
    "result_two_role_still_role_symmetric_5",
    "result_two_role_still_role_symmetric_6",
    "result_two_role_still_role_symmetric_7",
    "result_one_role_still_role_symmetric_3",
    "result_one_role_still_role_symmetric_4",
    "result_one_role_still_role_symmetric_5",
    "result_one_role_still_role_symmetric_6",
    "result_one_role_still_role_symmetric_7",
    "gapfill_profiles",
]

MELO_PROFILE_DIR = "melo_profile_runs"


TARGET_COUNTS = (28, 0, 30, 10)
NEIGHBOUR_COUNTS = [
    (27, 1, 30, 10),
    (28, 0, 31, 9),
    (28, 0, 29, 11),
]
ALL_COUNTS = [TARGET_COUNTS, *NEIGHBOUR_COUNTS]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hp",
        type=int,
        nargs="+",
        required=True,
        help="Holding periods to evaluate (e.g. --hp 0 20 40)",
    )
    return parser.parse_args()


def _coerce_profile_block(raw_block: Sequence) -> List[Tuple[str, str, str, float]]:
    rows: List[Tuple[str, str, str, float]] = []
    for entry in raw_block:
        if isinstance(entry, dict):
            player = entry.get("player") or entry.get("Player") or entry.get("id") or entry.get("profile")
            role = entry.get("role") or entry.get("Role")
            strat = entry.get("strategy") or entry.get("Strategy")
            payoff = entry.get("payoff") or entry.get("Payoff") or entry.get("value")
        elif isinstance(entry, (list, tuple)) and len(entry) >= 4:
            player, role, strat, payoff = entry[:4]
        else:
            continue
        try:
            rows.append((str(player), str(role), str(strat), float(payoff)))
        except (TypeError, ValueError):
            continue
    return rows


def load_profile_blocks(path: Path) -> List[List[Tuple[str, str, str, float]]]:
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return []

    blocks: List[List[Tuple[str, str, str, float]]] = []

    def _append(candidate) -> None:
        if not candidate:
            return
        coerced = _coerce_profile_block(candidate)
        if coerced:
            blocks.append(coerced)

    if isinstance(payload, list):
        for entry in payload:
            if isinstance(entry, dict) and "legacy_rows" in entry:
                _append(entry["legacy_rows"])
            elif isinstance(entry, list):
                _append(entry)
    elif isinstance(payload, dict) and "legacy_rows" in payload:
        _append(payload["legacy_rows"])
    return blocks


def collect_seed_profiles(repo_root: Path, hp: int) -> List[List[Tuple[str, str, str, float]]]:
    profiles: List[List[Tuple[str, str, str, float]]] = []
    patterns = [
        f"**/holding_period_{hp}/**/raw_payoff_data.json",
        f"**/pilot_egta_run_{hp}_*/**/raw_payoff_data.json",
    ]
    for root in SEED_ROOTS:
        base = (repo_root / root)
        if not base.exists():
            continue
        for pattern in patterns:
            for hit in glob.glob(str(base / pattern), recursive=True):
                profiles.extend(load_profile_blocks(Path(hit)))
    return profiles


def collect_melo_profiles(repo_root: Path, hp: int) -> List[List[Tuple[str, str, str, float]]]:
    profiles: List[List[Tuple[str, str, str, float]]] = []
    base = (repo_root / MELO_PROFILE_DIR)
    if not base.exists():
        return profiles
    pattern = f"holding_period_{hp}_*/results.json"
    for hit in glob.glob(str(base / pattern)):
        profiles.extend(load_profile_blocks(Path(hit)))
    return profiles


def build_profile_lookup(
    profiles: List[List[Tuple[str, str, str, float]]]
) -> Dict[Tuple[int, ...], Dict[Tuple[str, str], List[float]]]:
    lookup: Dict[Tuple[int, ...], Dict[Tuple[str, str], List[float]]] = {}
    for block in profiles:
        counts = [0] * len(CANONICAL_ORDER)
        pay_map: Dict[Tuple[str, str], List[float]] = {pair: [] for pair in CANONICAL_ORDER}
        for _, role, strat, payoff in block:
            key = (role, strat)
            if key not in pay_map:
                continue
            counts[CANONICAL_ORDER.index(key)] += 1
            pay_map[key].append(payoff)
        counts_t = tuple(counts)
        slot = lookup.setdefault(counts_t, {pair: [] for pair in CANONICAL_ORDER})
        for pair, vals in pay_map.items():
            if vals:
                slot[pair].extend(vals)
    return lookup


def _mean(values: Sequence[float]) -> float:
    """Return the arithmetic mean of *values*."""
    if not values:
        raise ValueError("Cannot average empty payoff sample.")
    return float(sum(values) / len(values))


def _lookup_payoff(
    counts: Tuple[int, ...],
    strat_idx: int,
    lookup: Dict[Tuple[int, ...], Dict[Tuple[str, str], List[float]]],
) -> float:
    role, strat = CANONICAL_ORDER[strat_idx]
    samples = lookup.get(counts, {}).get((role, strat), [])
    if counts[strat_idx] > 0:
        if not samples:
            raise ValueError(f"Missing payoff data for {counts} – {role}:{strat}")
        return _mean(samples)
    return 0.0


def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parent
    sys.path.insert(0, str(repo_root))
    ga_bundle = repo_root / "marketsim" / "egta" / "gameanalysis-old"
    if ga_bundle.exists():
        sys.path.insert(0, str(ga_bundle))

    from gameanalysis import paygame, regret as ga_regret  # type: ignore
    from marketsim.egta.solvers.equilibria import replicator_dynamics  # type: ignore
    from marketsim.egta.core.game import Game  # type: ignore
    sys.path.insert(0, str(repo_root))

    for hp in args.hp:
        seed_profiles = collect_seed_profiles(repo_root, hp)
        melo_profiles = collect_melo_profiles(repo_root, hp)
        combined = seed_profiles + melo_profiles
        if not combined:
            print(f"HP {hp}: no payoff data found")
            continue

        lookup = build_profile_lookup(combined)
        missing = [counts for counts in ALL_COUNTS if counts not in lookup]
        if missing:
            print(f"HP {hp}: missing required profiles {missing}")
            continue

        profiles_rows: List[List[int]] = []
        payoffs_rows: List[List[float]] = []

        for counts in ALL_COUNTS:
            profiles_rows.append(list(counts))
            payoff_vec = [_lookup_payoff(counts, idx, lookup) for idx in range(len(CANONICAL_ORDER))]
            if any(math.isnan(x) for x in payoff_vec):
                raise ValueError(f"NaN payoff detected for profile {counts} at HP {hp}")
            payoffs_rows.append(payoff_vec)

        # Shift payoffs so that all entries are strictly positive (regret is translation invariant)
        min_payoff = min(min(row) for row in payoffs_rows)
        if min_payoff <= 0:
            shift = -min_payoff + 1e-9
            payoffs_rows = [[val + shift for val in row] for row in payoffs_rows]

        profiles_arr = np.asarray(profiles_rows, dtype=int)
        payoffs_arr = np.asarray(payoffs_rows, dtype=float)

        role_names = ["MOBI", "ZI"]
        strat_names = [
            [s for r, s in CANONICAL_ORDER if r == "MOBI"],
            [s for r, s in CANONICAL_ORDER if r == "ZI"],
        ]
        num_players = [
            TARGET_COUNTS[ROLE_INDICES["MOBI"][0]] + TARGET_COUNTS[ROLE_INDICES["MOBI"][1]],
            TARGET_COUNTS[ROLE_INDICES["ZI"][0]] + TARGET_COUNTS[ROLE_INDICES["ZI"][1]],
        ]

        ga_game = paygame.game_names(
            role_names,
            num_players,
            strat_names,
            profiles_arr,
            payoffs_arr,
        )

        target_counts_vec = np.asarray(TARGET_COUNTS, dtype=int)
        pure_regret = float(ga_regret.pure_strategy_regret(ga_game, target_counts_vec))

        ms_game = Game.from_payoff_data(combined, normalize_payoffs=False)
        # Start RD from the canonical corner mixture
        start_mix = ms_game.profile_to_mixture(np.asarray(TARGET_COUNTS, dtype=float))
        rd_mix = replicator_dynamics(
            ms_game,
            mixture=start_mix,
            iters=5000,
            converge_threshold=1e-4,
            use_multiple_starts=False,
        )
        rd_regret = float(ms_game.regret(rd_mix))
        rd_details: List[str] = []
        idx = 0
        for role, strats in zip(ms_game.role_names, ms_game.strategy_names_per_role):
            seg = rd_mix[idx : idx + len(strats)]
            rd_details.append(
                f"{role}: " + ", ".join(f"{strats[j]}={seg[j].item():.3f}" for j in range(len(strats)))
            )
            idx += len(strats)

        print(f"\n=== HP {hp} ===")
        print("Profiles used (counts):")
        for counts in ALL_COUNTS:
            print("  ", counts)
        print("\nPayoff table (profile rows × strategy columns):")
        for counts, row in zip(ALL_COUNTS, payoffs_rows):
            row_fmt = ", ".join(f"{val:10.4f}" for val in row)
            print(f"  {counts}: [{row_fmt}]")

        print(f"\nGameAnalysis pure-strategy regret for {TARGET_COUNTS}: {pure_regret:.6f}\n")
        print(f"Replicator dynamics regret (epsilon=1e-4): {rd_regret:.6f}")
        for line in rd_details:
            print(f"    {line}")


if __name__ == "__main__":
    main()
