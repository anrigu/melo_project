#!/usr/bin/env python3
"""compute_profile_regret.py

Load every payoff profile referenced by plot_bootstrap_compare2.py together with
all MELO profile runs, build the empirical game for each holding period, and
report the regret of a target pure profile (default: counts (28, 0; 30, 10)).
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch


# Canonical role/strategy ordering used in the bootstrap script
CANONICAL_ORDER: List[Tuple[str, str]] = [
    ("MOBI", "MOBI_0_100_shade0_0"),
    ("MOBI", "MOBI_100_0_shade250_500"),
    ("ZI", "ZI_0_100_shade250_500"),
    ("ZI", "ZI_100_0_shade250_500"),
]

IDX_CAN_MAP = {pair: idx for idx, pair in enumerate(CANONICAL_ORDER)}
ROLE_INDICES = {
    "MOBI": [0, 1],
    "ZI": [2, 3],
}

# Direct copy of the seed roots scanned in plot_bootstrap_compare2.py
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
DEFAULT_COUNTS = (28, 0, 30, 10)
PROFILE_NEIGHBOURS = [
    (28, 0, 30, 10),
    (27, 1, 30, 10),
    (28, 0, 31, 9),
    (28, 0, 29, 11),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hp",
        type=int,
        nargs="*",
        help="Holding-period values to analyse (default: detect all available).",
    )
    parser.add_argument(
        "--counts",
        type=int,
        nargs=4,
        metavar=("M_MELO", "M_CDA", "Z_MELO", "Z_CDA"),
        default=DEFAULT_COUNTS,
        help="Pure-profile counts in canonical order (default: %(default)s).",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Repository root (default: directory containing this script).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print deviation payoffs alongside regret.",
    )
    parser.add_argument(
        "--seed-only",
        action="store_true",
        help="Use only pooled seed profiles (skip melo_profile_runs data).",
    )
    parser.add_argument(
        "--replicator-iters",
        type=int,
        default=5000,
        help="Iterations for replicator dynamics diagnostics (default: %(default)s).",
    )
    return parser.parse_args()


def _coerce_profile_block(raw_block: Sequence) -> List[Tuple[str, str, str, float]]:
    """Convert one raw profile block into four-tuples (player, role, strat, payoff)."""
    coerced: List[Tuple[str, str, str, float]] = []
    for row in raw_block:
        if isinstance(row, dict):
            player = row.get("player") or row.get("Player") or row.get("profile") or row.get("id")
            role = row.get("role") or row.get("Role")
            strat = row.get("strategy") or row.get("Strategy")
            payoff = row.get("payoff") or row.get("Payoff") or row.get("value")
        elif isinstance(row, (list, tuple)):
            if len(row) < 4:
                raise ValueError("Row must contain at least four entries")
            player, role, strat, payoff = row[:4]
        else:
            raise ValueError("Unsupported row format")

        player = str(player)
        role = str(role)
        strat = str(strat)
        payoff = abs(float(payoff))
        if math.isnan(payoff):
            raise ValueError("Encountered NaN payoff")
        coerced.append((player, role, strat, payoff))
    return coerced


def load_profile_blocks(path: Path) -> List[List[Tuple[str, str, str, float]]]:
    """Load every payoff block encoded in the given JSON file."""
    try:
        with path.open("r") as fh:
            payload = json.load(fh)
    except Exception as exc:
        print(f"[warn] Skipping {path}: {exc}")
        return []

    blocks: List[List[Tuple[str, str, str, float]]] = []

    def _maybe_add(candidate) -> None:
        if not candidate:
            return
        block = _coerce_profile_block(candidate)
        if block:
            blocks.append(block)

    if isinstance(payload, list):
        for entry in payload:
            if isinstance(entry, dict) and "legacy_rows" in entry:
                _maybe_add(entry["legacy_rows"])
            elif isinstance(entry, list):
                _maybe_add(entry)
    elif isinstance(payload, dict) and "legacy_rows" in payload:
        _maybe_add(payload["legacy_rows"])
    else:
        print(f"[warn] Unrecognised JSON structure in {path}")
    return blocks


def collect_seed_profiles(repo_root: Path, hp: int) -> List[List[Tuple[str, str, str, float]]]:
    """Gather all profiles used by the bootstrap script for a specific HP."""
    profiles: List[List[Tuple[str, str, str, float]]] = []
    patterns = [
        f"**/holding_period_{hp}/**/raw_payoff_data.json",
        f"**/pilot_egta_run_{hp}_*/**/raw_payoff_data.json",
    ]
    for seed_root in SEED_ROOTS:
        base = (repo_root / seed_root).resolve()
        if not base.exists():
            continue
        for pattern in patterns:
            for hit in glob.glob(str(base / pattern), recursive=True):
                profiles.extend(load_profile_blocks(Path(hit)))
    return profiles


def collect_melo_profiles(repo_root: Path, hp: int) -> List[List[Tuple[str, str, str, float]]]:
    """Gather every MELO profile run for the given HP."""
    profiles: List[List[Tuple[str, str, str, float]]] = []
    base = (repo_root / MELO_PROFILE_DIR).resolve()
    if not base.exists():
        return profiles
    pattern = f"holding_period_{hp}_*/results.json"
    for hit in glob.glob(str(base / pattern)):
        profiles.extend(load_profile_blocks(Path(hit)))
    return profiles


def discover_hps(repo_root: Path) -> List[int]:
    """Discover all holding-period values present in the data directories."""
    hp_set = set()
    pattern = re.compile(r"holding_period_(\d+)|pilot_egta_run_(\d+)_")

    def _scan_base(base: Path) -> None:
        if not base.exists():
            return
        for path in glob.glob(str(base / "**" / "*.json"), recursive=True):
            match = pattern.search(path)
            if match:
                hp = match.group(1) or match.group(2)
                hp_set.add(int(hp))

    for seed_root in SEED_ROOTS:
        _scan_base((repo_root / seed_root).resolve())
    _scan_base((repo_root / MELO_PROFILE_DIR).resolve())
    return sorted(hp_set)


def build_counts_map(game, counts: Sequence[int]) -> Dict[str, Dict[str, int]]:
    """Map the user-specified counts onto the game's role/strategy ordering."""
    if len(counts) != len(CANONICAL_ORDER):
        raise ValueError(f"Expected {len(CANONICAL_ORDER)} counts, got {len(counts)}")

    mapping: Dict[str, Dict[str, int]] = {
        role: {strat: 0 for strat in strats}
        for role, strats in zip(game.role_names, game.strategy_names_per_role)
    }

    for (role, strat), count in zip(CANONICAL_ORDER, counts):
        if role not in mapping or strat not in mapping[role]:
            if count != 0:
                raise ValueError(f"Strategy {role}:{strat} absent from empirical game")
            continue
        mapping[role][strat] = count
    return mapping


def counts_to_mixture(game, counts_map: Dict[str, Dict[str, int]]) -> torch.Tensor:
    """Convert integer counts to a per-strategy mixture vector aligned with the game."""
    mixture: List[float] = []
    for role_idx, (role, strats) in enumerate(zip(game.role_names, game.strategy_names_per_role)):
        role_counts = counts_map.get(role)
        if not role_counts:
            raise ValueError(f"No counts supplied for role {role}")
        total_players = sum(role_counts.get(strat, 0) for strat in strats)
        expected_players = int(game.num_players_per_role[role_idx])
        if total_players != expected_players:
            raise ValueError(
                f"Counts for role {role} sum to {total_players}, "
                f"but empirical game expects {expected_players}"
            )
        for strat in strats:
            mixture.append(role_counts.get(strat, 0) / total_players)
    device = game.game.device
    return torch.tensor(mixture, dtype=torch.float32, device=device)


def describe_deviations(game, mixture: torch.Tensor) -> Iterable[str]:
    """Yield per-role deviation payoff summaries for debugging."""
    dev = game.deviation_payoffs(mixture).cpu()
    idx = 0
    for role, strats in zip(game.role_names, game.strategy_names_per_role):
        seg = dev[idx: idx + len(strats)]
        mix_seg = mixture[idx: idx + len(strats)].cpu()
        value = float((mix_seg * seg).sum())
        parts = [f"{strat}={mix_seg[j]:.3f}->{seg[j].item():.4f}" for j, strat in enumerate(strats)]
        yield f"    {role}: V={value:.4f} | " + ", ".join(parts)
        idx += len(strats)
    return


def _normalise_per_role(vec: torch.Tensor, role_sizes: Sequence[int]) -> torch.Tensor:
    """Normalise mixture entries so each role's probabilities sum to 1."""
    out = vec.clone()
    start = 0
    for size in role_sizes:
        seg = out[start : start + size]
        total = seg.sum()
        if float(total) <= 0:
            seg.fill_(1.0 / size)
        else:
            seg.div_(total)
        out[start : start + size] = seg
        start += size
    return out


def build_profile_lookup(
    profiles: List[List[Sequence]],
) -> Dict[Tuple[int, ...], Dict[Tuple[str, str], List[float]]]:
    """Aggregate payoff samples keyed by canonical counts."""
    lookup: Dict[Tuple[int, ...], Dict[Tuple[str, str], List[float]]] = {}
    for block in profiles:
        counts = [0] * len(CANONICAL_ORDER)
        pay_map: Dict[Tuple[str, str], List[float]] = {pair: [] for pair in CANONICAL_ORDER}
        for entry in block:
            if isinstance(entry, dict):
                role = entry.get("role") or entry.get("Role")
                strat = entry.get("strategy") or entry.get("Strategy")
                payoff = entry.get("payoff") or entry.get("Payoff") or entry.get("value")
            elif isinstance(entry, (list, tuple)) and len(entry) >= 4:
                _, role, strat, payoff = entry[:4]
            else:
                continue
            key = (str(role), str(strat))
            idx = IDX_CAN_MAP.get(key)
            if idx is None:
                continue
            counts[idx] += 1
            try:
                pay_map[key].append(float(payoff))
            except (TypeError, ValueError):
                continue
        counts_t = tuple(counts)
        if counts_t not in lookup:
            lookup[counts_t] = {pair: [] for pair in CANONICAL_ORDER}
        for pair, vals in pay_map.items():
            if vals:
                lookup[counts_t][pair].extend(vals)
    return lookup


def compute_local_regret(
    target_counts: Tuple[int, ...],
    lookup: Dict[Tuple[int, ...], Dict[Tuple[str, str], List[float]]],
) -> Tuple[float, Dict[str, Dict[str, float]]]:
    """Compute regret via one-player deviations using neighbouring profiles."""
    if target_counts not in lookup:
        raise ValueError(f"No payoff samples found for counts {target_counts}")

    role_results: Dict[str, Dict[str, float]] = {}

    def _mean(vals: List[float]) -> float:
        return float(sum(vals) / len(vals))

    for role, idxs in ROLE_INDICES.items():
        total_players = sum(target_counts[i] for i in idxs)
        if total_players <= 0:
            raise ValueError(f"Role {role} has zero players in counts {target_counts}")

        strat_payoffs: Dict[str, float] = {}
        for local_pos, global_idx in enumerate(idxs):
            key = CANONICAL_ORDER[global_idx]
            samples = lookup[target_counts].get(key, [])
            payoff = _mean(samples) if samples else None

            if payoff is None:
                neighbour = list(target_counts)
                other_idx = idxs[1 - local_pos]
                if neighbour[other_idx] <= 0:
                    raise ValueError(f"Missing neighbour for deviation {role}:{key[1]}")
                neighbour[global_idx] += 1
                neighbour[other_idx] -= 1
                neighbour_t = tuple(neighbour)
                neighbour_samples = lookup.get(neighbour_t, {}).get(key, [])
                if not neighbour_samples:
                    raise ValueError(f"Missing neighbour profile for counts {target_counts} with deviation {role}:{key[1]}")
                payoff = _mean(neighbour_samples)

            strat_payoffs[key[1]] = payoff

        role_value = sum(
            (target_counts[idx] / total_players) * strat_payoffs[CANONICAL_ORDER[idx][1]]
            for idx in idxs
        )
        role_results[role] = {
            "value": role_value,
            "best": max(strat_payoffs.values()),
            "regret": max(strat_payoffs.values()) - role_value,
            "payoffs": strat_payoffs,
        }

    overall = max(info["regret"] for info in role_results.values())
    return overall, role_results


def _print_neighbour_profile_table(lookup: Dict[Tuple[int, ...], Dict[Tuple[str, str], List[float]]]):
    print("Profiles used (counts):")
    for counts in PROFILE_NEIGHBOURS:
        if counts in lookup:
            print("   ", counts)
        else:
            print("   ", counts, "(missing)")
    print("\nPayoff table (profile rows × strategy columns):")
    for counts in PROFILE_NEIGHBOURS:
        strat_map = lookup.get(counts)
        if not strat_map:
            print(f"  {counts}: missing")
            continue
        row = []
        for pair in CANONICAL_ORDER:
            samples = strat_map.get(pair, [])
            if samples:
                row.append(sum(samples) / len(samples))
            else:
                row.append(0.0)
        row_fmt = ", ".join(f"{val:10.4f}" for val in row)
        print(f"  {counts}: [{row_fmt}]")
    print()



def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()

    if not repo_root.exists():
        raise SystemExit(f"Repository root {repo_root} does not exist")

    sys.path.insert(0, str(repo_root))
    ga_bundle = repo_root / "marketsim" / "egta" / "gameanalysis-old"
    if ga_bundle.exists():
        sys.path.insert(0, str(ga_bundle))

    from marketsim.egta.core.game import Game  # type: ignore
    from gameanalysis import paygame, regret as ga_regret  # type: ignore
    import gameanalysis.utils as _ga_utils  # type: ignore
    from marketsim.egta.solvers.equilibria import replicator_dynamics  # type: ignore

    try:
        from scipy.special import comb as _sp_comb  # type: ignore
        _ga_utils.comb = _sp_comb  # type: ignore[attr-defined]
    except Exception:
        def _fallback_comb(n, k, exact: bool = False):
            return math.comb(n, k)
        _ga_utils.comb = _fallback_comb  # type: ignore[attr-defined]

    hp_values = sorted(set(args.hp)) if args.hp else discover_hps(repo_root)
    if not hp_values:
        raise SystemExit("No holding-period data found")

    include_melo = not args.seed_only
    target_counts_tuple = tuple(args.counts)

    for hp in hp_values:
        seed_profiles = collect_seed_profiles(repo_root, hp)
        melo_profiles = collect_melo_profiles(repo_root, hp) if include_melo else []
        combined = seed_profiles + melo_profiles if include_melo else seed_profiles
        if not combined:
            print(f"HP {hp}: no payoff data found (skipping)")
            continue

        profile_lookup = build_profile_lookup(combined)

        extra = (
            f" + {len(melo_profiles)} from {MELO_PROFILE_DIR}"
            if include_melo and melo_profiles
            else ""
        )
        print(
            f"HP {hp}: loaded {len(seed_profiles)} profiles from seed roots{extra} "
            f"-> {len(combined)} total"
        )

        if args.verbose:
            _print_neighbour_profile_table(profile_lookup)

        game = Game.from_payoff_data(combined, normalize_payoffs=False)

        ref_game = None
        if seed_profiles:
            try:
                ref_game = Game.from_payoff_data(seed_profiles, normalize_payoffs=False)
            except Exception:
                ref_game = None

        if (
            ref_game is not None
            and ref_game.game.rsg_config_table is not None
            and game.game.rsg_config_table is not None
        ):
            ref_cfgs = {
                tuple(cfg.tolist()): idx
                for idx, cfg in enumerate(ref_game.game.rsg_config_table.t())
            }
            for col_idx, cfg in enumerate(game.game.rsg_config_table.t()):
                ref_idx = ref_cfgs.get(tuple(cfg.tolist()))
                if ref_idx is None:
                    continue
                mask = torch.isnan(game.game.rsg_payoff_table[:, col_idx])
                if mask.any():
                    game.game.rsg_payoff_table[mask, col_idx] = ref_game.game.rsg_payoff_table[mask, ref_idx]
        counts_map = build_counts_map(game, args.counts)
        mixture = counts_to_mixture(game, counts_map)
        regret = float(game.regret(mixture))

        m_counts = args.counts[:2]
        z_counts = args.counts[2:]

        payoff_table = game.game.rsg_payoff_table.detach().clone()
        config_table = game.game.rsg_config_table.detach().clone()
        complete_mask = ~torch.isnan(payoff_table).any(dim=0)
        if not bool(complete_mask.all()):
            dropped = int((~complete_mask).sum().item())
            if args.verbose:
                print(f"[debug] dropping {dropped} configs with NaNs before gameanalysis conversion")
        payoff_table = payoff_table[:, complete_mask]
        config_table = config_table[complete_mask]

        profiles_arr = config_table.cpu().numpy().astype(int)
        payoffs_arr = payoff_table.t().cpu().numpy()
        zero_mask = profiles_arr == 0
        payoffs_arr[zero_mask] = 0.0
        if np.isnan(payoffs_arr).any():
            raise ValueError("NaNs remain in payoff table after filtering")

        num_players = [int(x) for x in game.num_players_per_role]
        ga_game = paygame.game_names(
            game.role_names,
            num_players,
            game.strategy_names_per_role,
            profiles_arr,
            payoffs_arr,
        )
        pay_dbg = ga_game.payoffs()
        nan_count = np.isnan(pay_dbg).sum()
        if nan_count:
            if args.verbose:
                print(f"[debug] gameanalysis payoffs contain {nan_count} NaNs after construction; replacing with zeros")
            pay_dbg = np.nan_to_num(pay_dbg, nan=0.0)
            ga_game = paygame.game_replace(ga_game, ga_game.profiles(), pay_dbg)
        ga_mix = mixture.detach().cpu().numpy()
        ga_reg = float(ga_regret.mixture_regret(ga_game, ga_mix))

        print(
            f"  regret((MOBI {m_counts}, ZI {z_counts})) = {regret:.6f} "
            f"(gameanalysis: {ga_reg:.6f})"
        )

        try:
            mix_rd = replicator_dynamics(
                game,
                mixture=mixture.clone(),
                iters=args.replicator_iters,
                converge_threshold=1e-7,
                use_multiple_starts=False,
            )
            role_sizes = [len(strats) for strats in game.strategy_names_per_role]
            mix_rd = _normalise_per_role(mix_rd, role_sizes)
            reg_rd = float(game.regret(mix_rd))
            rd_info = []
            idx = 0
            for role, strats in zip(game.role_names, game.strategy_names_per_role):
                seg = mix_rd[idx: idx + len(strats)]
                rd_info.append(
                    f"{role} mix = "
                    + ", ".join(f"{strats[j]}={seg[j].item():.3f}" for j in range(len(strats)))
                )
                idx += len(strats)
            print(f"  replicator dynamics regret = {reg_rd:.6f}")
            if args.verbose:
                for line in rd_info:
                    print(f"    {line}")
        except Exception as exc:
            print(f"[warn] replicator dynamics failed: {exc}")

        try:
            local_reg, role_details = compute_local_regret(target_counts_tuple, profile_lookup)
            print(f"  local regret from adjacent profiles = {local_reg:.6f}")
            if args.verbose:
                for role, stats in role_details.items():
                    payoffs_str = ", ".join(f"{k}={v:.4f}" for k, v in stats["payoffs"].items())
                    print(
                        f"    [local] {role}: V={stats['value']:.4f}, "
                        f"best={stats['best']:.4f}, regret={stats['regret']:.4f} | {payoffs_str}"
                    )
        except Exception as exc:
            if args.verbose:
                print(f"[warn] local-regret unavailable: {exc}")

        if args.verbose:
            for line in describe_deviations(game, mixture):
                print(line)


if __name__ == "__main__":
    main()
