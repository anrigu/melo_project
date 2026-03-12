#!/usr/bin/env python3
"""
Scan the standard EGTA results plus melo_profile_runs_28_rest_ZIs, build the
role-symmetric game per holding period, and report which DPR baseline profiles
(28, 0, k, 40-k) are free of profitable deviations.

Run from the project root:
    PYTHONPATH=$PWD python analysis/check_dpr_baselines.py
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict, Counter, OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

from marketsim.egta.core.game import Game
from marketsim.game.role_symmetric_game import RoleSymmetricGame

ROOT = Path(__file__).resolve().parent.parent

SEED_ROOTS = [
    # "result_two_role_still_role_symmetric_3",
    # "result_two_role_still_role_symmetric_4",
    # "result_two_role_still_role_symmetric_5",
    # "result_two_role_still_role_symmetric_6",
    # "result_two_role_still_role_symmetric_7",
    # "gapfill_profiles2",
    # "gapfill_profiles",
    # "gapfill_profiles3",
    # "gapfill_profiles6",
    # "gapfill_profiles_tester",
    # New extra source:
    "melo_profile_runs_28_rest_ZIs",
    #"melo_profile_28_50k"
]

CANONICAL_ORDER = [
    ("MOBI", "MOBI_0_100_shade0_0"),
    ("MOBI", "MOBI_100_0_shade250_500"),
    ("ZI",   "ZI_0_100_shade250_500"),
    ("ZI",   "ZI_100_0_shade250_500"),
]

HP_RE = re.compile(r"holding_period_(\d+)")

def load_profiles_from_path(path: Path) -> List[List[Tuple[str, str, str, float]]]:
    """Return list of legacy profiles read from *path*."""
    try:
        data = json.load(path.open())
    except Exception:
        return []

    profiles: List[List[Tuple[str, str, str, float]]] = []

    # Case 1: raw_payoff_data.json style → list of profiles
    if isinstance(data, list) and data and isinstance(data[0], (list, tuple, dict)):
        for prof in data:
            prof_rows: List[Tuple[str, str, str, float]] = []
            if isinstance(prof, dict):
                # Sometimes a dict keyed by 'legacy_rows'
                rows = prof.get("legacy_rows", [])
            else:
                rows = prof
            for idx, row in enumerate(rows):
                if isinstance(row, (list, tuple)) and len(row) >= 4:
                    pid, role, strat, payoff = row[:4]
                elif isinstance(row, dict):
                    pid = row.get("player_id") or f"p{idx}"
                    role = row.get("role") or row.get("Role")
                    strat = row.get("strategy") or row.get("Strategy")
                    payoff = row.get("payoff")
                else:
                    continue
                if role is None or strat is None or payoff is None:
                    continue
                prof_rows.append((str(pid), str(role), str(strat), float(payoff)))
            if prof_rows:
                profiles.append(prof_rows)
        return profiles

    if isinstance(data, list):
        for idx, entry in enumerate(data):
            if not isinstance(entry, dict):
                continue
            legacy = entry.get("legacy_rows")
            if not legacy:
                continue
            prof_rows = []
            for jdx, row in enumerate(legacy):
                if isinstance(row, (list, tuple)) and len(row) >= 4:
                    pid, role, strat, payoff = row[:4]
                else:
                    continue
                prof_rows.append((str(pid), str(role), str(strat), float(payoff)))
            if prof_rows:
                profiles.append(prof_rows)
        return profiles

    return []

def collect_profiles_by_hp() -> Dict[int, List[List[Tuple[str, str, str, float]]]]:
    """Aggregate profiles per holding period across all seed roots."""
    hp_profiles: Dict[int, List[List[Tuple[str, str, str, float]]]] = defaultdict(list)

    for root_name in SEED_ROOTS:
        root_dir = ROOT / root_name
        if not root_dir.exists():
            continue

        for raw_path in root_dir.glob("**/raw_payoff_data.json"):
            hp = extract_hp(raw_path)
            if hp is None:
                continue
            hp_profiles[hp].extend(load_profiles_from_path(raw_path))

        for res_path in root_dir.glob("**/results.json"):
            hp = extract_hp(res_path)
            if hp is None:
                continue
            hp_profiles[hp].extend(load_profiles_from_path(res_path))

    return hp_profiles

def extract_hp(path: Path) -> int | None:
    """Extract holding period integer from *path*."""
    match = HP_RE.search(str(path))
    return int(match.group(1)) if match else None

def build_game_from_profiles(
    profiles: List[List[Tuple[str, str, str, float]]]
) -> Game | None:
    """Construct RoleSymmetricGame (wrapped by Game) from payoff rows."""
    if not profiles:
        return None
    role_strategy_order: OrderedDict[str, List[str]] = OrderedDict()
    for role, strat in CANONICAL_ORDER:
        role_strategy_order.setdefault(role, [])
        if strat not in role_strategy_order[role]:
            role_strategy_order[role].append(strat)

    role_max_counts: Counter[str] = Counter()
    for profile in profiles:
        per_role_counts: Counter[str] = Counter()
        for _, role, strat, _ in profile:
            per_role_counts[role] += 1
            strat_list = role_strategy_order.setdefault(role, [])
            if strat not in strat_list:
                strat_list.append(strat)
        for role, count in per_role_counts.items():
            current = role_max_counts.get(role, 0)
            if count > current:
                role_max_counts[role] = count

    role_names = list(role_strategy_order.keys())
    if not role_names:
        return None

    num_players_per_role = [role_max_counts.get(role, 0) for role in role_names]
    if any(count <= 0 for count in num_players_per_role):
        print("[warn] Could not infer player counts per role from payoff data.")
        return None

    strategy_names_per_role = [role_strategy_order[role] for role in role_names]

    try:
        rsg = RoleSymmetricGame.from_payoff_data_rsg(
            payoff_data=profiles,
            role_names=role_names,
            num_players_per_role=num_players_per_role,
            strategy_names_per_role=strategy_names_per_role,
            normalize_payoffs=False,
        )
    except Exception as exc:
        print(f"[warn] RoleSymmetricGame.from_payoff_data_rsg failed: {exc}")
        return None

    return Game(rsg)

def check_dpr_baselines(game: Game, *, tol: float = 0.0) -> Dict[int, Dict[str, float]]:
    """Return per-k diagnostics for profiles (28,0,k,40-k)."""
    if not hasattr(game, "game") or not hasattr(game.game, "rsg_config_table"):
        return {}

    cfg = game.game.rsg_config_table.cpu().numpy().astype(int)
    pay = game.game.rsg_payoff_table.cpu().numpy()
    config_to_col = {
        tuple(int(x) for x in row): idx for idx, row in enumerate(cfg)
    }

    results: Dict[int, Dict[str, float]] = {}
    for k in range(41):
        target = [28, 0, k, 40 - k]
        key = tuple(target)
        col = config_to_col.get(key)
        if col is None:
            results[k] = {"status": "missing"}
            continue

        pay_vec = pay[:, col].astype(float)
        status = "equilibrium"
        notes: List[str] = []
        missing_neighbor = False

        # Only check ZI deviations as requested
        for role_name, strat_indices in (("ZI", [2, 3]),):
            counts = [target[i] for i in strat_indices]
            payoffs = [float(pay_vec[i]) for i in strat_indices]
            role_has_support = any(c > 0 for c in counts)

            if not role_has_support:
                notes.append(f"{role_name}: empty support")
                continue

            for strat_idx, count, strat_pay in zip(strat_indices, counts, payoffs):
                if count <= 0:
                    continue
                if math.isnan(strat_pay):
                    missing_neighbor = True
                    notes.append(
                        f"{role_name} {CANONICAL_ORDER[strat_idx][1]} payoff at k={k} is NaN"
                    )
                    continue

                for alt_idx in strat_indices:
                    if alt_idx == strat_idx:
                        continue
                    neighbor = target.copy()
                    neighbor[strat_idx] -= 1
                    neighbor[alt_idx] += 1
                    if neighbor[strat_idx] < 0 or neighbor[alt_idx] < 0:
                        continue

                    neighbor_key = tuple(neighbor)
                    neighbor_col = config_to_col.get(neighbor_key)
                    if neighbor_col is None:
                        missing_neighbor = True
                        notes.append(
                            f"{role_name} {CANONICAL_ORDER[strat_idx][1]}→{CANONICAL_ORDER[alt_idx][1]} "
                            f"k'={neighbor[2]} profile missing"
                        )
                        continue

                    alt_pay = float(pay[alt_idx, neighbor_col])
                    if math.isnan(alt_pay):
                        missing_neighbor = True
                        notes.append(
                            f"{role_name} {CANONICAL_ORDER[strat_idx][1]}→{CANONICAL_ORDER[alt_idx][1]} "
                            f"k'={neighbor[2]} payoff NaN"
                        )
                        continue
                    delta = alt_pay - strat_pay
                    if delta > tol:
                        status = "profitable-deviation"
                        notes.append(
                            f"{role_name} {CANONICAL_ORDER[strat_idx][1]}→{CANONICAL_ORDER[alt_idx][1]} "
                            f"k'={neighbor[2]} improves by {delta:.2f}"
                        )

        if status == "equilibrium":
            if missing_neighbor:
                notes.append("neighbor data missing")
            if k in {0, 40}:
                notes.append("corner profile")

        results[k] = {
            "status": status,
            "payoffs": {
                CANONICAL_ORDER[i][1]: float(pay_vec[i])
                for i in range(len(CANONICAL_ORDER))
            },
            "notes": "; ".join(notes),
        }
    return results

def main() -> None:
    hp_profiles = collect_profiles_by_hp()
    if not hp_profiles:
        print("No profiles found.")
        return

    hp_equilibria_summary: List[Tuple[int, List[Tuple[int, List[int], str]]]] = []

    for hp in sorted(hp for hp in hp_profiles if hp % 20 == 0 and hp <= 400):
        profiles = hp_profiles[hp]
        print(f"\n=== Holding Period {hp} ===")
        print(f"  Loaded {len(profiles)} profiles")

        game = build_game_from_profiles(profiles)
        if game is None or not hasattr(game, "game") or game.game.rsg_config_table is None:
            print("  Could not build RoleSymmetricGame.")
            continue

        summary = check_dpr_baselines(game)
        cfg = game.game.rsg_config_table.cpu().numpy().astype(int)
        equilibria: List[Tuple[int, List[int], str]] = []

        for k in range(41):
            info = summary.get(k)
            if info is None:
                print(f"  k={k:2d}: missing")
                continue
            target = [28, 0, k, 40 - k]
            status = info["status"]
            notes = info.get("notes", "")
            payoffs = info.get("payoffs", {})
            pay_str = ", ".join(f"{s}={payoffs.get(s, float('nan')):.1f}" for _, s in CANONICAL_ORDER)
            lines = [
                f"  k={k:2d}: {status:20s} | {pay_str}"
            ]
            if status != "missing":
                match = (cfg == target).all(axis=1)
                if match.any():
                    lines.append(f"          profile: {target}")
            for line in lines:
                print(line)
            if notes:
                print(f"          → {notes}")
            if status == "equilibrium":
                equilibria.append((k, target, pay_str))

        hp_equilibria_summary.append((hp, equilibria))
        if equilibria:
            print("  Equilibria without profitable deviation:")
            for k, target, pay_str in equilibria:
                print(f"    k={k:2d}: profile {target} | {pay_str}")
        else:
            print("  Equilibria without profitable deviation: none")

    print("\n=== Equilibria Summary ===")
    for hp, equilibria in hp_equilibria_summary:
        if equilibria:
            print(f"HP {hp}:")
            for k, target, pay_str in equilibria:
                print(f"  k={k:2d}: profile {target} | {pay_str}")
        else:
            print(f"HP {hp}: none")

if __name__ == "__main__":
    main()

"""
#!/usr/bin/env python3
"""
# Scan the standard EGTA results plus melo_profile_runs_28_rest_ZIs, build the
# role-symmetric game per holding period, and report which DPR baseline profiles
# (28, 0, k, 40-k) are free of profitable deviations.

# Run from the project root:
#     PYTHONPATH=$PWD python analysis/check_dpr_baselines.py
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

from marketsim.egta.core.game import Game

ROOT = Path(__file__).resolve().parent.parent

SEED_ROOTS = [
    # "result_two_role_still_role_symmetric_3",
    # "result_two_role_still_role_symmetric_4",
    # "result_two_role_still_role_symmetric_5",
    # "result_two_role_still_role_symmetric_6",
    # "result_two_role_still_role_symmetric_7",
    # "gapfill_profiles2",
    # "gapfill_profiles",
    # "gapfill_profiles3",
    # "gapfill_profiles6",
    # "gapfill_profiles_tester",
    # New extra source:
    "melo_profile_runs_28_rest_ZIs",
]

CANONICAL_ORDER = [
    ("MOBI", "MOBI_0_100_shade0_0"),
    ("MOBI", "MOBI_100_0_shade250_500"),
    ("ZI",   "ZI_0_100_shade250_500"),
    ("ZI",   "ZI_100_0_shade250_500"),
]

HP_RE = re.compile(r"holding_period_(\d+)")

def load_profiles_from_path(path: Path) -> List[List[Tuple[str, str, str, float]]]:
    
    #Return list of legacy profiles read from *path*.
    try:
        data = json.load(path.open())
    except Exception:
        return []

    profiles: List[List[Tuple[str, str, str, float]]] = []

    # Case 1: raw_payoff_data.json style → list of profiles
    if isinstance(data, list) and data and isinstance(data[0], (list, tuple, dict)):
        for prof in data:
            prof_rows: List[Tuple[str, str, str, float]] = []
            if isinstance(prof, dict):
                # Sometimes a dict keyed by 'legacy_rows'
                rows = prof.get("legacy_rows", [])
            else:
                rows = prof
            for idx, row in enumerate(rows):
                if isinstance(row, (list, tuple)) and len(row) >= 4:
                    pid, role, strat, payoff = row[:4]
                elif isinstance(row, dict):
                    pid = row.get("player_id") or f"p{idx}"
                    role = row.get("role") or row.get("Role")
                    strat = row.get("strategy") or row.get("Strategy")
                    payoff = row.get("payoff")
                else:
                    continue
                if role is None or strat is None or payoff is None:
                    continue
                prof_rows.append((str(pid), str(role), str(strat), float(payoff)))
            if prof_rows:
                profiles.append(prof_rows)
        return profiles

    if isinstance(data, list):
        for idx, entry in enumerate(data):
            if not isinstance(entry, dict):
                continue
            legacy = entry.get("legacy_rows")
            if not legacy:
                continue
            prof_rows = []
            for jdx, row in enumerate(legacy):
                if isinstance(row, (list, tuple)) and len(row) >= 4:
                    pid, role, strat, payoff = row[:4]
                else:
                    continue
                prof_rows.append((str(pid), str(role), str(strat), float(payoff)))
            if prof_rows:
                profiles.append(prof_rows)
        return profiles

    return []

def collect_profiles_by_hp() -> Dict[int, List[List[Tuple[str, str, str, float]]]]:
    #Aggregate profiles per holding period across all seed roots.
    hp_profiles: Dict[int, List[List[Tuple[str, str, str, float]]]] = defaultdict(list)

    for root_name in SEED_ROOTS:
        root_dir = ROOT / root_name
        if not root_dir.exists():
            continue

        for raw_path in root_dir.glob("**/raw_payoff_data.json"):
            hp = extract_hp(raw_path)
            if hp is None:
                continue
            hp_profiles[hp].extend(load_profiles_from_path(raw_path))

        for res_path in root_dir.glob("**/results.json"):
            hp = extract_hp(res_path)
            if hp is None:
                continue
            hp_profiles[hp].extend(load_profiles_from_path(res_path))

    return hp_profiles

def extract_hp(path: Path) -> int | None:
    #Extract holding period integer from *path*.
    match = HP_RE.search(str(path))
    return int(match.group(1)) if match else None

def build_game_from_profiles(
    profiles: List[List[Tuple[str, str, str, float]]]
) -> Game | None:
    #Construct RoleSymmetricGame (wrapped by Game) from payoff rows.
    if not profiles:
        return None
    try:
        return Game.from_payoff_data(profiles, normalize_payoffs=False)
    except Exception as exc:
        print(f"[warn] Game.from_payoff_data failed: {exc}")
        return None

def check_dpr_baselines(game: Game, *, tol: float = 0.0) -> Dict[int, Dict[str, float]]:
    #Return per-k diagnostics for profiles (28,0,k,40-k).
    if not hasattr(game, "game") or not hasattr(game.game, "rsg_config_table"):
        return {}

    cfg = game.game.rsg_config_table.cpu().numpy().astype(int)
    pay = game.game.rsg_payoff_table.cpu().numpy()
    config_to_col = {
        tuple(int(x) for x in row): idx for idx, row in enumerate(cfg)
    }

    results: Dict[int, Dict[str, float]] = {}
    for k in range(41):
        target = [28, 0, k, 40 - k]
        key = tuple(target)
        col = config_to_col.get(key)
        if col is None:
            results[k] = {"status": "missing"}
            continue

        pay_vec = pay[:, col].astype(float)
        status = "equilibrium"
        notes: List[str] = []
        missing_neighbor = False

        # Only check ZI deviations as requested
        for role_name, strat_indices in (("ZI", [2, 3]),):
            counts = [target[i] for i in strat_indices]
            payoffs = [float(pay_vec[i]) for i in strat_indices]
            role_has_support = any(c > 0 for c in counts)

            if not role_has_support:
                notes.append(f"{role_name}: empty support")
                continue

            for strat_idx, count, strat_pay in zip(strat_indices, counts, payoffs):
                if count <= 0:
                    continue
                if math.isnan(strat_pay):
                    missing_neighbor = True
                    notes.append(
                        f"{role_name} {CANONICAL_ORDER[strat_idx][1]} payoff at k={k} is NaN"
                    )
                    continue

                for alt_idx in strat_indices:
                    if alt_idx == strat_idx:
                        continue
                    neighbor = target.copy()
                    neighbor[strat_idx] -= 1
                    neighbor[alt_idx] += 1
                    if neighbor[strat_idx] < 0 or neighbor[alt_idx] < 0:
                        continue

                    neighbor_key = tuple(neighbor)
                    neighbor_col = config_to_col.get(neighbor_key)
                    if neighbor_col is None:
                        missing_neighbor = True
                        notes.append(
                            f"{role_name} {CANONICAL_ORDER[strat_idx][1]}→{CANONICAL_ORDER[alt_idx][1]} "
                            f"k'={neighbor[2]} profile missing"
                        )
                        continue

                    alt_pay = float(pay[alt_idx, neighbor_col])
                    if math.isnan(alt_pay):
                        missing_neighbor = True
                        notes.append(
                            f"{role_name} {CANONICAL_ORDER[strat_idx][1]}→{CANONICAL_ORDER[alt_idx][1]} "
                            f"k'={neighbor[2]} payoff NaN"
                        )
                        continue
                    delta = alt_pay - strat_pay
                    if delta > tol:
                        status = "profitable-deviation"
                        notes.append(
                            f"{role_name} {CANONICAL_ORDER[strat_idx][1]}→{CANONICAL_ORDER[alt_idx][1]} "
                            f"k'={neighbor[2]} improves by {delta:.2f}"
                        )

        if status == "equilibrium":
            if missing_neighbor:
                notes.append("neighbor data missing")
            if k in {0, 40}:
                notes.append("corner profile")

        results[k] = {
            "status": status,
            "payoffs": {
                CANONICAL_ORDER[i][1]: float(pay_vec[i])
                for i in range(len(CANONICAL_ORDER))
            },
            "notes": "; ".join(notes),
        }
    return results

def main() -> None:
    hp_profiles = collect_profiles_by_hp()
    if not hp_profiles:
        print("No profiles found.")
        return

    hp_equilibria_summary: List[Tuple[int, List[Tuple[int, List[int], str]]]] = []

    for hp in sorted(hp for hp in hp_profiles if hp % 20 == 0 and hp <= 400):
        profiles = hp_profiles[hp]
        print(f"\n=== Holding Period {hp} ===")
        print(f"  Loaded {len(profiles)} profiles")

        game = build_game_from_profiles(profiles)
        if game is None or not hasattr(game, "game") or game.game.rsg_config_table is None:
            print("  Could not build RoleSymmetricGame.")
            continue

        summary = check_dpr_baselines(game)
        cfg = game.game.rsg_config_table.cpu().numpy().astype(int)
        equilibria: List[Tuple[int, List[int], str]] = []

        for k in range(41):
            info = summary.get(k)
            if info is None:
                print(f"  k={k:2d}: missing")
                continue
            target = [28, 0, k, 40 - k]
            status = info["status"]
            notes = info.get("notes", "")
            payoffs = info.get("payoffs", {})
            pay_str = ", ".join(f"{s}={payoffs.get(s, float('nan')):.1f}" for _, s in CANONICAL_ORDER)
            lines = [
                f"  k={k:2d}: {status:20s} | {pay_str}"
            ]
            if status != "missing":
                match = (cfg == target).all(axis=1)
                if match.any():
                    lines.append(f"          profile: {target}")
            for line in lines:
                print(line)
            if notes:
                print(f"          → {notes}")
            if status == "equilibrium":
                equilibria.append((k, target, pay_str))

        hp_equilibria_summary.append((hp, equilibria))
        if equilibria:
            print("  Equilibria without profitable deviation:")
            for k, target, pay_str in equilibria:
                print(f"    k={k:2d}: profile {target} | {pay_str}")
        else:
            print("  Equilibria without profitable deviation: none")

    print("\n=== Equilibria Summary ===")
    for hp, equilibria in hp_equilibria_summary:
        if equilibria:
            print(f"HP {hp}:")
            for k, target, pay_str in equilibria:
                print(f"  k={k:2d}: profile {target} | {pay_str}")
        else:
            print(f"HP {hp}: none")

if __name__ == "__main__":
    main()

"""