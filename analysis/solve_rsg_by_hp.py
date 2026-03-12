import argparse
import json
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple
import numpy as np
import torch
import pathlib, sys, os
import concurrent.futures, multiprocessing as _mp

# -------------------------------------------------------------------
# Optional progress bar
# -------------------------------------------------------------------
try:
    from tqdm import tqdm  # type: ignore
except ImportError:  # graceful fallback
    def tqdm(x, **kwargs):  # pragma: no cover
        return x  # passthrough iterator

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from marketsim.game.role_symmetric_game import RoleSymmetricGame
from marketsim.egta.core.game import Game
from marketsim.egta.solvers.equilibria import replicator_dynamics, regret as game_regret


def load_profiles_from_file(fpath: Path) -> List[List[Tuple[str, str, str, float]]]:
    """Return list-of-profiles loaded from *fpath*."""
    with open(fpath) as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"JSON error in {fpath}: {e}")
            return []
    
    if not isinstance(data, list):
        return []
    return data 


def group_profiles_by_hp(root: Path) -> Dict[int, List[List[Tuple[str, str, str, float]]]]:
    """Walk *root* collecting profiles grouped by holding-period integer."""
    grouped: Dict[int, List[List[Tuple[str, str, str, float]]]] = defaultdict(list)

    for hp_dir in root.rglob("holding_period_*"):
        
        import re
        m = re.search(r"holding_period_(\d+)", hp_dir.name)
        if m is None:
            continue
        hp_val = int(m.group(1))
        for pf in hp_dir.glob("**/raw_payoff_data.json"):
            grouped[hp_val].extend(load_profiles_from_file(pf))
    return grouped


def infer_metadata(profiles: List[List[Tuple[str, str, str, float]]]):
    """Infer role names, player counts, strategy lists from sample *profiles*."""
    if not profiles:
        raise ValueError("No profiles passed to infer_metadata()")
    
    role_set = set()
    role_strat_set: Dict[str, set] = defaultdict(set)
    role_counts: Counter = Counter()
    
    for prof in profiles:
        for _, role, strat, _ in prof:
            role_set.add(role)
            role_strat_set[role].add(strat)
        # assume player counts constant across profiles
        role_counts.update([r for _, r, _, _ in prof])
        break  

    role_names = sorted(role_set)
    num_players_per_role = [role_counts[r] for r in role_names]
    strategy_names_per_role = [sorted(role_strat_set[r]) for r in role_names]
    return role_names, num_players_per_role, strategy_names_per_role


def solve_rsg(payoff_data: List[List[Tuple[str, str, str, float]]], meta, *, min_obs: int = 10, device="cpu"):
    role_names, num_players, strat_names = meta

    # ------------------------------------------------------------------
    # 1) Filter profiles with < min_obs observations
    # ------------------------------------------------------------------
    prof_counter: Dict[Tuple[Tuple[str,str], ...], List[List[Tuple[str,str,str,float]]]] = defaultdict(list)
    for prof in payoff_data:
        key = tuple(sorted((r,s) for _, r, s, _ in prof))
        prof_counter[key].append(prof)

    filtered_profiles: List[List[Tuple[str,str,str,float]]] = []
    for lst in prof_counter.values():
        if len(lst) >= min_obs:
            filtered_profiles.extend(lst)

    rsg = RoleSymmetricGame.from_payoff_data_rsg(
        filtered_profiles,
        role_names=role_names,
        num_players_per_role=num_players,
        strategy_names_per_role=strat_names,
        device=device,
    )

    # -------------------------------------------------------------
    # Interpolate missing payoffs: replace NaNs with row-mean value
    # -------------------------------------------------------------
    if rsg.rsg_payoff_table is not None and torch.isnan(rsg.rsg_payoff_table).any():
        tbl = rsg.rsg_payoff_table.clone()
        for i in range(tbl.shape[0]):
            nan_mask = torch.isnan(tbl[i])
            if nan_mask.any():
                if (~nan_mask).any():
                    mean_val = tbl[i, ~nan_mask].mean()
                    tbl[i, nan_mask] = mean_val - 1e-6  # conservative
                else:
                    # entire row NaN → set zeros
                    tbl[i, nan_mask] = 0.0
        rsg.rsg_payoff_table = tbl

    game = Game(rsg)

    seeds = []

    # Helper to create role-wise biased mixture toward first strategy
    def make_bias(weight_first: float) -> torch.Tensor:
        mix_vec = torch.zeros(rsg.num_strategies, dtype=torch.float32)
        g = 0
        for strat_list in rsg.strategy_names_per_role:
            n_s = len(strat_list)
            if n_s == 0:
                continue
            mix_vec[g] = weight_first
            if n_s > 1:
                mix_vec[g + 1 : g + n_s] = (1 - weight_first) / (n_s - 1)
            g += n_s
        return mix_vec

    for w in [0.9, 0.8, 0.7, 0.6]:
        seeds.append(make_bias(w))

    # pure on first strategy of each role
    seed_pure = make_bias(1.0)
    seeds.append(seed_pure)

    # explicit corner seeds if they exist (CDA / MELO)
    def build_corner_vec(keyword: str):
        kw = keyword.lower()
        vec = torch.zeros(rsg.num_strategies, dtype=torch.float32)
        off = 0
        for strat_list in rsg.strategy_names_per_role:
            idx_match = next((j for j, n in enumerate(strat_list) if kw in n.lower()), None)
            if idx_match is None:
                return None
            vec[off + idx_match] = 1.0
            off += len(strat_list)
        return vec

    for corner_kw in ["_100_0", "_0_100"]:
        v = build_corner_vec(corner_kw)
        if v is not None:
            seeds.append(v)

    best_mix = None
    best_reg = float("inf")
    for s in seeds:
        m = replicator_dynamics(game, mixture=s, iters=3000)
        r_val = game_regret(game, m)
        rv = r_val.item() if torch.is_tensor(r_val) else float(r_val)
        if rv < best_reg:
            best_reg = rv
            best_mix = m

    mix = best_mix
    reg = best_reg

    return mix.cpu().numpy().tolist(), best_reg, rsg


def _solve_entry(entry):
    """Worker: solve one HP group.

    Parameters
    ----------
    entry : tuple
        (hp_val, profiles, regret_threshold)
    """

    hp_val, profiles, regret_thr, min_obs = entry
    if not profiles:
        return hp_val, None

    meta_local = infer_metadata(profiles)
    mix, reg, rsg = solve_rsg(profiles, meta_local, min_obs=min_obs)

    eq_list = []
    def _is_near_uniform(mix_vec: List[float]) -> bool:
        """Return True if every role with 2 strategies is ~0.5/0.5 (±0.01)."""
        idx = 0
        for s_list in rsg.strategy_names_per_role:
            rs = len(s_list)
            slice_vals = mix_vec[idx: idx + rs]
            idx += rs
            if rs == 2 and all(0.49 <= p <= 0.51 for p in slice_vals):
                continue
            else:
                return False
        return True

    if reg <= regret_thr and not _is_near_uniform(mix):
        eq_list.append({"type": "replicator", "regret": reg, "mixture": mix})

    # corners helper
    def build_corner(keyword: str):
        kw = keyword.lower()
        vec = np.zeros(rsg.num_strategies, dtype=float)
        off = 0
        for strat_list in rsg.strategy_names_per_role:
            cand = [j for j, n in enumerate(strat_list) if kw in n.lower()]
            if not cand:
                return None
            vec[off + cand[0]] = 1.0
            off += len(strat_list)
        return vec

    for cn in ["_100_0", "_0_100"]:
        corner = build_corner(cn)
        if corner is not None:
            reg_c = game_regret(Game(rsg), torch.tensor(corner, dtype=torch.float32))
            reg_c_val = reg_c.item() if torch.is_tensor(reg_c) else float(reg_c)
            if reg_c_val <= regret_thr:
                eq_list.append({"type": f"corner{cn}", "regret": reg_c_val, "mixture": corner.tolist()})

    if not eq_list:
        # if all others filtered out, keep replicator even if uniform so we have at least one solution
        eq_list.append({"type": "replicator", "regret": reg, "mixture": mix})

    return hp_val, {
        "equilibria": eq_list,
        "role_names": meta_local[0],
        "strategy_names_per_role": meta_local[2],
    }


def main():
    parser = argparse.ArgumentParser(description="Solve RSGs grouped by holding period.")
    parser.add_argument("root_dir", type=Path, help="Directory containing pilot_egta_run_* folders")
    parser.add_argument("--out", type=Path, default=Path("rsg_solutions.json"), help="Output JSON file")
    parser.add_argument("--regret-threshold", type=float, default=1e-4, help="Max regret for accepting a mixture as equilibrium")
    parser.add_argument("--jobs", type=int, default=max(_mp.cpu_count() - 1, 1), help="Parallel processes (default: CPU-1)")
    parser.add_argument("--min-obs", type=int, default=10, help="Minimum observations per profile to keep it in the RSG table")
    args = parser.parse_args()

    grouped = group_profiles_by_hp(args.root_dir)
    solutions: Dict[int, dict] = {}

    entries = [(hp, profs, args.regret_threshold, args.min_obs) for hp, profs in grouped.items()]

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.jobs, mp_context=_mp.get_context("spawn")) as ex:
        for hp_val, result in tqdm(ex.map(_solve_entry, entries), total=len(entries), desc="Solving HP"):
            if result is not None:
                solutions[hp_val] = result

                flat_names: List[str] = []
                for rn, s_list in zip(result["role_names"], result["strategy_names_per_role"]):
                    for sn in s_list:
                        flat_names.append(f"{rn}:{sn}")

                best_reg = min(e['regret'] for e in result['equilibria'])
                print(f"HP {hp_val}: stored {len(result['equilibria'])} equilibria (best regret={best_reg:.3e})")

                for eq in result["equilibria"]:
                    mix_vec = eq["mixture"]

                    pretty_parts = []
                    idx = 0
                    for rn, strat_list in zip(result["role_names"], result["strategy_names_per_role"]):
                        role_size = len(strat_list)
                        role_slice = mix_vec[idx: idx + role_size]
                        idx += role_size
                        role_desc = ", ".join(
                            [f"{sn}:{p:.2f}" for sn, p in zip(strat_list, role_slice) if p > 0.005]
                        )
                        pretty_parts.append(f"{rn}[{role_desc}]")

                    print("   ·", eq["type"], f"reg={eq['regret']:.2e} ->", " | ".join(pretty_parts))

    args.out.write_text(json.dumps(solutions, indent=2))
    print(f"Saved solutions to {args.out}")


if __name__ == "__main__":
    main() 