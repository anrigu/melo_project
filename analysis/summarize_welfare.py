from __future__ import annotations

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#!/usr/bin/env python3
from marketsim.egta.core.game import RoleSymmetricGame

"""
Summarise welfare metrics for every equilibrium.

Changes relative to the original:
  • Uses game.mixture_values rather than hand-rolling from deviation payoffs.
  • Robust to NaN / inf both in raw observations and mixture evaluation.
  • Sanitises the output in one sweep.
"""


import json, os, math, sys
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch

# --------------------------------------------------------------------------- #
# CONFIG ––– adapt just these paths                                           #
# --------------------------------------------------------------------------- #
RESULTS_DIR     = "/Users/gabesmithline/Desktop/100_2_strats_no_shade/comprehensive_rsg_results_20250614_185419"
EQS_FILE        = os.path.join(RESULTS_DIR, "equilibria_detailed.json")
GAME_DETAILS    = os.path.join(RESULTS_DIR, "game_details.json")
RAW_PAYOFF_DATA = os.path.join(RESULTS_DIR, "raw_payoff_data.json")
OBS_FILE        = os.path.join(RESULTS_DIR, "observations.json")
OUTPUT_JSON     = os.path.join(RESULTS_DIR, "raw_welfare_metrics_with_stats_250.json")

# --------------------------------------------------------------------------- #
# tiny helper                                                                 #
# --------------------------------------------------------------------------- #
def _sanitize(obj):
    """Recursively replace NaN/Inf with null so json.dump never fails."""
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj


# --------------------------------------------------------------------------- #
# load helpers                                                                #
# --------------------------------------------------------------------------- #
def load_equilibria(path: str) -> List[dict]:
    with open(path) as f:
        return json.load(f)

def load_observations(path: str) -> List[dict]:
    with open(path) as f:
        return json.load(f)

def compute_optimum_welfare(obs: List[dict]) -> float:
    """Highest average *sum* of player payoffs across all fully-sampled profiles."""
    sums = defaultdict(list)
    for e in obs:
        if not e["payoffs"]:
            continue
        prof = tuple(tuple(p) for p in e["profile"])
        if any(map(lambda x: x is None or math.isnan(x) or math.isinf(x), e["payoffs"])):
            continue
        sums[prof].append(sum(e["payoffs"]))
    if not sums:
        return 0.0
    return max(np.mean(v) for v in sums.values())

def load_game(details_path: str, payoff_path: str, device: torch.device):
     # local import
    meta = json.load(open(details_path))
    raw  = json.load(open(payoff_path))
    return RoleSymmetricGame.from_payoff_data_rsg(
        payoff_data             = raw,
        role_names              = meta["role_names"],
        num_players_per_role    = meta["num_players_per_role"],
        strategy_names_per_role = meta["strategy_names_per_role"],
        device                  = device,
        normalize_payoffs       = False
    )

# --------------------------------------------------------------------------- #
# If your RoleSymmetricGame lacks mixture_values, add this (5 lines)          #
# class RoleSymmetricGame:                                                   #
#     ...                                                                    #
#     def mixture_values(self, mix_tensor: torch.Tensor) -> torch.Tensor:     #
#         """Return per-role expected payoff of the mixture itself."""        #
#         return (mix_tensor * self.deviation_payoffs(mix_tensor)).sum(dim=1)#
# --------------------------------------------------------------------------- #

def shannon_entropy(p_dict: Dict[str, float]) -> float:
    ps = np.array(list(p_dict.values()), dtype=float)
    ps = ps[ps > 0]
    return float(-(ps * np.log(ps)).sum()) if ps.size else 0.0


# --------------------------------------------------------------------------- #
# MAIN                                                                        #
# --------------------------------------------------------------------------- #
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    equilibria   = load_equilibria(EQS_FILE)
    observations = load_observations(OBS_FILE)
    game         = load_game(GAME_DETAILS, RAW_PAYOFF_DATA, device)
    opt_welfare  = compute_optimum_welfare(observations)

    # ---------------------------------------------------------------
    # Quick per-strategy raw payoff averages to avoid zero-payoff bug
    # ---------------------------------------------------------------
    avg_by_role: Dict[str, Dict[str, float]] = defaultdict(dict)
    count_by_role: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for prof in observations:
        for (role, strat), payoff in zip(prof["profile"], prof["payoffs"]):
            if payoff is None or math.isnan(payoff) or math.isinf(payoff):
                continue
            avg_by_role[role].setdefault(strat, 0.0)
            count_by_role[role][strat] += 1
            avg_by_role[role][strat] += payoff

    # turn sums into means
    for role, smap in avg_by_role.items():
        for strat in smap:
            smap[strat] /= count_by_role[role][strat]

    summaries: List[dict] = []

    for eq in equilibria:
        mix = torch.tensor(eq["mixture_vector"], dtype=torch.float32, device=device)

        # -----------------------------------------------------------
        # Expected payoffs via simple per-strategy means (raw scale)
        # -----------------------------------------------------------
        mix_vals_by_role = {}
        for role, dist in eq["mixture_by_role"].items():
            epay = 0.0
            for strat, prob in dist.items():
                epay += prob * avg_by_role.get(role, {}).get(strat, 0.0)
            mix_vals_by_role[role] = epay

        # ------------------------------------------------------------------ #
        # aggregate per role                                                 #
        # ------------------------------------------------------------------ #
        role_epay, role_reg = {}, {}
        collective = total_regret = 0.0
        offset = 0
        for ridx, rname in enumerate(game.role_names):
            n_s = len(game.strategy_names_per_role[ridx])
            r_strats = mix[offset:offset + n_s]
            r_mix    = mix[offset:offset + n_s]

            role_name = rname
            expected_pay = mix_vals_by_role.get(role_name, 0.0)

            # Best dev: use max avg payoff for role
            if avg_by_role.get(role_name):
                best_dev = max(avg_by_role[role_name].values())
            else:
                best_dev = 0.0

            reg          = best_dev - expected_pay

            n_players = int(game.num_players_per_role[ridx])
            collective      += expected_pay * n_players
            total_regret    += reg * n_players
            role_epay[rname] = expected_pay
            role_reg[rname]  = reg

            offset += n_s

        price_of_anarchy = (opt_welfare / collective) if collective > 0 else None
        payoff_gap       = opt_welfare - collective

        entropy_by_role = {
            role: shannon_entropy(dist) for role, dist in eq["mixture_by_role"].items()
        }

        summaries.append({
            "equilibrium_id":       eq["equilibrium_id"],
            "mixture_by_role":      eq["mixture_by_role"],
            "role_expected_payoffs": role_epay,
            "role_regrets":         role_reg,
            "collective_welfare":   collective,
            "total_regret":         total_regret,
            "price_of_anarchy":     price_of_anarchy,
            "entropy":              entropy_by_role,
            "payoff_gap":           payoff_gap,
        })

    with open(OUTPUT_JSON, "w") as fp:
        json.dump(_sanitize(summaries), fp, indent=2, allow_nan=False)

    print(f"✓ wrote summary to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()

