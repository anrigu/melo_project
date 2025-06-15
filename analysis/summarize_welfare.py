#!/usr/bin/env python3
"""
Summarise welfare & regrets for stored equilibria (raw payoff units, NaN-robust).

Expected directory structure of <results_dir>:

  experiment_parameters.json
  equilibria_detailed.json
  raw_payoff_data.json

Author: 2025-06-15
"""
import os, sys, json, argparse, torch
from collections import defaultdict
from typing import Dict, Tuple

# --------------------------------------------------------------------- #
#  Project import path (so that `marketsim` is importable)
# --------------------------------------------------------------------- #
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from marketsim.game.role_symmetric_game import RoleSymmetricGame


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Compute raw-unit welfare & regrets (NaN-robust).")
    ap.add_argument("results_dir", type=str,
                    help="Folder containing experiment_parameters.json, "
                         "equilibria_detailed.json, raw_payoff_data.json")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--normalize", action="store_true",
                    help="Normalise payoffs when building the game "
                         "(subtract global mean, divide by std).")
    return ap.parse_args()


def safe_nanmax(x: torch.Tensor) -> torch.Tensor:
    """torch.nanmax replacement that works on all PyTorch versions."""
    if torch.isnan(x).all():
        return torch.tensor(float('-inf'), dtype=x.dtype, device=x.device)
    return torch.max(torch.nan_to_num(x, nan=float('-inf')))


def build_strategy_sample_means(
    payoff_data,
    role_names,
    strategy_names_per_role
) -> Dict[str, Dict[str, Tuple[float, int]]]:
    """
    Returns a dict:
      means[role][strategy] → (mean_raw_payoff, sample_count)
    Ensures every role/strategy listed in the experiment parameters
    appears (default mean=0, count=0) so look-ups never KeyError.
    """
    sums   = defaultdict(lambda: defaultdict(float))
    counts = defaultdict(lambda: defaultdict(int))

    for profile in payoff_data:       
        for _pid, role, strat, val in profile:
            if val is None or not isinstance(val, (int, float)):
                continue
            t = torch.tensor(val)
            if torch.isnan(t) or torch.isinf(t):
                continue
            sums  [role][strat] += float(val)
            counts[role][strat] += 1

    means = defaultdict(dict)
    for role in role_names:
        for strat in strategy_names_per_role[role]:
            n = counts [role].get(strat, 0)
            s = sums   [role].get(strat, 0.0)
            means[role][strat] = (s / n if n else 0.0, n)

    return means 


def role_stats(game: RoleSymmetricGame,
               mixture: torch.Tensor,
               strat_means_raw: Dict[str, Dict[str, Tuple[float, int]]]):
    """
    Expected payoffs / regrets / welfare for one mixture.
    Any missing deviation payoff → replaced by the *sample mean* for that
    rôle/strategy (falls back to 0 if there were no samples).
    """
    dev_raw = game.deviation_payoffs(mixture, ignore_incomplete=True)
    dev_raw = dev_raw * game.scale + game.offset      

  
    filled = dev_raw.clone()
    for g_idx in range(game.num_strategies):
        if torch.isnan(filled[g_idx]) or torch.isinf(filled[g_idx]):
            r_idx, s_local = game.get_role_and_strategy_index(g_idx)
            role  = game.role_names[r_idx]
            strat = game.strategy_names_per_role[r_idx][s_local]
            filled[g_idx] = strat_means_raw[role][strat][0]  # mean (may be 0)
    filled = torch.nan_to_num(filled, nan=0.0, posinf=0.0, neginf=0.0)
 
    role_pay, role_reg, welfare = {}, {}, 0.0
    off = 0
    for r_idx, role in enumerate(game.role_names):
        n_s   = int(game.num_strategies_per_role[r_idx])
        probs = mixture[off:off + n_s]
        dev_r = filled [off:off + n_s]

        exp_pay = (probs * dev_r).sum().item()
        br_pay  = safe_nanmax(dev_r). item()
        if br_pay == float('-inf'):                 

        role_pay[role] = exp_pay
        role_reg[role] = br_pay - exp_pay
        welfare        += exp_pay * game.num_players_per_role[r_idx].item()
        off            += n_s

    return role_pay, role_reg, welfare

# --------------------------------------------------------------------- #
#  main
# --------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()
    rd   = args.results_dir

    # ---------- load inputs ------------------------------------------- #
    with open(os.path.join(rd, "experiment_parameters.json")) as f:
        params = json.load(f)
    with open(os.path.join(rd, "equilibria_detailed.json")) as f:
        equilibria = json.load(f)
    with open(os.path.join(rd, "raw_payoff_data.json")) as f:
        payoff_data = json.load(f)

    role_names  = params["role_names"]
    strat_names = {
        r: s_list for r, s_list in zip(role_names, params["strategy_names_per_role"])
    }

    # ---------- build game -------------------------------------------- #
    game = RoleSymmetricGame.from_payoff_data_rsg(
        payoff_data             = payoff_data,
        role_names              = role_names,
        num_players_per_role    = params["num_players_per_role"],
        strategy_names_per_role = params["strategy_names_per_role"],
        device                  = args.device,
        normalize_payoffs       = args.normalize,
    )

    # ---------- sample means (raw units) ------------------------------ #
    means_raw = build_strategy_sample_means(payoff_data, role_names, strat_names)

    # ---------- iterate over equilibria ------------------------------- #
    results = []
    for eq in equilibria:
        mix = torch.zeros(game.num_strategies, device=args.device)
        for r_idx, role in enumerate(role_names):
            for s_idx, strat in enumerate(params["strategy_names_per_role"][r_idx]):
                gi = game.role_starts[r_idx] + s_idx
                mix[gi] = eq["mixture_by_role"][role].get(strat, 0.0)

        role_pay, role_reg, welfare = role_stats(game, mix, means_raw)
        results.append({
            "equilibrium_id":        eq["equilibrium_id"],
            "mixture_by_role":       eq["mixture_by_role"],
            "role_expected_payoffs": role_pay,
            "role_regrets":          role_reg,
            "collective_welfare":    welfare,
        })

    # ---------- write -------------------------------------------------- #
    out_fp = os.path.join(rd, "raw_welfare_metrics.json")
    with open(out_fp, "w") as f:
        json.dump(results, f, indent=2)
    print(f"✓ Wrote raw welfare metrics to {out_fp}")


if __name__ == "__main__":
    main()
