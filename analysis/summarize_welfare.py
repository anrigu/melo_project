#!/usr/bin/env python3
import os
import sys
import json
import torch
import argparse

# make sure we can import marketsim
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from marketsim.game.role_symmetric_game import RoleSymmetricGame

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute raw‐unit welfare & regrets for stored equilibria."
    )
    parser.add_argument(
        "results_dir",
        type=str,
        help="Path to the results folder (must contain experiment_parameters.json, equilibria_detailed.json, raw_payoff_data.json)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device to use (cpu or cuda)"
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="If set, normalize payoffs (subtract mean, divide by std)"
    )
    return parser.parse_args()

def _compute_role_stats(mixture: torch.Tensor,
                        avg_by_role: dict[str, dict[str, float]],
                        game: RoleSymmetricGame):
    """Expected payoffs/regrets using *per-strategy sample means*.

    Parameters
    ----------
    mixture        Flat tensor (len == game.num_strategies) already on device.
    avg_by_role    {role: {strategy: mean_raw_payoff}}
    game           The RoleSymmetricGame (only used for num_players_per_role).
    """

    role_payoffs: dict[str, float] = {}
    role_regrets: dict[str, float] = {}
    collective = 0.0

    offset = 0
    for r_idx, role in enumerate(game.role_names):
        n_s = int(game.num_strategies_per_role[r_idx])
        r_mix = mixture[offset:offset + n_s]
        strat_names = game.strategy_names_per_role[r_idx]

        # Expected payoff under mixture
        epay = 0.0
        for p, s in zip(r_mix.tolist(), strat_names):
            epay += p * avg_by_role.get(role, {}).get(s, 0.0)

        # Best-response payoff ≈ best average sample payoff for role
        br = max(avg_by_role.get(role, {}).values()) if avg_by_role.get(role) else 0.0

        role_payoffs[role] = epay
        role_regrets[role] = br - epay

        collective += epay * int(game.num_players_per_role[r_idx])

        offset += n_s

    return {
        "role_expected_payoffs": role_payoffs,
        "role_regrets": role_regrets,
        "collective_welfare": collective,
    }

if __name__ == "__main__":
    args = parse_args()
    rd = args.results_dir

    # Load inputs
    params_fp = os.path.join(rd, "experiment_parameters.json")
    eq_fp = os.path.join(rd, "equilibria_detailed.json")
    raw_fp = os.path.join(rd, "raw_payoff_data.json")
    for p in (params_fp, eq_fp, raw_fp):
        if not os.path.isfile(p):
            print(f"ERROR: missing required file: {p}", file=sys.stderr)
            sys.exit(1)

    with open(params_fp) as f:
        params = json.load(f)
    with open(eq_fp) as f:
        equilibria_info = json.load(f)
    with open(raw_fp) as f:
        payoff_data = json.load(f)

    # Unpack experiment settings
    role_names = params["role_names"]
    num_players_per_role = params["num_players_per_role"]
    strategy_names_per_role = params["strategy_names_per_role"]

    # Build the symmetric game from raw payoff data
    game_raw = RoleSymmetricGame.from_payoff_data_rsg(
        payoff_data=payoff_data,
        role_names=role_names,
        num_players_per_role=num_players_per_role,
        strategy_names_per_role=strategy_names_per_role,
        device=args.device,
        normalize_payoffs=args.normalize,
    )

    # ------------------------------------------------------------------
    # 1. Compute simple per-strategy sample means from payoff_data  (raw units)
    # ------------------------------------------------------------------

    avg_by_role: dict[str, dict[str, float]] = {}
    count_by_role: dict[str, dict[str, int]] = {}

    for prof in payoff_data:  # each profile is list[(pid, role, strat, payoff)]
        for _pid, r_name, s_name, payoff in prof:
            if payoff is None or not isinstance(payoff, (int, float)) or torch.isnan(torch.tensor(payoff)) or torch.isinf(torch.tensor(payoff)):
                continue
            avg_by_role.setdefault(r_name, {}).setdefault(s_name, 0.0)
            count_by_role.setdefault(r_name, {}).setdefault(s_name, 0)
            avg_by_role[r_name][s_name] += float(payoff)
            count_by_role[r_name][s_name] += 1

    # convert sums to means
    for r_name, smap in avg_by_role.items():
        for s_name in smap:
            cnt = count_by_role[r_name][s_name]
            if cnt:
                smap[s_name] /= cnt

    # ------------------------------------------------------------------
    # Optional debug print
    print("Constructed game:")
    print(" Configs:", game_raw.rsg_config_table.shape)
    print(" Payoffs:", game_raw.rsg_payoff_table.shape)

    # Compute and collect stats for each equilibrium
    results = []
    for eq in equilibria_info:
        # Rebuild mixture-tensor by name to ensure correct ordering
        mix = torch.zeros(game_raw.num_strategies, dtype=torch.float32, device=args.device)
        for r_idx, role in enumerate(role_names):
            role_dist = eq["mixture_by_role"][role]
            for s_local_idx, strat in enumerate(strategy_names_per_role[r_idx]):
                gi = int(game_raw.role_starts[r_idx] + s_local_idx)
                mix[gi] = role_dist.get(strat, 0.0)

        stats = _compute_role_stats(mix, avg_by_role, game_raw)
        results.append({
            "equilibrium_id": eq["equilibrium_id"],
            "mixture_by_role": eq["mixture_by_role"],
            **stats
        })

    # Write out
    out_fp = os.path.join(rd, "raw_welfare_metrics.json")
    with open(out_fp, "w") as f:
        json.dump(results, f, indent=2)

    print(f"✓ Wrote raw welfare metrics to {out_fp}")
