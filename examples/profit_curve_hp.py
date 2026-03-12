import argparse
import os
import random
import numpy as np
import torch
import json
import matplotlib.pyplot as plt
from typing import Dict, List

from marketsim.egta.simulators.melo_wrapper import MeloSimulator


def build_profile(mobi_mix: Dict[str, float], zi_mix: Dict[str, float],
                  num_mobi: int, num_zi: int) -> List[tuple]:
    """Convert role-level mixture dictionaries into a full pure profile list."""
    profile: List[tuple] = []

    # MOBI role
    remaining = num_mobi
    for strat, prob in mobi_mix.items():
        cnt = int(round(prob * num_mobi))
        remaining -= cnt
        profile.extend([("MOBI", strat)] * cnt)
    # assign any rounding residual to the first strategy
    if remaining > 0 and mobi_mix:
        first = next(iter(mobi_mix))
        profile.extend([("MOBI", first)] * remaining)

    # ZI role
    remaining = num_zi
    for strat, prob in zi_mix.items():
        cnt = int(round(prob * num_zi))
        remaining -= cnt
        profile.extend([("ZI", strat)] * cnt)
    if remaining > 0 and zi_mix:
        first = next(iter(zi_mix))
        profile.extend([("ZI", first)] * remaining)

    return profile


def profit_vs_holding_period(hp_values: List[int],
                             mobi_mix: Dict[str, float],
                             zi_mix:   Dict[str, float],
                             reps: int = 500,
                             base_seed: int = 123,
                             num_mobi: int = 28,
                             num_zi: int = 40,
                             sim_time: int = 10_000) -> Dict[int, Dict[str, float]]:
    """Return average payoffs per role strategy across HP grid (deterministic seeds)."""
    out: Dict[int, Dict[str, float]] = {}

    for hp in hp_values:
        # reset global RNGs so each HP sees identical random seeds list
        random.seed(base_seed)
        np.random.seed(base_seed & 0xFFFF_FFFF)
        torch.manual_seed(base_seed)

        simulator = MeloSimulator(
            num_strategic_mobi=num_mobi,
            num_strategic_zi=num_zi,
            sim_time=sim_time,
            lam=6e-3,
            lam_r=6e-3,
            lam_melo=1e-3,
            lam_melo_mobi=1e-3,
            lam_melo_zi=6e-3,
            mean=1e6,
            r=0.001,
            shock_var=1e6,
            q_max=10,
            pv_var=5_000_000,
            shade=[250, 500],
            holding_period=hp,
            num_background_zi=0,
            num_background_hbl=0,
            reps=reps,
            mobi_strategies=list(mobi_mix.keys()),
            zi_strategies=list(zi_mix.keys()),
            log_profile_details=False,
            parallel=True,
        )

        profile = build_profile(mobi_mix, zi_mix, num_mobi, num_zi)
        obs = simulator.simulate_profile(profile)
        payoffs = obs.payoffs

        # payoffs list is [MOBI players first, then ZI players]
        mobi_avg = float(payoffs[:num_mobi].mean())
        zi_avg   = float(payoffs[num_mobi:].mean())
        out[hp] = {"MOBI": mobi_avg, "ZI": zi_avg}
        print(f"HP={hp} → MOBI {mobi_avg:.2f} | ZI {zi_avg:.2f}")
    return out


def plot_curve(res: Dict[int, Dict[str, float]], out_png: str = None):
    x = sorted(res.keys())
    mobi = [res[hp]["MOBI"] for hp in x]
    zi   = [res[hp]["ZI"]   for hp in x]

    plt.figure(figsize=(6,4))
    plt.plot(x, mobi, "o-", label="MOBI")
    plt.plot(x, zi, "s--", label="ZI")
    plt.xlabel("Holding period")
    plt.ylabel("Mean payoff")
    plt.title("Profit vs holding-period (deterministic seeds)")
    plt.legend()
    plt.tight_layout()
    if out_png:
        plt.savefig(out_png)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deterministic profit curve vs holding period")
    parser.add_argument("--hp", type=int, nargs="*", default=[0,50,100,200,400],
                        help="Space-separated list of holding periods")
    parser.add_argument("--reps", type=int, default=500, help="Repetitions per HP")
    parser.add_argument("--out", type=str, default=None, help="Optional PNG output path")
    args = parser.parse_args()

    # opponent mix: copied from current equilibrium intuition (all CDA initially)
    mobi_mix = {
        "MOBI_0_100_shade0_0": 0.5,
        "MOBI_100_0_shade0_250": 0.5,
    }
    zi_mix = {
        "ZI_0_100_shade250_500": 0.5,
        "ZI_100_0_shade250_500": 0.5,
    }

    res = profit_vs_holding_period(args.hp, mobi_mix, zi_mix, reps=args.reps)

    plot_curve(res, args.out) 