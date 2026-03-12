#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Iterable, Sequence

from marketsim.egta.simulators.melo_wrapper import MeloSimulator

# Canonical strategy ordering (matches plot_bootstrap_compare.py)
CANONICAL_ORDER: Sequence[tuple[str, str]] = (
    ("MOBI", "MOBI_0_100_shade0_0"),
    ("MOBI", "MOBI_100_0_shade250_500"),
    ("ZI",   "ZI_0_100_shade250_500"),
    ("ZI",   "ZI_100_0_shade250_500"),
)

DEFAULT_PROFILES: Sequence[tuple[int, int, int, int]] = (
    (28, 0, 0, 40),
    (28, 0, 1, 39)
)


"""
(28, 0, 1, 39),
    (28, 0, 2, 38),
    (28, 0, 3, 37),
    (28, 0, 4, 36),
    (28, 0, 5, 35),
    (28, 0, 6, 34),
    (28, 0, 7, 33),
    (28, 0, 8, 32),
    (28, 0, 9, 31),
    (28, 0, 10, 30),
    (28, 0, 11, 29),
    (28, 0, 12, 28),
    (28, 0, 13, 27),
    (28, 0, 14, 26),
    (28, 0, 15, 25),
    (28, 0, 16, 24),
    (28, 0, 17, 23),
    (28, 0, 18, 22),
    (28, 0, 19, 21),
    (28, 0, 20, 20),
    (28, 0, 21, 19),
    (28, 0, 22, 18),
    (28, 0, 23, 17),
    (28, 0, 24, 16),
    (28, 0, 25, 15),
    (28, 0, 26, 14),
    (28, 0, 27, 13),
    (28, 0, 28, 12),
    (28, 0, 29, 11),
    (28, 0, 30, 10),
    (28, 0, 31, 9),
    (28, 0, 32, 8),
    (28, 0, 33, 7),
    (28, 0, 34, 6),
    (28, 0, 35, 5),
    (28, 0, 36, 4),
    (28, 0, 37, 3),
    (28, 0, 38, 2),
    (28, 0, 39, 1),
    (28, 0, 40, 0),
"""
def counts_to_profile(counts: Iterable[int]) -> list[tuple[str, str]]:
    """Expand (mobi_me, mobi_cda, zi_me, zi_cda) counts into a profile list."""
    profile: list[tuple[str, str]] = []
    for (role, strat), n in zip(CANONICAL_ORDER, counts):
        profile.extend([(role, strat)] * int(n))
    return profile

def simulate_profiles(holding_period: int,
                      profiles: Sequence[tuple[int, int, int, int]],
                      *,
                      reps: int,
                      parallel: bool,
                      num_workers: int) -> list[list[tuple[str, str, str, float]]]:
    """Run the MELO simulator for each count tuple and return payoff rows."""
    simulator = MeloSimulator(
        num_strategic_mobi=28,
        num_strategic_zi=40,
        sim_time=10000,
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
        holding_period=holding_period,
        mobi_strategies=[s for _, s in CANONICAL_ORDER[:2]],
        zi_strategies=[s for _, s in CANONICAL_ORDER[2:]],
        reps=reps,
        num_workers=num_workers,
        parallel=parallel,
        log_profile_details=True,
    )

    results: list[list[tuple[str, str, str, float]]] = []
    for counts in profiles:
        profile = counts_to_profile(counts)
        observation = simulator.simulate_profile(profile)
        rows: list[tuple[str, str, str, float]] = []
        for idx, ((role, strat), payoff) in enumerate(zip(profile, observation.payoffs)):
            rows.append((f"p{idx}", role, strat, float(payoff)))
        results.append(rows)
    return results

def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate the default DPR profiles.")
    parser.add_argument("--holding-period", type=int, required=True,
                        help="Holding period to use when running the simulator.")
    parser.add_argument("--output", type=Path,
                        default=Path("dpr_default_profiles_raw_payoff_data.json"),
                        help="Destination JSON file (same structure as raw_payoff_data.json).")
    parser.add_argument("--reps", type=int, default=5000,
                        help="Number of simulation repetitions per profile.")
    parser.add_argument("--num-workers", type=int, default=8,
                        help="Worker processes/threads if --parallel is set.")
    parser.add_argument("--parallel", action="store_true",
                        help="Enable parallel execution of simulator repetitions.")
    args = parser.parse_args()

    results = simulate_profiles(
        args.holding_period,
        DEFAULT_PROFILES,
        reps=args.reps,
        parallel=args.parallel,
        num_workers=args.num_workers,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as fh:
        json.dump(results, fh, indent=2)
    print(f"Wrote {len(results)} profiles to {args.output}")

if __name__ == "__main__":
    main()
