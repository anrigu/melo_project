import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np

# -------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------

def classify_market(strategy_name: str) -> str | None:
    """Return 'CDA' or 'MELO' based on the _100_0 / _0_100 pattern.

    The convention is:
        *_100_0*   → CDA
        *_0_100*   → MELO
    Any other mix is ignored (returns None).
    """
    parts = strategy_name.split("_")
    if len(parts) < 3:
        return None
    first, second = parts[1], parts[2]
    if first == "100" and second == "0":
        return "CDA"
    if first == "0" and second == "100":
        return "MELO"
    return None


def mean(values: List[float]) -> float | None:
    return float(np.mean(values)) if values else None


# -------------------------------------------------------------
# Statistic helpers
# -------------------------------------------------------------

def compute_stat(values: List[float], metric: str = "mean", trim_pct: float = 10.0) -> float | None:
    """Compute statistic of *values* according to *metric*.

    Parameters
    ----------
    values : list[float]
        Sample of payoffs.
    metric : str
        One of {"mean", "median", "trimmed_mean"}.
    trim_pct : float
        For trimmed_mean, percentage to trim from *each* tail (0–50).
    """
    if not values:
        return None

    arr = np.asarray(values)
    if metric == "mean":
        return float(arr.mean())
    if metric == "median":
        return float(np.median(arr))
    if metric == "trimmed_mean":
        if trim_pct <= 0:
            return float(arr.mean())
        trim_frac = trim_pct / 100.0
        k = int(len(arr) * trim_frac)
        if k == 0:
            return float(arr.mean())
        arr_sorted = np.sort(arr)
        trimmed = arr_sorted[k:-k] if k < len(arr_sorted) // 2 else arr_sorted
        return float(trimmed.mean())
    raise ValueError(f"Unknown metric: {metric}")


# -------------------------------------------------------------
# Bootstrap helpers
# -------------------------------------------------------------

def bootstrap_ci(
    values: List[float],
    metric: str,
    trim_pct: float,
    n_resamples: int = 0,
    ci: float = 95.0,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, Optional[Tuple[float, float]]]:
    """Return (stat, (low, high)) where low/high bound the percentile CI.

    If *n_resamples* is 0, the CI is None.
    """
    stat = compute_stat(values, metric, trim_pct)
    if n_resamples <= 0 or not values:
        return stat, None

    rng = rng or np.random.default_rng()
    resampled_stats = []
    arr = np.asarray(values)
    n = len(arr)
    for _ in range(n_resamples):
        sample = rng.choice(arr, size=n, replace=True)
        resampled_stats.append(compute_stat(sample.tolist(), metric, trim_pct))

    lower_q = (100.0 - ci) / 2.0
    upper_q = 100.0 - lower_q
    low = np.percentile(resampled_stats, lower_q)
    high = np.percentile(resampled_stats, upper_q)
    return stat, (low, high)


# -------------------------------------------------------------
# Core aggregation logic
# -------------------------------------------------------------

def collect_payoffs(
    root_dir: Path,
    metric: str = "mean",
    trim_pct: float = 10.0,
    bootstrap: int = 0,
    ci: float = 95.0,
) -> Tuple[
    Dict[str, Dict[int, Dict[str, float]]],  # statistics
    Dict[str, Dict[int, Dict[str, Tuple[float, float]]]],  # CI bounds (may be None)
]:
    """Aggregate mean payoffs by (role → holding_period → market).

    Returns
    -------
    data : dict
        {role: {holding_period: {market: mean_payoff}}}
    """
    data: Dict[str, Dict[int, Dict[str, List[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )

    for run_dir in sorted(root_dir.glob("pilot_egta_run_*")):
        try:
            holding_period = int(run_dir.name.split("_")[-1])
        except ValueError:
            continue

        payoff_files = list(run_dir.glob("**/raw_payoff_data.json"))
        if not payoff_files:
            print(f"⚠️  No raw_payoff_data.json found under {run_dir}")
            continue

        payoff_path = payoff_files[0]
        with open(payoff_path) as f:
            profiles = json.load(f)

        # Iterate over every (player) row in every profile
        for profile in profiles:
            for row in profile:
                # Format: [player_id, role, strategy_name, payoff]
                role: str = row[1]
                strategy: str = row[2]
                payoff: float = row[3]

                market = classify_market(strategy)
                if market is None:
                    # Ignore strategies that are neither pure-CDA nor pure-MELO
                    continue

                data[role][holding_period][market].append(payoff)

    stat_data: Dict[str, Dict[int, Dict[str, float]]] = defaultdict(dict)
    ci_data: Dict[str, Dict[int, Dict[str, Tuple[float, float]]]] = defaultdict(dict)
    for role, hp_dict in data.items():
        for hp, market_dict in hp_dict.items():
            stat_row = {}
            ci_row = {}
            for market, vals in market_dict.items():
                stat, bounds = bootstrap_ci(vals, metric, trim_pct, bootstrap, ci)
                stat_row[market] = stat
                if bounds is not None:
                    ci_row[market] = bounds
            stat_data[role][hp] = stat_row
            if ci_row:
                ci_data[role][hp] = ci_row

    return stat_data, ci_data


# -------------------------------------------------------------
# Plotting helper
# -------------------------------------------------------------

def make_bar_plot(
    role: str,
    hp_vals: List[int],
    means_cda: List[float],
    means_melo: List[float],
    q_max: int,
    out_dir: Path,
    cda_err: Optional[np.ndarray] = None,
    melo_err: Optional[np.ndarray] = None,
) -> None:
    plt.figure(figsize=(12, 5))
    x_pos = np.arange(len(hp_vals))
    width = 0.35

    plt.bar(
        x_pos - width / 2,
        means_cda,
        width,
        label="CDA (100_0)",
        color="steelblue",
        yerr=cda_err,
        capsize=4,
    )
    plt.bar(
        x_pos + width / 2,
        means_melo,
        width,
        label="MELO (0_100)",
        color="darkorange",
        yerr=melo_err,
        capsize=4,
    )

    plt.xticks(x_pos, hp_vals)
    plt.xlabel("Holding Period")
    plt.ylabel("Mean Payoff")
    plt.title(f"{role} mean payoff per market vs holding period (q_max={q_max})")
    plt.legend()
    plt.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    fname = out_dir / f"{role.lower()}_payoff_cda_vs_melo_q{q_max}.png"
    plt.savefig(fname)
    print(f"Saved {fname}")


# -------------------------------------------------------------
# Main CLI entry-point
# -------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Plot mean payoffs for CDA vs MELO across holding periods, for each role."
    )
    parser.add_argument(
        "root_dir",
        type=Path,
        help="Directory like 2_strat_results/2_strat_q_max_10 containing pilot_egta_run_* subdirs",
    )
    parser.add_argument(
        "--q-max",
        type=int,
        default=10,
        help="Value of q_max (only used for labelling the plot titles).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("analysis/plots"),
        help="Where to save the generated PNG figures.",
    )
    parser.add_argument(
        "--stat",
        choices=["mean", "median", "trimmed_mean"],
        default="mean",
        help="Statistic to plot across profiles.",
    )
    parser.add_argument(
        "--trim-pct",
        type=float,
        default=10.0,
        help="Percentage to trim from each tail when using trimmed_mean.",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=0,
        help="Number of bootstrap resamples for CI (0 disables).",
    )
    parser.add_argument(
        "--ci",
        type=float,
        default=95.0,
        help="Confidence level (in percent) for bootstrap intervals.",
    )
    args = parser.parse_args()

    stat_data, ci_data = collect_payoffs(
        args.root_dir, args.stat, args.trim_pct, args.bootstrap, args.ci
    )

    for role, hp_dict in stat_data.items():
        if not hp_dict:
            print(f"No data for role {role}")
            continue

        # Sort by holding period
        hp_vals = sorted(hp_dict.keys())
        means_cda = [hp_dict[hp].get("CDA", np.nan) for hp in hp_vals]
        means_melo = [hp_dict[hp].get("MELO", np.nan) for hp in hp_vals]

        if args.bootstrap > 0:
            # Build asymmetric yerr arrays (shape 2×N)
            cda_err = np.array(
                [
                    [
                        means_cda[i] - ci_data.get(role, {}).get(hp_vals[i], {}).get("CDA", (np.nan, np.nan))[0]
                        if "CDA" in ci_data.get(role, {}).get(hp_vals[i], {})
                        else np.nan,
                        ci_data.get(role, {}).get(hp_vals[i], {}).get("CDA", (np.nan, np.nan))[1]
                        - means_cda[i]
                        if "CDA" in ci_data.get(role, {}).get(hp_vals[i], {})
                        else np.nan,
                    ]
                    for i in range(len(hp_vals))
                ]
            ).T  # shape (2,N)

            melo_err = np.array(
                [
                    [
                        means_melo[i] - ci_data.get(role, {}).get(hp_vals[i], {}).get("MELO", (np.nan, np.nan))[0]
                        if "MELO" in ci_data.get(role, {}).get(hp_vals[i], {})
                        else np.nan,
                        ci_data.get(role, {}).get(hp_vals[i], {}).get("MELO", (np.nan, np.nan))[1]
                        - means_melo[i]
                        if "MELO" in ci_data.get(role, {}).get(hp_vals[i], {})
                        else np.nan,
                    ]
                    for i in range(len(hp_vals))
                ]
            ).T
        else:
            cda_err = melo_err = None

        make_bar_plot(
            role,
            hp_vals,
            means_cda,
            means_melo,
            args.q_max,
            args.out_dir,
            cda_err,
            melo_err,
        )


if __name__ == "__main__":
    main() 