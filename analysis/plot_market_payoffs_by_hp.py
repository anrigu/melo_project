import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re
import itertools

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
    hp_max: int = 1000,
    group_by: str = "market",
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

    #--------------------------------------------------------------
    # The original layout expects ``root_dir`` to contain many
    # pilot_egta_run_* sub-dirs.  Newer experiment dumps instead
    # have *one* pilot dir whose children are ``holding_period_*``
    # directories.  We auto-detect which layout we are dealing with.
    #--------------------------------------------------------------

    potential_run_dirs = sorted(root_dir.glob("pilot_egta_run_*"))

    # If no pilot dirs are found treat the root itself as *the* run dir.
    if not potential_run_dirs:
        potential_run_dirs = [root_dir]

    for run_dir in potential_run_dirs:
        hp_dirs = list(run_dir.glob("holding_period_*"))

        found_any_hp_payoff = False
        if hp_dirs:
            for hp_dir in hp_dirs:
                m = re.search(r"holding_period_(\d+)", hp_dir.name)
                if m is None:
                    continue
                holding_period = int(m.group(1))

                if hp_max is not None and holding_period > hp_max:
                    continue

                payoff_files = list(hp_dir.glob("**/raw_payoff_data.json"))
                if not payoff_files:
                    continue

                found_any_hp_payoff = True

                for pf in payoff_files:
                    with open(pf) as f:
                        profiles = json.load(f)

                    for profile in profiles:
                        for row in profile:
                            role: str = row[1]
                            strategy: str = row[2]
                            payoff: float = row[3]

                            yield_item = strategy if group_by == "strategy" else classify_market(strategy)
                            if yield_item is None:
                                continue
                            data[role][holding_period][yield_item].append(payoff)

            
            continue

        
        try:
            holding_period = int(run_dir.name.split("_")[-1])
        except ValueError:
            continue

        if hp_max is not None and holding_period > hp_max:
            continue

        payoff_files = list(run_dir.glob("**/raw_payoff_data.json"))
        if not payoff_files:
            print(f"⚠️  No raw_payoff_data.json found under {run_dir}")
            continue

        for pf in payoff_files:
            with open(pf) as f:
                profiles = json.load(f)

            for profile in profiles:
                for row in profile:
                    role: str = row[1]
                    strategy: str = row[2]
                    payoff: float = row[3]

                    yield_item = strategy if group_by == "strategy" else classify_market(strategy)
                    if yield_item is None:
                        continue
                    data[role][holding_period][yield_item].append(payoff)

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
    means_by_cat: Dict[str, List[float]],
    q_max: int,
    out_dir: Path,
) -> None:
    """Plot grouped bar chart for an arbitrary number of categories (strategies or markets)."""

    n_cat = len(means_by_cat)
    if n_cat == 0:
        return

    plt.figure(figsize=(14, 5))
    x_pos = np.arange(len(hp_vals))
    width = 0.8 / n_cat

    colors = itertools.cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])

    for i, (cat, means) in enumerate(sorted(means_by_cat.items())):
        plt.bar(
            x_pos - 0.4 + width / 2 + i * width,
            means,
            width,
            label=cat,
            color=next(colors),
            capsize=3,
        )

    plt.xticks(x_pos, hp_vals)
    plt.xlabel("Holding Period")
    plt.ylabel("Mean Payoff")
    plt.title(f"{role} mean payoff")
    plt.legend(ncol=3, fontsize="small")
    plt.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "strategy" if len(means_by_cat) > 2 else "market"
    fname = out_dir / f"{role.lower()}_payoff_by_{suffix}_q{q_max}.png"
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
    parser.add_argument(
        "--hp-max",
        type=int,
        default=1000,
        help="Ignore runs with holding period above this value (None to disable).",
    )
    parser.add_argument(
        "--group-by",
        choices=["market", "strategy"],
        default="market",
        help="Aggregate by 'market' (CDA/MELO) or by full 'strategy' name.",
    )
    args = parser.parse_args()

    stat_data, ci_data = collect_payoffs(
        args.root_dir, args.stat, args.trim_pct, args.bootstrap, args.ci, args.hp_max, args.group_by
    )

    for role, hp_dict in stat_data.items():
        if not hp_dict:
            print(f"No data for role {role}")
            continue

        # Sort by holding period
        hp_vals = sorted(hp_dict.keys())

        categories = sorted({cat for hp in hp_vals for cat in hp_dict[hp].keys()})
        means_by_cat = {cat: [hp_dict[hp].get(cat, np.nan) for hp in hp_vals] for cat in categories}

        make_bar_plot(
            role,
            hp_vals,
            means_by_cat,
            args.q_max,
            args.out_dir,
        )


if __name__ == "__main__":
    main() 