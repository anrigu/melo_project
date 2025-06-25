import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


# -------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------

def extract_hp_from_dir(path: Path) -> Optional[int]:
    """Return the integer HP encoded as ``HP=<x>`` in *path* or None."""
    for part in path.parts:
        if part.startswith("HP="):
            try:
                return int(part.split("=")[1])
            except ValueError:
                return None
    return None


def parse_strategy_name(strategy: str) -> Tuple[str, str]:
    """Return (market, shade_label) extracted from *strategy*.

    The *strategy* should look like, e.g.::

        MOBI_0_100_shade0_0
        ZI_100_0_shade250_500

    The first number pair (100_0 or 0_100) encodes the market:
    * 100_0 → CDA
    * 0_100 → MELO

    Everything from "shade" onwards is returned as the *shade_label*.
    If the pattern is not recognised, the original string is returned
    as *market* and shade_label is "".
    """
    # Regex capt: role, cda_pct, melo_pct, rest (may include shade...)
    m = re.match(r"^(?:MOBI|ZI)_(\d+)_(\d+)_(.*)$", strategy)
    if not m:
        return strategy, ""
    cda_pct, melo_pct, rest = m.groups()
    market: str
    if cda_pct == "100" and melo_pct == "0":
        market = "CDA"
    elif cda_pct == "0" and melo_pct == "100":
        market = "MELO"
    else:
        market = f"{cda_pct}_{melo_pct}"
    shade_label = rest  # e.g. "shade250_500" (may include other stuff)
    return market, shade_label


# -------------------------------------------------------------
# Core aggregation logic
# -------------------------------------------------------------

def collect_surplus(
    root_dir: Path,
) -> Dict[str, Dict[str, Dict[int, List[float]]]]:
    """Return nested dict ``data[role][strategy][hp] -> list[payoff]``."""
    data: Dict[str, Dict[str, Dict[int, List[float]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    file_counter = 0  # DEBUG: count how many payoff files we load

    # Search recursively for raw_payoff_data.json files
    for payoff_path in root_dir.rglob("raw_payoff_data.json"):
        file_counter += 1
        hp = extract_hp_from_dir(payoff_path)
        if hp is None:
            # Fallback: try to parse trailing integer in the immediate run directory name
            try:
                hp = int(payoff_path.parent.name.split("_")[-1])
            except ValueError:
                continue

        try:
            with open(payoff_path) as f:
                profiles = json.load(f)
        except Exception as exc:
            print(f"⚠️  Could not load {payoff_path}: {exc}")
            continue

        # Each *profile* is a list of rows [pid, role, strategy, payoff]
        for profile in profiles:
            for _, role, strategy, payoff in profile:
                if role not in {"MOBI", "ZI"}:
                    continue
                data[role][strategy][hp].append(payoff)
    return data, file_counter



def make_line_plot(
    role: str,
    hp_vals: List[int],
    strategy_to_series: Dict[str, List[float]],
    out_dir: Path,
) -> None:
    plt.figure(figsize=(12, 6))

    color_cycle = plt.cm.tab20.colors  # up to 20 unique colours
    colors = iter(color_cycle * ((len(strategy_to_series) // len(color_cycle)) + 1))

    for strategy, y_vals in sorted(strategy_to_series.items()):
        plt.plot(
            hp_vals,
            y_vals,
            marker="o",
            label=strategy,
            color=next(colors),
        )

    plt.xlabel("Holding Period")
    plt.ylabel("Mean Surplus")
    plt.title(f"{role} strategy surplus vs holding period")
    plt.xticks(hp_vals)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    fname = out_dir / f"{role.lower()}_strategy_surplus_by_hp.png"
    plt.savefig(fname)
    print(f"Saved {fname}")


# -------------------------------------------------------------
# CLI
# -------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Plot strategy surplus vs holding period for MOBI and ZI roles.")
    parser.add_argument(
        "root_dir",
        type=Path,
        help="Directory like many_strategies containing HP=* subdirectories with raw_payoff_data.json files.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("analysis/plots"),
        help="Where to save the generated figures.",
    )
    args = parser.parse_args()

    data, file_counter = collect_surplus(args.root_dir)

    for role, strat_dict in data.items():
        if not strat_dict:
            print(f"No data for role {role}")
            continue

        # Deduce global sorted list of HPs for which we have any data
        hp_vals = sorted({hp for strat in strat_dict.values() for hp in strat})
        if not hp_vals:
            print(f"No HP values found for role {role}")
            continue

        # Build per-strategy y-series aligned with hp_vals
        strategy_to_series: Dict[str, List[float]] = {}
        for strategy, hp_map in strat_dict.items():
            series = [
                (np.mean(hp_map[hp]) if hp in hp_map else np.nan)
                for hp in hp_vals
            ]
            # Label strategy nicely: market + shade
            market, shade = parse_strategy_name(strategy)
            label = f"{market} {shade}".strip()
            strategy_to_series[label] = series

        make_line_plot(role, hp_vals, strategy_to_series, args.out_dir)

    print(f"Loaded {file_counter} payoff files")


if __name__ == "__main__":
    main() 