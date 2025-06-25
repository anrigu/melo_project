import os
import re
import json
from collections import defaultdict
from typing import Dict, List

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
# Root directory that contains the result sub-folders produced by the EGTA runs
RESULTS_ROOT = os.path.join(os.path.dirname(__file__), os.pardir, "3_strat_results", "3_strategies_collected")

# Roles to aggregate.  Each role gets its own table.
ROLES_TO_AGGREGATE = ["MOBI", "ZI"]  # add or remove roles as needed

# Output file name template (role will be injected)
OUTPUT_CSV_TEMPLATE = os.path.join(RESULTS_ROOT, "mean_payoff_table_{role}_q10.csv")
OUTPUT_MD_TEMPLATE = os.path.join(RESULTS_ROOT, "mean_payoff_table_{role}_q10.md")


def find_raw_payoff_files(root: str) -> Dict[int, str]:
    """Recursively search *root* for raw_payoff_data.json files.

    The function returns a mapping *holding_period -> path_to_json*.
    If multiple files are found for the same holding period the first one
    encountered is kept and a warning is printed.
    """
    pattern = re.compile(r"holding_period_(\d+)")
    files: Dict[int, str] = {}

    for dirpath, _dirnames, filenames in os.walk(root):
        if "raw_payoff_data.json" in filenames:
            hp_match = pattern.search(dirpath)
            if not hp_match:
                # Could not infer holding period, skip.
                continue
            holding_period = int(hp_match.group(1))
            json_path = os.path.join(dirpath, "raw_payoff_data.json")
            if holding_period not in files:
                files[holding_period] = json_path
            else:
                print(
                    f"[WARN] Duplicate raw_payoff_data for holding_period={holding_period}.\n"
                    f"       Keeping: {files[holding_period]}\n       Skipping: {json_path}"
                )
    return files


def load_mean_payoffs(json_path: str, role_filter: str | None = None) -> Dict[str, float]:
    """Compute mean payoff per strategy from *json_path*.

    Parameters
    ----------
    json_path : str
        Path to a raw_payoff_data.json file.
    role_filter : str | None, optional
        If provided, only records whose *role* field matches `role_filter` are
        considered. The *role* is the second element of each inner list entry.
    """
    with open(json_path, "r", encoding="utf-8") as fh:
        data: List[List[List]] = json.load(fh)

    payoff_by_strategy: Dict[str, List[float]] = defaultdict(list)

    for replicate in data:
        for player_record in replicate:
            _, role, strategy, payoff = player_record
            if role_filter is not None and role != role_filter:
                continue
            payoff_by_strategy[strategy].append(payoff)

    return {s: float(np.mean(v)) for s, v in payoff_by_strategy.items() if v}


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    if not os.path.isdir(RESULTS_ROOT):
        raise FileNotFoundError(
            f"Results directory not found: {RESULTS_ROOT}. "
            "Please check the RESULTS_ROOT path."
        )

    print(f"Scanning for raw_payoff_data.json under {RESULTS_ROOT} ...")
    payoff_files = find_raw_payoff_files(RESULTS_ROOT)
    if not payoff_files:
        raise RuntimeError("No raw_payoff_data.json files found. Check directory structure.")

    print(f"Found {len(payoff_files)} raw payoff files.")

    for role in ROLES_TO_AGGREGATE:
        # Collect mean payoffs per holding period for *role*
        records: List[pd.Series] = []
        for holding_period, json_path in sorted(payoff_files.items()):
            means = load_mean_payoffs(json_path, role_filter=role)
            ser = pd.Series(means, name=holding_period)
            records.append(ser)
            print(f"  [{role}] HP={holding_period:>4}: strategies={len(ser)}")

        df = pd.DataFrame(records).sort_index()
        df.index.name = "holding_period"

        # Persist to CSV and Markdown
        output_csv = OUTPUT_CSV_TEMPLATE.format(role=role)
        output_md = OUTPUT_MD_TEMPLATE.format(role=role)
        df.to_csv(output_csv)
        with open(output_md, "w", encoding="utf-8") as fh:
            fh.write(df.to_markdown())

        print(f"\nMean payoff table for {role} created:")
        print(df.head())
        print(f"Saved CSV to: {output_csv}\nSaved Markdown table to: {output_md}\n")


if __name__ == "__main__":  # pragma: no cover
    main() 