#!/usr/bin/env python3
"""
filter_equilibria_e4.py  –  recursively scan one or more result directories
and keep only equilibria whose regret ≤ 1e-4 ("e-4 range").

For every     .../equilibria*.json    found it writes
    .../equilibria*_e4.json
next to the original file containing only the filtered entries.

Additionally it prints a per-holding-period summary:
    HP  count_kept / total_in_original
so you can quickly see how many near-zero-regret equilibria exist.

Usage
-----
    python scripts/filter_equilibria_e4.py DIR [DIR ...]

Example (the five seeds requested):
    python scripts/filter_equilibria_e4.py \
        result_two_role_still_role_symmetric_3/ZI_arrival_6e-3_longer_sim_time_order_quantity_542 \
        result_two_role_still_role_symmetric_4/ZI_arrival_6e-3_longer_sim_time_order_quantity_543 \
        result_two_role_still_role_symmetric_5/ZI_arrival_6e-3_longer_sim_time_order_quantity_544 \
        result_two_role_still_role_symmetric_6/ZI_arrival_6e-3_longer_sim_time_order_quantity_545 \
        result_two_role_still_role_symmetric_7/ZI_arrival_6e-3_longer_sim_time_order_quantity_546

The script is read-only w.r.t. the original JSON files.
"""
from __future__ import annotations
import argparse, json, re, sys, pathlib, itertools
from collections import defaultdict

THRESH = 5e-4  # regret threshold

HP_RE = re.compile(r"holding_period_(\d+)")


def scan_and_filter(base_dir: pathlib.Path):
    summary = defaultdict(lambda: [0, 0])  # hp -> [kept, total]

    for p in base_dir.rglob("equilibria*.json"):
        try:
            data = json.loads(p.read_text())
        except Exception:
            continue  # skip malformed or empty files
        if not isinstance(data, list):
            continue

        hp_match = HP_RE.search(str(p))
        hp_val = int(hp_match.group(1)) if hp_match else -1

        kept = [e for e in data if float(e.get("regret", 1.0)) <= THRESH]
        summary[hp_val][0] += len(kept)
        summary[hp_val][1] += len(data)

        # write filtered file next to original if anything kept
        if kept:
            out_path = p.with_name(p.stem + f"_e4.json")
            out_path.write_text(json.dumps(kept, indent=2))
    return summary


def merge_summaries(summaries):
    merged = defaultdict(lambda: [0, 0])
    for summ in summaries:
        for hp, (k, t) in summ.items():
            merged[hp][0] += k; merged[hp][1] += t
    return merged


def main(argv=None):
    ap = argparse.ArgumentParser(description="Filter equilibria with regret ≤ 1e-4.")
    ap.add_argument("dirs", nargs='+', help="base result directories to scan")
    args = ap.parse_args(argv)

    all_summaries = []
    for d in args.dirs:
        base = pathlib.Path(d)
        if not base.exists():
            print(f"Warning: {d} not found", file=sys.stderr)
            continue
        print(f"Scanning {base} …")
        all_summaries.append(scan_and_filter(base))

    merged = merge_summaries(all_summaries)
    if not merged:
        print("No equilibria files found or none met threshold.")
        return 0

    print("\nHolding-period summary (kept / total):")
    for hp in sorted(k for k in merged if k >= 0):
        kept, total = merged[hp]
        pct = 100 * kept / total if total else 0.0
        print(f"  HP {hp:4d}: {kept:3d} / {total:3d}  ({pct:4.1f} %) interior equilibria ≤ 1e-4 regret")
    return 0


if __name__ == "__main__":
    sys.exit(main()) 