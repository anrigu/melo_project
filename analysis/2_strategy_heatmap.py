#!/usr/bin/env python3
"""Plot 2-D heat maps of agent and collective welfare.

For each profile in raw_payoff_data.json we map
  x = share of MOBIs choosing M-ELO  (0 → 1)
  y = share of ZIs   choosing M-ELO
and average payoffs at duplicate (x, y) points.

The script then draws three panels (ZI, MOBI, Collective) using
tricontour-filled colour plots so you can immediately spot hotspots.
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

# ─── CONSTANTS ──────────────────────────────────────────────────────────────
MOBI_MELO_PREFIX = "MOBI_0_100"
ZI_MELO_PREFIX   = "ZI_0_100"

# ─── CLI ────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Heat-map welfare visualisation for one EGTA results directory")
    ap.add_argument("results_dir", type=Path, help="Folder containing raw_payoff_data.json")
    ap.add_argument("--output", "-o", type=Path, help="Save PNG instead of showing interactively")
    return ap.parse_args()

# ─── HELPERS ────────────────────────────────────────────────────────────────

def is_melo(strat: str, prefix: str) -> bool:
    return strat.startswith(prefix)

# ─── MAIN ───────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    pay_fp = args.results_dir / "raw_payoff_data.json"
    if not pay_fp.is_file():
        sys.exit(f"{pay_fp} not found")

    profiles = json.loads(pay_fp.read_text())

    grid = {}  # (x,y) -> lists of payoffs
    for prof in profiles:
        total_mobi = total_zi = mobi_on_melo = zi_on_melo = 0
        mobi_pays, zi_pays = [], []
        for _, role, strat, payoff in prof:
            if role == "MOBI":
                total_mobi += 1
                mobi_pays.append(payoff)
                if is_melo(strat, MOBI_MELO_PREFIX):
                    mobi_on_melo += 1
            elif role == "ZI":
                total_zi += 1
                zi_pays.append(payoff)
                if is_melo(strat, ZI_MELO_PREFIX):
                    zi_on_melo += 1
        if total_mobi == 0 or total_zi == 0:
            continue  # malformed profile
        x = mobi_on_melo / total_mobi
        y = zi_on_melo   / total_zi
        key = (x, y)
        collective = sum(mobi_pays) + sum(zi_pays)
        if key not in grid:
            grid[key] = {"mobi": [], "zi": [], "collective": []}
        grid[key]["mobi"].append(np.mean(mobi_pays))
        grid[key]["zi"].append(np.mean(zi_pays))
        grid[key]["collective"].append(collective)

    if len(grid) < 3:
        sys.exit("Need at least 3 distinct (x,y) points for triangulation")

    xs, ys = map(np.array, zip(*grid.keys()))
    Z_mobi = np.array([np.mean(v["mobi"]) for v in grid.values()])
    Z_zi   = np.array([np.mean(v["zi"])   for v in grid.values()])
    Z_coll = np.array([np.mean(v["collective"]) for v in grid.values()])

    tri = Triangulation(xs, ys)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    titles = ["ZI Welfare", "MOBI Welfare", "Collective Welfare"]
    Z_sets = [Z_zi, Z_mobi, Z_coll]

    for ax, title, Z in zip(axes, titles, Z_sets):
        tcf = ax.tricontourf(tri, Z, levels=20, cmap="viridis")
        fig.colorbar(tcf, ax=ax, orientation="vertical")
        ax.set_xlabel("MOBI share on M-ELO")
        ax.set_ylabel("ZI share on M-ELO")
        ax.set_title(title)

    if args.output:
        out = args.output.with_suffix('.png')
        plt.savefig(out, dpi=300)
        print(f"Heat-map saved to {out}")
    else:
        plt.show()

if __name__ == "__main__":
    main() 