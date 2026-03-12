#!/usr/bin/env python3
"""
visualize_surplus.py – Standalone visualisations for one EGTA results folder.

Usage examples
--------------
1. Surplus distributions (hist + KDE) for MOBI agents
   $ python visualize_surplus.py /path/to/results --plot dist --role MOBI

2. Mean ± SE bar for ZI agents, save to PNG
   $ python visualize_surplus.py /path/to/results --plot mean --role ZI -o mean_se.png

3. Equilibrium share pie
   $ python visualize_surplus.py /path/to/results --plot share

4. Pay-off difference curve (π_CDA – π_MELO) for MOBI
   $ python visualize_surplus.py /path/to/results --plot diff --role MOBI

The script covers visualisations 1–5 from the suggested list; others can be
added following the same pattern.
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ─── STRATEGY LABELS ──────────────────────────────────────────────────────────
MOBI_MELO = "MOBI_0_100_shade0_0"
MOBI_CDA  = "MOBI_100_0_shade250_500"
ZI_MELO   = "ZI_0_100_shade250_500"
ZI_CDA    = "ZI_100_0_shade250_500"

ROLE_STRAT = {
    "MOBI": {"MELO": MOBI_MELO, "CDA": MOBI_CDA},
    "ZI":   {"MELO": ZI_MELO,   "CDA": ZI_CDA},
}

# ─── CLI ─────────────────────────────────────────────────────────────────────
PLOT_CHOICES = {"dist", "mean", "share", "diff", "phase"}

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Visualise surplus data for one EGTA results folder")
    ap.add_argument("results_dir", type=Path, help="Directory containing raw_payoff_data.json & equilibria_detailed.json")
    ap.add_argument("--plot", choices=PLOT_CHOICES, required=True, help="Which visualisation to draw")
    ap.add_argument("--role", choices=["MOBI", "ZI"], help="Agent role (required for dist, mean, diff plots)")
    ap.add_argument("--output", "-o", type=Path, help="Write figure to this PNG instead of showing")
    return ap.parse_args()

# ─── DATA HELPERS ────────────────────────────────────────────────────────────

def load_profiles(results_dir: Path):
    fp = results_dir / "raw_payoff_data.json"
    if not fp.is_file():
        sys.exit(f"raw_payoff_data.json not found in {results_dir}")
    return json.loads(fp.read_text())


def extract_payoffs(profiles, role: str) -> Dict[str, List[float]]:
    """Return dict {'MELO': [...], 'CDA': [...]} of payoffs for the given role."""
    strs = ROLE_STRAT[role]
    data = {"MELO": [], "CDA": []}
    for prof in profiles:
        for _, r, s, payoff in prof:
            if r != role:
                continue
            if s == strs["MELO"]:
                data["MELO"].append(payoff)
            elif s == strs["CDA"]:
                data["CDA"].append(payoff)
    return data

# ─── PLOTTERS ────────────────────────────────────────────────────────────────

def plot_surplus_distribution(payoffs: Dict[str, List[float]], role: str):
    plt.figure(figsize=(8,5))
    for label, color in [("CDA", "#2196F3"), ("MELO", "#8BC34A")]:
        sns.histplot(payoffs[label], kde=True, stat="density", label=label, color=color, alpha=0.4)
    plt.xlabel(f"{role} surplus")
    plt.ylabel("Density")
    plt.title(f"Surplus distribution – {role}")
    plt.legend()
    plt.tight_layout()


def plot_mean_se_bar(payoffs: Dict[str, List[float]], role: str):
    means = {k: np.mean(v) for k, v in payoffs.items() if v}
    ses   = {k: np.std(v, ddof=1)/np.sqrt(len(v)) for k, v in payoffs.items() if v}
    labels = list(means)
    plt.figure(figsize=(6,5))
    plt.bar(labels, [means[l] for l in labels], yerr=[ses[l] for l in labels], capsize=5, color=["#2196F3", "#8BC34A"])
    plt.ylabel("Mean surplus ± SE")
    plt.title(f"Mean surplus ({role})")
    plt.tight_layout()


def plot_equilibrium_share(results_dir: Path):
    fp = results_dir / "equilibria_detailed.json"
    if not fp.is_file():
        sys.exit("equilibria_detailed.json not found, cannot plot share")
    eq = json.loads(fp.read_text())[0]  # first equilibrium
    shares = {}
    for role, mix in eq["mixture_by_role"].items():
        for strat, frac in mix.items():
            key = f"{role}\n{strat.split('_shade')[0]}"  # shorter label
            shares[key] = frac
    plt.figure(figsize=(5,5))
    plt.pie(list(shares.values()), labels=list(shares.keys()), autopct="%1.0f%%", startangle=90)
    plt.title("Equilibrium strategy mix")
    plt.tight_layout()


def _payoff_diff_by_share(profiles, role: str):
    """Return sorted arrays xs (share) and diffs (π_CDA−π_MELO) for given role."""
    strs = ROLE_STRAT[role]
    buckets = {}
    for prof in profiles:
        k = sum(1 for _, r, s, _ in prof if r == role and s == strs["MELO"])
        pay_M = [p for _, r, s, p in prof if r == role and s == strs["MELO"]]
        pay_C = [p for _, r, s, p in prof if r == role and s == strs["CDA"]]
        if pay_M and pay_C:
            buckets.setdefault(k, []).append(np.mean(pay_C) - np.mean(pay_M))
    if not buckets:
        return None, None
    ks  = np.array(sorted(buckets))
    dif = np.array([np.mean(buckets[k]) for k in ks])
    xs  = ks / max(ks)  # map k to share in [0,1]
    return xs, dif


def plot_payoff_difference_curve(profiles, role: str):
    xs, diffs = _payoff_diff_by_share(profiles, role)
    if xs is None:
        sys.exit("No profiles with both strategies present – cannot draw diff curve")
    plt.figure(figsize=(8,5))
    plt.axhline(0, color="black", lw=1)
    plt.plot(xs, diffs, marker="o")
    plt.xlabel(f"Share of {role}s playing M-ELO (x)")
    plt.ylabel("π_CDA − π_MELO")
    plt.title(f"Payoff difference curve – {role}")
    plt.tight_layout()


def plot_phase_line(profiles, role: str):
    """1-D phase portrait: arrows show sign of dx/dt where x = share MELO."""
    xs, diffs = _payoff_diff_by_share(profiles, role)
    if xs is None:
        sys.exit("Cannot construct phase line – need payoff differences")

    # Determine direction: positive diff ⇒ prefer CDA ⇒ x decreases (arrow left)
    # We take sign = -sign(diffs).
    dirs = np.sign(-diffs)

    plt.figure(figsize=(8, 2))
    plt.axhline(0, color="black")
    plt.xlim(0, 1)
    plt.yticks([])
    plt.xlabel(f"Share of {role}s playing M-ELO (x)")
    plt.title(f"Phase line – {role}")

    # Draw arrows between points
    for i in range(len(xs) - 1):
        mid = 0.5 * (xs[i] + xs[i+1])
        direction = -np.sign(diffs[i])  # as above
        if direction == 0:
            continue
        dx = 0.05 if direction > 0 else -0.05
        plt.annotate("", xy=(mid + dx, 0), xytext=(mid, 0),
                     arrowprops=dict(arrowstyle="->", color="blue"))

    # Mark potential equilibria where diff changes sign
    for i in range(len(xs)-1):
        if diffs[i] == 0 or diffs[i]*diffs[i+1] < 0:
            eq_x = xs[i] if diffs[i]==0 else 0.5*(xs[i]+xs[i+1])
            plt.plot(eq_x, 0, marker="o", color="red")

    plt.tight_layout()

# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    profiles = load_profiles(args.results_dir)

    if args.plot in {"dist", "mean", "diff"} and not args.role:
        sys.exit("--role is required for this plot type")

    if args.plot == "dist":
        pay = extract_payoffs(profiles, args.role)
        plot_surplus_distribution(pay, args.role)
    elif args.plot == "mean":
        pay = extract_payoffs(profiles, args.role)
        plot_mean_se_bar(pay, args.role)
    elif args.plot == "share":
        plot_equilibrium_share(args.results_dir)
    elif args.plot == "diff":
        plot_payoff_difference_curve(profiles, args.role)
    elif args.plot == "phase":
        plot_phase_line(profiles, args.role)
    else:
        sys.exit("Unknown plot type")

    if args.output:
        out = args.output.with_suffix('.png')
        plt.savefig(out, dpi=300)
        print(f"Figure saved to {out}")
    else:
        plt.show()

if __name__ == "__main__":
    main() 