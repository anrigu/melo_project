#!/usr/bin/env python3
"""bifurcation_q_vs_hp.py

Plot the bifurcation curve
    hp  ↦  q^(hp) = Pr[ ZI plays MELO ]
for the deviation-preserving-reduced (4×4) game.

Run:
    python scripts/bifurcation_q_vs_hp.py           # default 0..300 in steps of 5
    python scripts/bifurcation_q_vs_hp.py --min 40 --max 200 --step 2

The script reuses *analysis.plot_bootstrap_compare.find_pooled_equilibrium_dpr*,
so it works out-of-the-box with the existing profile data.
It does **not** trigger additional simulations; if an HP value cannot be
constructed from the stored profiles it is skipped (shown as gap in the plot).
"""
from __future__ import annotations

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless default – change if you need interactive
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# ---------------------------------------------------------------------------
# Make project root importable so we can reach analysis.plot_bootstrap_compare
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# pylint: disable=wrong-import-position
from analysis.plot_bootstrap_compare import find_pooled_equilibrium_dpr
# pylint: enable=wrong-import-position


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot q^(hp) bifurcation curve from DPR equilibria.")
    ap.add_argument("--min", type=int, default=0, help="minimum HP (inclusive)")
    ap.add_argument("--max", type=int, default=300, help="maximum HP (inclusive)")
    ap.add_argument("--step", type=int, default=5, help="HP grid step size")
    ap.add_argument("--n_red", type=int, default=4, help="players per role in DPR reduction")
    ap.add_argument("--out", type=str, default="bifurcation_q_vs_hp.png", help="output PNG filename")
    args = ap.parse_args()

    hp_vals = np.arange(args.min, args.max + 1, args.step)
    q_vals: list[float] = []

    print("Computing DPR equilibria …")
    for hp in hp_vals:
        try:
            mix_can = find_pooled_equilibrium_dpr(hp, n_red=args.n_red, print_only=True)
            q_hp = float(mix_can[2])  # canonical idx 2 = ZI_MELO
        except Exception as exc:
            print(f"  HP {hp:>3}: could not build game ({exc}) – skipping")
            q_hp = np.nan
        q_vals.append(q_hp)

    hp_arr = np.array(hp_vals)
    q_arr = np.array(q_vals)

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(hp_arr, q_arr, lw=2, color="black", label=r"$q^{(hp)}$ = Pr[ZI MELO]")

    # highlight interior-equilibrium segment (0<q<1)
    interior_mask = (q_arr > 1e-3) & (q_arr < 1 - 1e-3)
    ax.scatter(hp_arr[interior_mask], q_arr[interior_mask], c="red", zorder=5, s=30, label="mixed EQ")

    ax.set_xlabel("Holding Period (HP)")
    ax.set_ylabel("Pr[ ZI plays MELO ] in DPR equilibrium")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Bifurcation diagram – ZI-MELO share vs HP")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()

    out_path = Path(args.out).with_suffix(".png").resolve()
    fig.savefig(out_path, dpi=150)
    print("Saved", out_path.relative_to(Path.cwd()))


if __name__ == "__main__":
    main()
