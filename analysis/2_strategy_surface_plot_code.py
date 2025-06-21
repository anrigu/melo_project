#!/usr/bin/env python3
# Plot welfare surfaces and overlay Nash equilibria (pure or mixed)
import argparse, json, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation, LinearTriInterpolator

# ─── ARGPARSE ──────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Visualise welfare surfaces and equilibria locations.")
    ap.add_argument("results_dir", type=str,
                    help="Folder containing raw_payoff_data.json and "
                         "raw_welfare_metrics.json")
    ap.add_argument("--output", "-o", type=str, default=None,
                    help="Path to save the figure (PNG). If omitted, the plot "
                         "is shown interactively.")
    return ap.parse_args()

# Helper predicates to identify the M-ELO (0/100) strategies which may have
# extra suffixes such as shading parameters.
# We treat any strategy string that *starts with* the base name as belonging to
# that conceptual strategy family.
MOBI_MELO_PREFIX = "MOBI_0_100"
ZI_MELO_PREFIX   = "ZI_0_100"

def mobi_is_on_melo(strat: str) -> bool:
    return strat.startswith(MOBI_MELO_PREFIX)

def zi_is_on_melo(strat: str) -> bool:
    return strat.startswith(ZI_MELO_PREFIX)

# ─── LOAD DATA ─────────────────────────────────────────────────────────
args = parse_args()
results_dir = args.results_dir

with open(os.path.join(results_dir, "raw_payoff_data.json")) as f:
    payoff_profiles = json.load(f)

with open(os.path.join(results_dir, "equilibria_detailed.json")) as f:
    equilibria = json.load(f)

# ─── MAP PROFILES TO (x, y) GRID ───────────────────────────────────────
profile_metrics = {}
for profile in payoff_profiles:
    total_mobi = total_zi = mobi_on_melo = zi_on_melo = 0
    mobi_payoffs, zi_payoffs = [], []

    for _, role, strat, payoff in profile:
        if role == "MOBI":
            total_mobi += 1
            mobi_payoffs.append(payoff)
            if mobi_is_on_melo(strat):           # MOBI chooses M-ELO
                mobi_on_melo += 1
        elif role == "ZI":
            total_zi += 1
            zi_payoffs.append(payoff)
            if zi_is_on_melo(strat):             # ZI chooses M-ELO
                zi_on_melo += 1

    x = mobi_on_melo / total_mobi              # MOBI share on M-ELO
    y = zi_on_melo   / total_zi                #  ZI  share on M-ELO
    key = (x, y)

    collective = sum(mobi_payoffs) + sum(zi_payoffs)
    mobi_avg   = np.mean(mobi_payoffs)
    zi_avg     = np.mean(zi_payoffs)

    if key not in profile_metrics:
        profile_metrics[key] = {"collective": [], "mobi": [], "zi": []}
    profile_metrics[key]["collective"].append(collective)
    profile_metrics[key]["mobi"].append(mobi_avg)
    profile_metrics[key]["zi"].append(zi_avg)

Xs, Ys, Z_collective, Z_mobi, Z_zi = [], [], [], [], []
for (x, y), vals in profile_metrics.items():
    Xs.append(x);  Ys.append(y)
    Z_collective.append(np.mean(vals["collective"]))
    Z_mobi.append(np.mean(vals["mobi"]))
    Z_zi.append(np.mean(vals["zi"]))

Xs, Ys = map(np.array, (Xs, Ys))
Z_collective, Z_mobi, Z_zi = map(np.array, (Z_collective, Z_mobi, Z_zi))

# Require at least 3 distinct (x,y) points for triangulation. If not available,
# print a helpful message and exit gracefully.
if len(Xs) < 3:
    raise ValueError(
        f"Need at least 3 distinct (x, y) points for surface plotting; got {len(Xs)}.\n"
        "Check that your raw_payoff_data.json contains variation in strategy mixes "
        "(in particular MOBI_0_100* and ZI_0_100* strategies).")

tri = Triangulation(Xs, Ys)

# ─── EQUILIBRIUM LOCATIONS ────────────────────────────────────────────
def xy_from_equilibrium(eq):
    """Return (x, y) share for an equilibrium record."""
    m_mix = eq.get("mixture_by_role", {}).get("MOBI", {})
    z_mix = eq.get("mixture_by_role", {}).get("ZI",   {})

    x_share = sum(frac for strat, frac in m_mix.items() if mobi_is_on_melo(strat))
    y_share = sum(frac for strat, frac in z_mix.items() if zi_is_on_melo(strat))
    return x_share, y_share

eq_pts = [xy_from_equilibrium(eq) for eq in equilibria]

# ─── INTERPOLATORS FOR TRUE HEIGHT ─────────────────────────────────────
interp_funcs = [
    LinearTriInterpolator(tri, Z_zi),
    LinearTriInterpolator(tri, Z_mobi),
    LinearTriInterpolator(tri, Z_collective),
]
surface_offsets = [0.02 * (zs.max() - zs.min()) for zs in
                   (Z_zi, Z_mobi, Z_collective)]  # 2 % lift

# ─── PLOTTING ──────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 6))
titles = ["ZI Welfare", "MOBI Welfare", "Collective Welfare"]
Z_sets = [Z_zi,         Z_mobi,         Z_collective]

for i in range(3):
    ax = fig.add_subplot(1, 3, i + 1, projection="3d")
    ax.plot_trisurf(tri, Z_sets[i], linewidth=0.2, alpha=0.85, shade=False)

    # maximum-welfare point
    idx_max = np.argmax(Z_sets[i])
    z_lift  = surface_offsets[i]
    ax.scatter(Xs[idx_max], Ys[idx_max], Z_sets[i][idx_max] + z_lift,
               c="red", marker="^", s=80, depthshade=False)
    ax.text(Xs[idx_max], Ys[idx_max],
            Z_sets[i][idx_max] + 1.5 * z_lift,          # text a bit higher
            f"{Z_sets[i][idx_max]:.2f}", color="red",
            fontsize=8, ha="left", va="bottom")

    # equilibria
    interp = interp_funcs[i]
    for xe, ye in eq_pts:
        z_val = interp(xe, ye)
        if np.ma.is_masked(z_val):                      # outside convex hull
            idx = np.argmin(np.abs(Xs - xe) + np.abs(Ys - ye))
            z_val = Z_sets[i][idx]
        ax.scatter(xe, ye, float(z_val) + z_lift,
                   c="black", s=60, depthshade=False)

    ax.set_xlabel("MOBI support on M-ELO")
    ax.set_ylabel("ZI support on M-ELO")
    ax.set_zlabel(titles[i])
    ax.set_title(f"{titles[i]} Surface")

fig.tight_layout()

# Save or show according to argument
if args.output:
    out_path = args.output if args.output.lower().endswith(".png") else args.output + ".png"
    plt.savefig(out_path, dpi=300)
    print(f"Figure saved to {out_path}")
else:
    plt.show()
