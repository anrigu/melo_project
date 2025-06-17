import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
import os, sys, json, argparse, torch
# ----------- Load data -----------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Compute raw-unit welfare & regrets (NaN-robust).")
    ap.add_argument("results_dir", type=str,
                    help="Folder containing experiment_parameters.json, "
                         "equilibria_detailed.json, raw_payoff_data.json")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--normalize", action="store_true",
                    help="Normalise payoffs when building the game "
                         "(subtract global mean, divide by std).")
    return ap.parse_args()


args = parse_args()
rd   = args.results_dir

with open(os.path.join(rd, 'raw_payoff_data.json')) as f:
    payoff_profiles = json.load(f)

with open(os.path.join(rd,'raw_welfare_metrics.json')) as f:
    equilibria = json.load(f)

# ----------- Map profiles to (x, y) -----------


profile_metrics = {}
for profile in payoff_profiles:
    total_mobi = total_zi = 0
    mobi_on_melo = zi_on_melo = 0
    mobi_payoffs, zi_payoffs = [], []
    
    for _, role, strat, payoff in profile:
        if role == "MOBI":
            total_mobi += 1
            mobi_payoffs.append(payoff)
            if strat == "MOBI_0_100":   
                mobi_on_melo += 1
        elif role == "ZI":
            total_zi += 1
            zi_payoffs.append(payoff)
            if strat == "ZI_0_100":     
                zi_on_melo += 1
    
    # Fractions for axes
    x = mobi_on_melo / total_mobi
    y = zi_on_melo / total_zi
    key = (x, y)
    
    # Aggregate welfare metrics
    collective = sum(mobi_payoffs) + sum(zi_payoffs)
    mobi_avg   = np.mean(mobi_payoffs)
    zi_avg     = np.mean(zi_payoffs)
    
    if key not in profile_metrics:
        profile_metrics[key] = {"collective": [], "mobi": [], "zi": []}
    profile_metrics[key]["collective"].append(collective)
    profile_metrics[key]["mobi"].append(mobi_avg)
    profile_metrics[key]["zi"].append(zi_avg)

# Convert to arrays for plotting
Xs, Ys, Z_collective, Z_mobi, Z_zi = [], [], [], [], []
for (x, y), vals in profile_metrics.items():
    Xs.append(x)
    Ys.append(y)
    Z_collective.append(np.mean(vals["collective"]))
    Z_mobi.append(np.mean(vals["mobi"]))
    Z_zi.append(np.mean(vals["zi"]))

Xs, Ys = np.array(Xs), np.array(Ys)
Z_collective, Z_mobi, Z_zi = map(np.array, (Z_collective, Z_mobi, Z_zi))

tri = Triangulation(Xs, Ys)

# ----------- Determine equilibrium points -----------
eq_pts = []
for eq in equilibria:
    mobi_strat = next(iter(eq["mixture_by_role"]["MOBI"]))
    zi_strat   = next(iter(eq["mixture_by_role"]["ZI"]))
    x_eq = 1.0 if mobi_strat == "MOBI_0_100" else 0.0
    y_eq = 1.0 if zi_strat   == "ZI_0_100"   else 0.0
    eq_pts.append((x_eq, y_eq))

# ----------- Plotting -----------
fig = plt.figure(figsize=(18, 6))
titles = ["ZI Welfare", "MOBI Welfare", "Collective Welfare"]
Z_sets = [Z_zi, Z_mobi, Z_collective]

for i in range(3):
    ax = fig.add_subplot(1, 3, i+1, projection='3d')
    surf = ax.plot_trisurf(
        tri, Z_sets[i],            # keep colour mapping by value
        linewidth=0.2,
        alpha=0.85,
        shade=False)

    # --- mark maximum-welfare point ---
    max_idx = np.argmax(Z_sets[i])               # index of maximum value
    x_max, y_max, z_max = Xs[max_idx], Ys[max_idx], Z_sets[i][max_idx]
    ax.scatter(x_max, y_max, z_max,
               c='red', marker='^', s=80, label='Max welfare')
    ax.text(x_max, y_max, z_max,
            f"{z_max:.2f}", color='red', fontsize=9,
            horizontalalignment='left', verticalalignment='bottom')

    # equilibrium dots
    for xe, ye in eq_pts:
        # find closest sample point to position marker vertically
        dists = np.abs(Xs - xe) + np.abs(Ys - ye)
        idx = np.argmin(dists)
        ax.scatter(xe, ye, Z_sets[i][idx], c='black', s=60)
    ax.set_xlabel("MOBI support on M‑ELO")
    ax.set_ylabel("ZI support on M‑ELO")
    ax.set_zlabel(titles[i])
    ax.set_title(f"H={len(profile)}: {titles[i]} Surface")
    
fig.tight_layout()
plt.show()
