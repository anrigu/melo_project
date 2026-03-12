import json
import glob
import re
from pathlib import Path
from collections import defaultdict
from statistics import mean, stdev

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os, sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from marketsim.game.role_symmetric_game import RoleSymmetricGame

# Directories for the five seeds
SEED_ROOTS = [
    "result_two_role_still_role_symmetric_3",
    "result_two_role_still_role_symmetric_4",
    "result_two_role_still_role_symmetric_5",
    "result_two_role_still_role_symmetric_6",
    "result_two_role_still_role_symmetric_7",
]

seed_means = defaultdict(list)

for root in SEED_ROOTS: 
    for raw_file in glob.glob(f"{root}/**/raw_payoff_data.json", recursive=True):
        # infer holding period
        m = re.search(r"holding_period_(\d+)", raw_file)
        if m:
            hp = int(m.group(1))
        else:
            m2 = re.search(r"pilot_egta_run_(\d+)", raw_file)
            if not m2:
                continue
            hp = int(m2.group(1))

        with open(raw_file) as fh:
            data = json.load(fh)
        # detect format: list of lists or list of dicts
        local_stats = defaultdict(list)  # (role, strat) -> payoffs
        if data and isinstance(data[0], list):
            for profile in data:
                for entry in profile:
                    if len(entry) >= 4:
                        _, role, strat, payoff = entry[:4]
                        local_stats[(role, strat)].append(payoff)
        elif data and isinstance(data[0], dict):
            for obs in data:
                prof = obs.get("profile", [])
                pays = obs.get("payoffs", [])
                for p, payoff in zip(prof, pays):
                    role = p[1] if len(p) > 2 else p[0]
                    strat = p[2] if len(p) > 2 else p[1]
                    local_stats[(role, strat)].append(payoff)
        else:
            continue

        # compute file-level means and record per seed
        for (role, strat), lst in local_stats.items():
            if lst:
                seed_means[(hp, role, strat)].append(sum(lst) / len(lst))

# Build DataFrame with mean ± std across seeds
records = []
for (hp, role, strat), lst in seed_means.items():
    mu = mean(lst)
    sigma = stdev(lst) if len(lst) > 1 else 0.0
    records.append({
        "Holding Period": hp,
        "Role": role,
        "Strategy": strat,
        "Mean": mu,
        "Std": sigma,
    })

df = pd.DataFrame(records)
if df.empty:
    raise SystemExit("No data collected – check paths.")

sns.set_style("whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
for ax, role in zip(axes, df["Role"].unique()):
    sub = df[df["Role"] == role]
    for strat, g in sub.groupby("Strategy"):
        g_sorted = g.sort_values("Holding Period")
        ax.plot(g_sorted["Holding Period"], g_sorted["Mean"], marker="o", label=strat)
        ax.fill_between(
            g_sorted["Holding Period"],
            g_sorted["Mean"] - g_sorted["Std"],
            g_sorted["Mean"] + g_sorted["Std"],
            alpha=0.2,
        )
    ax.set_title(f"{role} – Payoff ±1σ vs Holding Period")
    ax.set_xlabel("Holding Period")
    ax.set_ylabel("Mean Payoff")
    ax.legend()

plt.tight_layout()

output = Path(__file__).with_name("payoffs_seed_avg_by_hp.png")
fig.savefig(output, dpi=300)
print(f"Saved aggregated payoff plot to {output.relative_to(Path.cwd())}") 