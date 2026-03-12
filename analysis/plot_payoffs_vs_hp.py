import json
import glob
import re
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import seaborn as sns

try:
    import pandas as pd
except ImportError:
    raise SystemExit("Please pip-install pandas to run this script.")

# Root directory containing the results
ROOT = Path(__file__).resolve().parents[1] / (
    "result_2_role_still_role_symmetric_2_strategies_test_price_based_updates_false_normalize_payoffs_false_2"
)

# Collect payoffs: (hp, role, strat) -> [profits]
raw = defaultdict(list)
for f in glob.glob(str(ROOT / "**/raw_payoff_data.json"), recursive=True):
    # infer holding period
    m = re.search(r"holding_period_(\d+)", f)
    if m:
        hp = int(m.group(1))
    else:
        m2 = re.search(r"pilot_egta_run_(\d+)", f)
        if not m2:
            continue
        hp = int(m2.group(1))

    with open(f) as fh:
        data = json.load(fh)

    # Two possible formats: (1) list of player tuples per profile; (2) list of dict observations
    if data and isinstance(data[0], list) and data[0] and isinstance(data[0][0], list):
        # format (1)
        for profile in data:
            for entry in profile:
                if len(entry) < 4:
                    continue
                _, role, strat, payoff = entry[:4]
                raw[(hp, role, strat)].append(payoff)
    elif data and isinstance(data[0], dict):
        # format (2)
        for obs in data:
            prof = obs.get("profile", [])
            pays = obs.get("payoffs", [])
            for p, payoff in zip(prof, pays):
                if len(p) == 3:
                    _, role, strat = p
                elif len(p) == 2:
                    role, strat = p
                else:
                    continue
                raw[(hp, role, strat)].append(payoff)

# Build tidy DataFrame
records = []
for (hp, role, strat), lst in raw.items():
    if lst:
        records.append({
            "Holding Period": hp,
            "Role": role,
            "Strategy": strat,
            "Mean Payoff": sum(lst) / len(lst),
        })

df = pd.DataFrame(records)

# Sort for plotting
df.sort_values(["Role", "Strategy", "Holding Period"], inplace=True)

sns.set_style("whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
roles = df["Role"].unique()
colors = sns.color_palette("Set2", 2)

for ax, role in zip(axes, roles):
    sub = df[df["Role"] == role]
    sns.lineplot(
        data=sub,
        x="Holding Period",
        y="Mean Payoff",
        hue="Strategy",
        marker="o",
        palette="Set1",
        ax=ax,
    )
    ax.set_title(f"{role} – Mean Payoff vs Holding Period")
    ax.set_xlabel("Holding Period")
    ax.set_ylabel("Mean Payoff")
    ax.legend(title="Strategy")

plt.tight_layout()
out_path = Path(__file__).with_name("payoffs_by_hp.png")
fig.savefig(out_path, dpi=300)
print(f"Saved payoff plot to {out_path.relative_to(Path.cwd())}") 