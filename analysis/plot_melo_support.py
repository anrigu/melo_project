import json
import glob
import re
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import seaborn as sns

# Directory that contains all result folders
BASE_DIR = (
    Path(__file__).resolve().parents[1]
    / "result_2_role_still_role_symmetric_2_strategies_test_price_based_updates_false_normalize_payoffs_false_2"
)

# Identify pure-strategy mixtures so we can ignore them
PURE_PROFILES = {
    (1.0, 0.0, 1.0, 0.0),
    (0.0, 1.0, 1.0, 0.0),
    (0.0, 1.0, 0.0, 1.0),
    (1.0, 0.0, 0.0, 1.0),
}

# 1. Collect data: holding-period → best mixed-equilibrium mixture (lowest regret)
hp_to_support = {}
for eq_file in glob.glob(str(BASE_DIR / "**/equilibria.json"), recursive=True):
    # Extract holding period either from `holding_period_X` or the pilot run suffix
    m = re.search(r"holding_period_(\d+)", eq_file)
    if m:
        hp = int(m.group(1))
    else:
        m2 = re.search(r"pilot_egta_run_(\d+)", eq_file)
        if not m2:
            continue  # Skip if we cannot infer HP
        hp = int(m2.group(1))

    with open(eq_file) as fh:
        equilibria = json.load(fh)

    # Filter out pure profiles
    mixed_profiles = [e for e in equilibria if tuple(round(x, 3) for x in e["mixture"]) not in PURE_PROFILES]
    if not mixed_profiles:
        continue

    # Pick profile with minimal regret
    best_profile = min(mixed_profiles, key=lambda e: e["regret"])
    mix = best_profile["mixture"]

    # Record MELO support probability for each role (index 1 for role 1, index 3 for role 2)
    hp_to_support.setdefault(hp, []).append((mix[1], mix[3]))

# 2. Aggregate (take average if duplicate files for same HP)
plot_data = []  # rows: (hp, role_label, support)
for hp, pairs in sorted(hp_to_support.items()):
    role1_avg = sum(r1 for r1, _ in pairs) / len(pairs)
    role2_avg = sum(r2 for _, r2 in pairs) / len(pairs)
    plot_data.append((hp, "MOBI (Role 1)", role1_avg))
    plot_data.append((hp, "ZI (Role 2)", role2_avg))

# 3. Build DataFrame for seaborn
try:
    import pandas as pd
except ImportError:
    raise SystemExit("pandas is required for this script. Install via `pip install pandas`. ")

df = pd.DataFrame(plot_data, columns=["Holding Period", "Role", "M-ELO Support"])

# 4. Plot
sns.set_style("whitegrid")
fig, ax = plt.subplots(figsize=(14, 6))

sns.barplot(
    data=df,
    x="Holding Period",
    y="M-ELO Support",
    hue="Role",
    palette="Set2",
    ax=ax,
)

ax.set_title("M-ELO Support in Best Mixed Equilibrium vs. Holding Period")
ax.set_ylabel("Probability of Playing M-ELO")
ax.set_xlabel("Holding Period")
ax.set_ylim(0, 1)
plt.xticks(rotation=45)
plt.tight_layout()

# 5. Save figure
output_path = Path(__file__).with_name("melo_support_by_hp.png")
fig.savefig(output_path, dpi=300)
print(f"Saved plot to {output_path.relative_to(Path.cwd())}") 