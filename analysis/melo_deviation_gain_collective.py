import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path

# ---------- parameters ----------
RAW_FILE = "/Users/gabesmithline/Desktop/pilot_egta_medium_run_250_CDA_Shade_26913541/comprehensive_rsg_results_20250615_163200/raw_payoff_data.json"
BOOT     = 20000                 # bootstrap replications
M_MELO, M_CDA = "MOBI_0_100", "MOBI_100_0"
Z_MELO        = "ZI_0_100"

# ---------- load ----------
profiles = json.loads(Path(RAW_FILE).read_text())

# ---------- collect payoff differences ----------
diffs = defaultdict(list)           # key = m_melo, value = list of Δ
for prof in profiles:
    m_melo = sum(1 for _, r, s, _ in prof if r == "MOBI" and s == M_MELO)
    pay_M  = [p for _, r, s, p in prof if r == "MOBI" and s == M_MELO]
    pay_C  = [p for _, r, s, p in prof if r == "MOBI" and s == M_CDA]
    if pay_M and pay_C:
        diffs[m_melo].append(np.mean(pay_C) - np.mean(pay_M))  # +ve ⇒ CDA better

# ---------- bootstrap mean & CI ----------
rng   = np.random.default_rng(1)
rows  = []
MAX_M = 28                       # 0 … 28 MOBIs on M-ELO
for m in range(MAX_M + 1):
    arr = np.array(diffs.get(m, []))
    if arr.size:
        boots = rng.choice(arr, size=(BOOT, arr.size), replace=True).mean(axis=1)
        mean, lo, hi = arr.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5)
    else:
        mean = lo = hi = np.nan
    rows.append([m, mean, lo, hi])

df = (pd.DataFrame(rows, columns=["m", "mean", "lo", "hi"])
        .interpolate(limit_direction="both"))      # fill missing m

# ---------- plot ----------
x, y, lo, hi = df["m"].to_numpy(), df["mean"].to_numpy(), df["lo"], df["hi"]

plt.figure(figsize=(9, 5))
plt.axhline(0, color="black", lw=1)
plt.fill_between(x, lo, hi, color="steelblue", alpha=0.25, label="95 % CI")
plt.plot(x, y, color="steelblue", marker="o", label="mean Δ  (CDA − M-ELO)")
plt.xlabel("# MOBIs currently on M-ELO   (ZIs pooled across all counts)")
plt.ylabel("Payoff gain if one MOBI switches to CDA")
plt.title("Average deviation gain bootstrapped over all ZI liquidity states\n"
          "(positive ⇒ CDA strictly better for the deviator)")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()