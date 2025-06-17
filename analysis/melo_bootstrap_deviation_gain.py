import json, numpy as np, pandas as pd, matplotlib.pyplot as plt
from collections import defaultdict

# PARAMETERS
path   = '/Users/gabesmithline/Desktop/pilot_egta_medium_run_300_CDA_Shade_26913543/comprehensive_rsg_results_20250615_161022/raw_payoff_data.json'
BOOT   = 2000                # bootstrap replications
THRESH = [0, 2, 5, 10, 20, 40]       # ZI-on-M-ELO cut-offs to plot
colors = ['red','orange','green','blue', 'yellow', 'maroon']
M_MELO, M_CDA = 'MOBI_0_100', 'MOBI_100_0'
Z_MELO        = 'ZI_0_100'

profiles = json.load(open(path))

def deviation_curve(zi_cap):
    """Return DataFrame with m, mean, lo, hi (interpolated) for given ZI cap."""
    diffs = defaultdict(list)
    for prof in profiles:
        z_melo = sum(1 for _,r,s,_ in prof if r=='ZI' and s==Z_MELO)
        if z_melo > zi_cap:
            continue                                 # keep only runs inside cap
        m_melo = sum(1 for _,r,s,_ in prof if r=='MOBI' and s==M_MELO)
        pay_M  = [p for _,r,s,p in prof if r=='MOBI' and s==M_MELO]
        pay_C  = [p for _,r,s,p in prof if r=='MOBI' and s==M_CDA]
        if pay_M and pay_C:
            diffs[m_melo].append(np.mean(pay_C) - np.mean(pay_M))

    rng   = np.random.default_rng(0)
    rows  = []
    for m in range(29):                              # 0 … 28 MOBIs on M-ELO
        arr = np.array(diffs.get(m, []))
        if arr.size:
            boots = rng.choice(arr, size=(BOOT, arr.size), replace=True).mean(axis=1)
            mean, lo, hi = arr.mean(), np.percentile(boots,2.5), np.percentile(boots,97.5)
        else:
            mean = lo = hi = np.nan
        rows.append([m, mean, lo, hi])

    return (pd.DataFrame(rows, columns=['m','mean','lo','hi'])
              .interpolate(limit_direction='both'))  # fill gaps for smooth curve

# ---------- plotting ----------
plt.figure(figsize=(10,5))
plt.axhline(0, color='black', lw=1)

for cap, col in zip(THRESH, colors):
    df = deviation_curve(cap)
    x, y, lo, hi = df['m'], df['mean'], df['lo'], df['hi']
    plt.fill_between(x, lo, hi, color=col, alpha=0.15)
    plt.plot(x, y, color=col, label=f'≤ {cap} ZI on M-ELO')

plt.xlabel('# MOBIs currently on M-ELO')
plt.ylabel('Payoff gain if one MOBI switches to CDA')
plt.title('MOBI deviation gain under varying ZI presence on M-ELO\n'
          '(positive ⇒ CDA strictly better)')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()

