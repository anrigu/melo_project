import json, pathlib, os, statistics, numpy as np, pandas as pd, matplotlib.pyplot as plt
from collections import defaultdict

path = "/mnt/data/raw_payoff_data.json"
assert os.path.isfile(path)

data = json.loads(pathlib.Path(path).read_text())

def venue_map(role):
    return ("{}_0_100".format(role), "{}_100_0".format(role))  # (MELO, CDA)

def collect(role):
    strat_melo, strat_cda = venue_map(role)
    pay_melo = defaultdict(list)
    pay_cda  = defaultdict(list)
    for prof in data:
        k = sum(1 for _,r,s,_ in prof if r==role and s==strat_melo)
        if any(r==role and s==strat_melo for _,r,s,_ in prof) and any(r==role and s==strat_cda for _,r,s,_ in prof):
            for _,r,s,pay in prof:
                if r!=role: continue
                if s==strat_melo:
                    pay_melo[k].append(pay)
                elif s==strat_cda:
                    pay_cda[k].append(pay)
    total = max(max(pay_melo, default=0), max(pay_cda, default=0))
    ks = list(range(total+1))
    melo_vals = [statistics.mean(pay_melo[k]) if k in pay_melo else np.nan for k in ks]
    cda_vals  = [statistics.mean(pay_cda[k])  if k in pay_cda  else np.nan for k in ks]
    melo_vals = pd.Series(melo_vals).interpolate(limit_direction='both').to_list()
    cda_vals  = pd.Series(cda_vals ).interpolate(limit_direction='both').to_list()
    return ks, melo_vals, cda_vals

def plot(role, ks, melo_vals, cda_vals):
    fig, ax = plt.subplots(figsize=(12,5))
    w=0.35
    # CDA bar on left, MELO bar right
    ax.bar([k-w/2 for k in ks], cda_vals,  width=w, color="#2196F3", label="Play CDA")
    ax.bar([k+w/2 for k in ks], melo_vals, width=w, color="#8BC34A", label="Play M‑ELO")
    for k, y_c, y_m in zip(ks, cda_vals, melo_vals):
        if np.isnan(y_c) or np.isnan(y_m): continue
        if abs(y_c - y_m) < 1e-6: continue
        start_x = k - w/2 if y_c < y_m else k + w/2
        end_x   = k + w/2 if y_c < y_m else k - w/2
        ax.annotate('', xy=(end_x, max(y_c, y_m)), xytext=(start_x, min(y_c, y_m)),
                    arrowprops=dict(arrowstyle='->', color='black', lw=1))
    ax.set_xlabel(f'# {role}s choosing M‑ELO')
    ax.set_ylabel('Average payoff')
    ax.set_title(f'{role} – average payoffs by strategy mix\n(arrows = profitable one‑agent deviations)')
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.legend()
    plt.xticks(ks, rotation=45)
    plt.tight_layout()
    return fig

fig_mobi = plot("MOBI", *collect("MOBI"))
fig_zi   = plot("ZI",   *collect("ZI"))
plt.show()
