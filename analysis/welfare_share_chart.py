import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
JSON_PATH = Path('analysis/rsne_rd_raw.json')  # path to the equilibria file
HP_ORDER = [0,10,20,30,40,50,60,70,80,90,100,120,140,160,180,200,220,240,260,280,300,320,340,400,500,600,700,800,900,1000]
DROP_AFTER = 340  # >=340 aggregated into one bucket

# ------------------------------------------------------------------
# Load JSON and extract payoffs per HP
# ------------------------------------------------------------------
raw = json.loads(JSON_PATH.read_text())

rows = []
for hp in HP_ORDER:
    hp_key = str(hp)
    if hp_key not in raw:
        continue
    eq_list = raw[hp_key]['equilibria']
    # choose which equilibrium row to use
    if hp >= DROP_AFTER:
        # prefer CDA corner; if not present, fall back to replicator
        cand = [e for e in eq_list if e['type'].startswith('corner_100_0')]
        if cand:
            eq = cand[0]
        else:
            eq = next(e for e in eq_list if e['type'] == 'replicator')
    else:
        eq = next(e for e in eq_list if e['type'] == 'replicator')
    mobi_pay, zi_pay = eq['payoffs_per_role']
    rows.append((hp, mobi_pay, zi_pay))

# ------------------------------------------------------------------
# Build DataFrame and category labels similar to original chart
# ------------------------------------------------------------------

df = pd.DataFrame(rows, columns=['t_hold','MOBI_payoff','ZI_payoff'])

# Optional: group into same coarse categories as old figure
bins = [ -1, 0, 10, 20, 40, 70, 100, 160, 400, 1e6]
labels = ['0','10','20','30–40','50–70','80–100','120–160','240–400','>400']
df['category'] = pd.cut(df['t_hold'], bins=bins, labels=labels)

# ------------------------------------------------------------------
# Build contiguous categories as per requested table
# ------------------------------------------------------------------

contig_labels = ['0','10','20','30','40','50','60','70','80','90','100','120',
                 '140','160','180','200','220','240','260','280','300','320','≥340']

# map each row to its label
label_for_hp = {}
for hp in HP_ORDER:
    if hp < 340:
        label_for_hp[hp] = str(hp)
    else:
        label_for_hp[hp] = '≥340'

df['cat'] = df['t_hold'].map(label_for_hp)

grp = df.groupby('cat', observed=True).mean(numeric_only=True).reindex(contig_labels)

# ------------------------------------------------------------------
# Plot
# ------------------------------------------------------------------
fig, ax1 = plt.subplots(figsize=(10,5))

x = np.arange(len(contig_labels))
ax1.plot(x, grp['MOBI_payoff'], marker='o', label='MOBI payoff')
ax1.plot(x, grp['ZI_payoff'], marker='s', label='ZI payoff')

ax1.set_xticks(x)
ax1.set_xticklabels(contig_labels, rotation=45)
ax1.set_xlabel('Holding-period category')
ax1.set_ylabel('Average payoff')
ax1.set_title('Equilibrium Payoffs vs. Holding-period (contiguous categories)')
ax1.legend(loc='upper left')

plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('updated_payoff_chart.png', dpi=300)
print('Chart saved to updated_payoff_chart.png') 