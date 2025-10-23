import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# Raw equilibrium shares of the midpoint (M-ELO) strategy per holding period
# ------------------------------------------------------------------

data = {
    't_hold': [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 400, 500, 600, 700, 800, 900, 1000],
    'p_ZI':   [0.00, 0.00, 1.00, 0.00, 0.00, 1.00, 1.00, 1.00, 0.00, 0.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    'p_MOBI': [0.00, 0.00, 0.57, 0.24, 0.29, 0.77, 0.54, 0.55, 0.00, 0.00, 0.55, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.48, 0.00, 0.63, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.00]
}

df = pd.DataFrame(data)

# ------------------------------------------------------------------
# Plot
# ------------------------------------------------------------------
plt.figure(figsize=(10, 5))
plt.plot(df['t_hold'], df['p_MOBI'], marker='o', linestyle='-', color='tab:blue', label='MOBI share in M-ELO')
plt.plot(df['t_hold'], df['p_ZI'], marker='s', linestyle='--', color='tab:orange', label='ZI share in M-ELO')

plt.xlabel('Holding-period $t_{hold}$')
plt.ylabel('Probability mass on M-ELO')
plt.ylim(-0.05, 1.05)
plt.title('Equilibrium use of M-ELO vs. Holding-period')
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('melo_share_chart.png', dpi=300)
print('Plot saved to melo_share_chart.png') 