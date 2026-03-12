import pandas as pd
import matplotlib.pyplot as plt

# Summary data by holding-period category
data = {
    'category': ['0', '10', '20', '30–40', '50–70', '80–100', '120–160', '240–400', '>400'],
    'MOBI_payoff': [22208.29, 14530.86, 14741.77, 14349.20, 14348.56, 21838.09, 21464.46, 18524.42, 21804.79],
    'ZI_payoff':   [ 9824.48, 12095.58, 11587.61, 12100.25, 11066.94, 10166.46, 10376.23, 10338.50, 10122.36],
}

df = pd.DataFrame(data)

# Plotting payoffs and welfare
fig, ax1 = plt.subplots(figsize=(10, 5))
ax2 = ax1.twinx()

x = range(len(df))
ax1.plot(x, df['MOBI_payoff'], marker='o', label='MOBI payoff')
ax1.plot(x, df['ZI_payoff'], marker='s', label='ZI payoff')

# Axis labels and ticks
ax1.set_xticks(x)
ax1.set_xticklabels(df['category'])
ax1.set_xlabel('Holding-period category')
ax1.set_ylabel('Average payoff')

# Legends
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

plt.title('Equilibrium Payoffs and Total Welfare vs. Holding-period')
plt.grid(alpha=0.3)
plt.tight_layout()
# Save to file
plt.savefig('hold_period_payoff_welfare.png', dpi=300)

