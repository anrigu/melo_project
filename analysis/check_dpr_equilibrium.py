#!/usr/bin/env python3
"""check_dpr_equilibrium.py

Verify that the canonical mixture
    base_mix = [1.0, 0.0, 0.75, 0.25]  # [MOBI_MELO, MOBI_CDA, ZI_MELO, ZI_CDA]
is a (weak) Nash equilibrium in the DPR reduced game using the DPR payoff
CSV tables produced earlier by plot_bootstrap_compare.

The script follows the algorithm outlined by the user:
  * For each HP in {0, 20, 40, 60, 80}, load the corresponding
    analysis/figures/dpr_table_hp_<HP>.csv file.
  * Assume n_red = 4 players per role.  The canonical mix therefore
    corresponds to counts (m = 4 MOBI_MELO, z = 3 ZI_MELO).
  * For every strategy that has positive mass in the mix, compare the
    incumbent payoff against a unilateral deviation to the alternative
    strategy of the same role and print the resulting regret.

All regrets should be \u2264 0 for the mix to be a weak Nash equilibrium.
"""
from pathlib import Path
import pandas as pd

# Canonical mixture and reduced player counts
base_mix = [1.0, 0.0, 0.75, 0.25]  # [MOBI_MELO, MOBI_CDA, ZI_MELO, ZI_CDA]

n_red = 4  # players per role in DPR game
m = int(base_mix[0] * n_red)  # MOBI_MELO count
z = int(base_mix[2] * n_red)  # ZI_MELO count

HP_LIST = [0, 20, 40, 60, 80]
ROOT = Path("analysis/figures")

roles_strats = [
    ("MOBI", "MELO", base_mix[0]),
    ("MOBI", "CDA", base_mix[1]),
    ("ZI", "MELO", base_mix[2]),
    ("ZI", "CDA", base_mix[3]),
]

def payoff_col(role: str, strat: str) -> str:
    return f"Pay_{role}_{strat}"

for hp in HP_LIST:
    csv_path = ROOT / f"dpr_table_hp_{hp}.csv"
    if not csv_path.exists():
        print(f"[warning] CSV file not found for HP {hp}: {csv_path}")
        continue

    df = pd.read_csv(csv_path)

    for role, strat, _ in roles_strats:
        # Determine current and deviated counts depending on the incumbent strategy
        if role == "MOBI":
            m_curr = m if strat == "MELO" else m - 1  # one MOBI player switches from MELO to CDA
            z_curr = z
            m_dev  = m_curr - 1 if strat == "MELO" else m_curr + 1  # switch to the other strategy
            z_dev  = z_curr  # ZI counts unchanged when MOBI deviates
        else:  # role == "ZI"
            m_curr = m
            z_curr = z if strat == "MELO" else z - 1  # one ZI player switches from MELO to CDA
            m_dev  = m_curr  # MOBI counts unchanged when ZI deviates
            z_dev  = z_curr - 1 if strat == "MELO" else z_curr + 1

        # Look up incumbent payoff
        mask_curr = (df['#_MOBI_MELO'] == m_curr) & (df['#_ZI_MELO'] == z_curr)
        if not mask_curr.any():
            raise ValueError(
                f"HP {hp}: missing current row for counts (m_curr={m_curr}, z_curr={z_curr})")
        curr_row = df.loc[mask_curr].iloc[0]
        curr_payoff = curr_row[payoff_col(role, strat)]

        # Look up deviated payoff
        mask_dev = (df['#_MOBI_MELO'] == m_dev) & (df['#_ZI_MELO'] == z_dev)
        if not mask_dev.any():
            raise ValueError(
                f"HP {hp}: missing deviation row for counts (m_dev={m_dev}, z_dev={z_dev})")
        dev_row = df.loc[mask_dev].iloc[0]
        other_strat = "CDA" if strat == "MELO" else "MELO"
        dev_payoff = dev_row[payoff_col(role, other_strat)]

        regret = dev_payoff - curr_payoff
        print(
            f"HP {hp:>3} | {role}-{strat} → curr={curr_payoff:.4f}, "
            f"dev_{other_strat}={dev_payoff:.4f}, regret={regret:.4g}"
        )
    print()


