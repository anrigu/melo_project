"""
Generate line charts for welfare data across holding periods.

This script reads the `HP_welfares.csv` file located in the same directory and
creates three separate line charts:

1. Total welfare for each profile
2. MOBI welfare for each profile
3. Background trader (ZI) welfare for each profile

Each chart is saved as a PNG file in the same directory with a descriptive
filename. If you run this script in an interactive environment, the figures
will also be displayed.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
CSV_PATH = SCRIPT_DIR / "HP_welfares_new2.csv"


CHART_CONFIG = {
    "Total Welfare by Profile": "_total",
    "MOBI Welfare by Profile": "_mobi",
    "Background Trader Welfare by Profile": "_zi", 
}

OUTPUT_TEMPLATE = "hp_welfares_{suffix}.png"  

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
if not CSV_PATH.exists():
    raise FileNotFoundError(f"CSV file not found at {CSV_PATH}.")

# Read CSV and ignore spaces after commas
# Also coerce non-numeric values (e.g., blanks) to NaN so plotting treats them as floats.
df = pd.read_csv(CSV_PATH, skipinitialspace=True)

# Ensure all profile/welfare columns are numeric
numeric_cols = df.columns.drop("HP")
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

# Ensure the holding period column exists
if "HP" not in df.columns:
    raise ValueError("Expected a column named 'HP' representing holding periods.")

hp_values = df["HP"]

# ---------------------------------------------------------------------------
# Determine global y-axis limits for consistency across all plots
# ---------------------------------------------------------------------------
_y_min = df[numeric_cols].min().min()
_y_max = df[numeric_cols].max().max()
_margin = 0.05 * (_y_max - _y_min) if _y_max != _y_min else 1.0
_Y_LOWER = _y_min - _margin
_Y_UPPER = _y_max + _margin

# ---------------------------------------------------------------------------
# Plotting helper
# ---------------------------------------------------------------------------
def plot_category(title: str, suffix: str) -> None:
    """Plot all columns ending with *suffix* against the holding period."""
    # Identify relevant columns
    cols = [col for col in df.columns if col.endswith(suffix)]
    if not cols:
        print(f"No columns found for suffix '{suffix}'. Skipping plot '{title}'.")
        return

    plt.figure(figsize=(10, 6))
    for col in cols:
        label = col[:-len(suffix)]
        plt.plot(hp_values, df[col], marker="o", label=label)

    plt.gca().set_title("")  # Explicitly clear any existing title
    plt.xlabel("Holding Period", fontweight="bold", fontsize=18)
    plt.ylabel("Welfare", fontweight="bold", fontsize=18)

    # Apply consistent y-axis limits across all charts
    plt.ylim(_Y_LOWER, _Y_UPPER)

    xticks = np.arange(0, hp_values.max() + 20, 20)
    plt.xticks(xticks, fontsize=16, fontweight="bold", rotation=-30)

    plt.axvline(x=40, color="black", linestyle=(0, (8, 8)), linewidth=2, zorder=4)

    plt.tick_params(axis='both', labelsize=16)
    plt.setp(plt.gca().get_yticklabels(), fontweight='bold')

    # Highlight the 1_0_0.097_0.903 profile at HP=140 in purple
    highlight_prefix = "1_0_0.097_0.903"
    highlight_col = f"{highlight_prefix}{suffix}"
    if highlight_col in df.columns and 140 in hp_values.values:
        idx_140 = df.index[hp_values == 140][0]
        print(idx_140)
        y_val = df.loc[idx_140, highlight_col]
        if pd.notna(y_val):
            plt.scatter(140, y_val, color="purple", s=80, zorder=5, edgecolors="black")

    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    # Save figure
    suffix_clean = suffix.lstrip('_')
    output_path = SCRIPT_DIR / OUTPUT_TEMPLATE.format(suffix=suffix_clean)
    plt.savefig(output_path, dpi=300)
    print(f"Saved: {output_path.name}")

    try:
        import matplotlib_inline.backend_inline
        plt.show()
    except ImportError:
        plt.close()


# ---------------------------------------------------------------------------
# Generate all charts
# ---------------------------------------------------------------------------
def main():
    for title, suffix in CHART_CONFIG.items():
        plot_category(title, suffix)


if __name__ == "__main__":
    main()