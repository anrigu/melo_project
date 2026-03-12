#!/usr/bin/env python3
"""Helper script to generate selected figures from plot_payoff_vs_eq.py.
Usage:
    python analysis/run_single_fig.py 10           # only Figure 10
    python analysis/run_single_fig.py 1 5 10        # figures 1,5,10
The script simply forwards the requested numbers to plot_payoff_vs_eq.py's
--fig option and executes the module in-place (run_path).
"""

import sys, os, runpy

if len(sys.argv) < 2:
    print("Usage: python analysis/run_single_fig.py <fig numbers ...>")
    sys.exit(1)

fig_nums = [str(int(n)) for n in sys.argv[1:]]

# Build a fake argv for the target module
sys.argv = ["plot_payoff_vs_eq.py", "--fig", *fig_nums]

SCRIPT_PATH = os.path.join(os.path.dirname(__file__), "plot_payoff_vs_eq.py")
runpy.run_path(SCRIPT_PATH, run_name="__main__") 