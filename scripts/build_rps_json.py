#!/usr/bin/env python3
"""Build a tiny 2-player Rock–Paper–Scissors game in GameAnalysis JSON.

The output is written to `rps2p.json` in the repo root so it can be fed to
`scripts/compare_quiesce_solvers.py` for new-vs-old solver comparison.
"""
import json
import os
from pathlib import Path

import numpy as np

# ---------------------------------------------------
# Ensure the vendored `gameanalysis` library is on path
# ---------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
GA_PARENT = ROOT / "marketsim" / "egta" / "gameanalysis-old"
if GA_PARENT.exists():
    import sys

    sys.path.insert(0, str(GA_PARENT))

from gameanalysis import rsgame, paygame  # type: ignore

# ------------------------- build the game -----------------------------
ROLE_NAMES = ["Player"]
NUM_PLAYERS = np.array([2])
STRAT_NAMES = [["P", "R", "S"]]
NUM_STRATS = 3

# Enumerate profiles for 2 players
# Order: RR, PP, SS, RP, RS, PS  (mirror profiles redundant in RSG)
PROFILES = np.array([
    [2, 0, 0],  # RR
    [0, 2, 0],  # PP
    [0, 0, 2],  # SS
    [1, 1, 0],  # RP
    [1, 0, 1],  # RS
    [0, 1, 1],  # PS
])

# Base 2-player payoff lookup (row strategy, col opponent)
BASE_PAY = np.array([
    [0, -1, 1],   # R vs (R,P,S)
    [1, 0, -1],   # P
    [-1, 1, 0],   # S
], dtype=float)

# Build PAYOFFS with zeros where the strategy isn't in the profile
PAYOFFS = np.zeros_like(PROFILES, dtype=float)
for i, prof in enumerate(PROFILES):
    for strat_idx, count in enumerate(prof):
        if count == 0:
            continue  # leave as 0
        # compute expected payoff for that strategy against the opponents
        # For two-player game, opponents_total = 2 - count (0 or 1 or 2)
        opp_pay = 0.0
        opp_cnt = 0
        for opp_idx, opp_count in enumerate(prof):
            if opp_idx == strat_idx:
                continue
            opp_pay += BASE_PAY[strat_idx, opp_idx] * opp_count
            opp_cnt += opp_count
        # average payoff vs opponents (if no opponents, it's self-play)
        PAYOFFS[i, strat_idx] = opp_pay / opp_cnt if opp_cnt > 0 else BASE_PAY[strat_idx, strat_idx]
        # Other strategies remain zero as required

# Base empty RsGame
base_game = rsgame.empty_names(ROLE_NAMES, NUM_PLAYERS, STRAT_NAMES)

# Fill with profiles & payoffs
rps_game = paygame.game_replace(base_game, PROFILES, PAYOFFS)

# Dump to JSON
outfile = ROOT / "rps2p.json"
outfile.write_text(json.dumps(rps_game.to_json(), indent=2))
print(f"✓ wrote {outfile.relative_to(ROOT)}  ({outfile.stat().st_size} bytes)") 