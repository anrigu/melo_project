#!/usr/bin/env python3
"""equilibrium_deviation_payoffs.py

For a list of holding-period (HP) values compute the deviation-payoff
vector at the **DPR equilibrium** (4×4 reduced game) and visualise it.

Two outputs are produced in the current directory:

1. `deviation_payoffs.csv` – tidy table with one row per (HP, Role,
   Strategy, Payoff).
2. `deviation_payoffs_plot.png` – grouped bar plot showing, for each HP,
   the expected payoff V_r(x̂) of the equilibrium **and** the individual
   deviation payoffs (MELO vs CDA) for both roles.  This lets you see at
   a glance which strategy has the advantage inside the equilibrium and
   how that changes across HP.

Usage examples
--------------
    python scripts/equilibrium_deviation_payoffs.py       # default HP grid 0..300 step 20
    python scripts/equilibrium_deviation_payoffs.py --hp 120 140 160  # custom set
"""
from __future__ import annotations

import argparse, itertools, sys
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")  # headless by default
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# import heavy helpers AFTER path tweak
from analysis.plot_bootstrap_compare import (
    find_pooled_equilibrium_dpr, collect_profiles_by_seed, build_game_from_profiles, IDX_CAN_MAP
)

# ---------------------------------------------------------------------------
# helper: expected payoff to each role under mixture (deviation payoffs)
# ---------------------------------------------------------------------------

def role_payoffs(game, mix_vec: torch.Tensor) -> List[float]:
    dv = game.deviation_payoffs(mix_vec)
    vals = []
    idx = 0
    for n in game.num_strategies_per_role:
        seg = slice(idx, idx + n)
        vals.append(float((mix_vec[seg] * dv[seg]).sum()))
        idx += n
    return vals

# ---------------------------------------------------------------------------
# main logic
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Collect deviation payoffs at DPR equilibria.")
    ap.add_argument("--hp", type=int, nargs="*", help="specific HP values (default 0..300 step 20)")
    ap.add_argument("--step", type=int, default=20, help="grid step if --hp omitted")
    ap.add_argument("--n_red", type=int, default=4, help="players per role in DPR reduction")
    args = ap.parse_args()

    if args.hp:
        hp_vals = sorted(set(args.hp))
    else:
        hp_vals = list(range(0, 301, args.step))

    rows = []

    for hp in hp_vals:
        print(f"HP {hp:>3} …", end="", flush=True)
        try:
            mix_can = find_pooled_equilibrium_dpr(hp, n_red=args.n_red, print_only=True)
        except Exception as exc:
            print("skip (", exc, ")")
            continue
        # rebuild full reduced game to compute per-strategy payoffs
        profiles = list(itertools.chain.from_iterable(collect_profiles_by_seed(hp).values()))
        game_full = build_game_from_profiles(profiles)
        if game_full is None:
            print("skip (no game)")
            continue

        # replicate DPR reduction logic
        from gameanalysis import paygame
        from gameanalysis.reduction import deviation_preserving
        role_names = game_full.role_names
        num_players = [int(x) for x in game_full.num_players_per_role]
        strat_names = game_full.strategy_names_per_role
        cfg = game_full.game.rsg_config_table.cpu().numpy().astype(int)
        pay = game_full.game.rsg_payoff_table.cpu().numpy().T
        pay[cfg == 0] = 0.0
        ga_full = paygame.game_names(role_names, num_players, strat_names, cfg, pay)
        ga_red = deviation_preserving.reduce_game(ga_full, [args.n_red]*len(role_names))

        # wrap into RSG helper for convenient deviation_payoffs
        from marketsim.game.role_symmetric_game import RoleSymmetricGame
        cfg_t = torch.tensor(ga_red.profiles(), dtype=torch.float32)
        pay_t = torch.tensor(ga_red.payoffs().T, dtype=torch.float32)
        game_red = RoleSymmetricGame(role_names, [args.n_red]*len(role_names), strat_names, cfg_t, pay_t)

        # map canonical 4-entry mix into game order
        mix_vec = []
        for role, strats in zip(role_names, strat_names):
            for strat in strats:
                mix_vec.append(float(mix_can[IDX_CAN_MAP[(role, strat)]]))
        mix_t = torch.tensor(mix_vec, dtype=torch.float32)
        # renormalise per role
        start = 0
        for n in [2, 2]:
            seg = slice(start, start+n)
            mix_t[seg] /= mix_t[seg].sum(); start += n

        # record per-role payoff and individual deviation payoffs
        dev = game_red.deviation_payoffs(mix_t)
        idx = 0
        for role, strats in zip(role_names, strat_names):
            seg = slice(idx, idx+len(strats))
            V_r = float((mix_t[seg] * dev[seg]).sum())
            rows.append({"HP": hp, "Role": role, "Strategy": "V_role", "Payoff": V_r})
            for j, strat in enumerate(strats):
                rows.append({"HP": hp, "Role": role, "Strategy": strat, "Payoff": float(dev[idx+j])})
            idx += len(strats)
        print("done")

    if not rows:
        print("No data collected – aborting")
        return

    df = pd.DataFrame(rows)
    out_csv = Path("deviation_payoffs.csv")
    df.to_csv(out_csv, index=False)
    print("Saved", out_csv)

    # simple grouped bar plot (per role)
    for role in df["Role"].unique():
        sub = df[(df["Role"] == role) & (df["Strategy"].isin(["V_role", *df[df["Role"]==role]["Strategy"].unique()]))]
        hp_sorted = sorted(sub["HP"].unique())
        width = 0.25
        x = np.arange(len(hp_sorted))

        fig, ax = plt.subplots(figsize=(10,4))
        bar_pos = {
            "V_role": x - width,
            strat_names[0][0 if role=="MOBI" else 0]: x,
            strat_names[0][1 if role=="MOBI" else 1]: x + width
        }
        labels_plotted = set()
        for strat, grp in sub.groupby("Strategy"):
            grp = grp.sort_values("HP")
            hp_idx = [hp_sorted.index(h) for h in grp["HP"]]
            ax.bar(bar_pos.get(strat, x), grp["Payoff"], width=width, label=strat)
            labels_plotted.add(strat)

        ax.set_xticks(x)
        ax.set_xticklabels(hp_sorted, rotation=45)
        ax.set_ylabel("Deviation payoff")
        ax.set_xlabel("Holding Period")
        ax.set_title(f"Deviation payoffs – {role}")
        ax.legend()
        plt.tight_layout()
        out_png = Path(f"deviation_payoffs_{role.lower()}.png")
        plt.savefig(out_png, dpi=150)
        plt.close()
        print("Saved", out_png)


if __name__ == "__main__":
    main()
