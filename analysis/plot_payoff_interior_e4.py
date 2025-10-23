#!/usr/bin/env python3
"""Plot expected payoffs versus *all* interior equilibria (regret≤5e-4).

For every holding period it:
  • Reads equilibria files in the five seed folders.
  • Keeps only entries with regret ≤ THRESH and that are interior (both roles mix).
  • Evaluates deviation payoffs of both strategies for each seed/game against each such mixture.
  • Averages across all such payoffs (seed × mixture) and plots mean ± 1 σ.
  • Saves figure as  analysis/payoffs_vs_interior_e4_by_hp.png
"""
from __future__ import annotations
import glob, json, os, re, sys
from pathlib import Path
from collections import defaultdict
from statistics import mean, stdev
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from marketsim.egta.core.game import Game
from marketsim.game.role_symmetric_game import RoleSymmetricGame

# -------------------------------------------------------------------------
SEED_ROOTS = [
    "result_two_role_still_role_symmetric_3/ZI_arrival_6e-3_longer_sim_time_order_quantity_542",
    "result_two_role_still_role_symmetric_4/ZI_arrival_6e-3_longer_sim_time_order_quantity_543",
    "result_two_role_still_role_symmetric_5/ZI_arrival_6e-3_longer_sim_time_order_quantity_544",
    "result_two_role_still_role_symmetric_6/ZI_arrival_6e-3_longer_sim_time_order_quantity_545",
    "result_two_role_still_role_symmetric_7/ZI_arrival_6e-3_longer_sim_time_order_quantity_546",
]
THRESH = 1e-3  # regret threshold

# -------------------------------------------------------------------------
CANONICAL_ORDER = [
    ("MOBI", "MOBI_0_100_shade0_0"),
    ("MOBI", "MOBI_100_0_shade250_500"),
    ("ZI", "ZI_0_100_shade250_500"),
    ("ZI", "ZI_100_0_shade250_500"),
]

HP_RE = re.compile(r"holding_period_(\d+)")

# -------------------------------------------------------------------------

def build_game_from_raw(raw_files):
    payoff_data = []
    for rf in raw_files:
        try:
            payoff_data.extend(json.load(open(rf)))
        except Exception:
            continue
    if not payoff_data:
        return None
    return Game.from_payoff_data(payoff_data, normalize_payoffs=False)


def is_interior(mix: list[float]) -> bool:
    # Strategy order: 0=M-ELO (MOBI), 1=CDA (MOBI), 2=M-ELO (ZI), 3=CDA (ZI)
    # Interior means *CDA* probabilities (indices 1 and 3) are strictly between 0 and 1.
    return 0.0 < mix[1] < 1.0 and 0.0 < mix[3] < 1.0

# -------------------------------------------------------------------------
records = []
gain_records = []  # for relative gains
role_records = []  # average V_r(x_int) per role and HP
mix_accum = defaultdict(lambda: defaultdict(list))  # hp -> (role,strat) -> list prob

# collect hp set
hp_set = set()
for root in SEED_ROOTS:
    for eq_file in glob.glob(f"{root}/**/equilibria*.json", recursive=True):
        m = HP_RE.search(eq_file)
        if m:
            hp_set.add(int(m.group(1)))

for hp in sorted(hp_set):
    per_strategy_vals: dict[tuple[str,str], list[float]] = defaultdict(list)
    per_role_vals: dict[str, list[float]] = defaultdict(list)

    for root in SEED_ROOTS:
        # equilibria files for this hp
        eq_files = glob.glob(f"{root}/**/holding_period_{hp}/**/equilibria*.json", recursive=True)
        if not eq_files:
            eq_files = glob.glob(f"{root}/**/pilot_egta_run_{hp}_*/**/equilibria*.json", recursive=True)
        if not eq_files:
            continue

        # build game for this seed+hp from raw data
        raw_files = glob.glob(f"{root}/**/holding_period_{hp}/**/raw_payoff_data.json", recursive=True)
        if not raw_files:
            raw_files = glob.glob(f"{root}/**/pilot_egta_run_{hp}_*/**/raw_payoff_data.json", recursive=True)
        game = build_game_from_raw(raw_files)
        if game is None:
            continue

        # iterate through equilibria entries
        for eq_path in eq_files:
            try:
                data = json.load(open(eq_path))
            except Exception:
                continue
            for e in data:
                reg = float(e.get("regret", 1.0))
                mix_vec = e.get("mixture") or e.get("mixture_vector")
                if reg > THRESH or not mix_vec or not is_interior(mix_vec):
                    continue

                # align mixture to game strategies
                present_probs = []
                prob_dict_this = {}
                for role, strats in zip(game.role_names, game.strategy_names_per_role):
                    for strat in strats:
                        idx = next((i for i,(r,s) in enumerate(CANONICAL_ORDER) if r==role and s==strat), None)
                        p = mix_vec[idx] if idx is not None else 0.0
                        present_probs.append(p)
                        prob_dict_this[(role,strat)] = p

                mix = torch.tensor(present_probs, dtype=torch.float32)
                # renorm per role
                start=0
                for strats in game.strategy_names_per_role:
                    seg = mix[start:start+len(strats)]
                    s_sum = seg.sum()
                    if s_sum>1e-12:
                        mix[start:start+len(strats)] = seg/s_sum
                    else:
                        mix[start:start+len(strats)] = 1.0/len(strats)
                    start+=len(strats)

                dev = game.deviation_payoffs(mix)
                idxg=0
                for role_name,strats in zip(game.role_names, game.strategy_names_per_role):
                    seg_size = len(strats)
                    mix_seg = mix[idxg : idxg + seg_size]
                    dev_seg = dev[idxg : idxg + seg_size]
                    role_value = float((mix_seg * dev_seg).sum().item())
                    per_role_vals[role_name].append(role_value)
                    # per-strategy loop
                    for strat in strats:
                        per_strategy_vals[(role_name, strat)].append(float(dev[idxg]))
                        gain_records.append({
                            "Holding Period": hp,
                            "Role": role_name,
                            "Strategy": strat,
                            "Gain": float(dev[idxg] - role_value),
                        })
                        idxg += 1

                # accumulate mixture probabilities
                for (role,strat), prob in prob_dict_this.items():
                    mix_accum[hp][(role,strat)].append(prob)

    # aggregate per-strategy
    for (role,strat), vals in per_strategy_vals.items():
        if vals:
            records.append({
                "Holding Period": hp,
                "Role": role,
                "Strategy": strat,
                "Mean": mean(vals),
                "Std": stdev(vals) if len(vals)>1 else 0.0,
            })

    # aggregate role-average
    for role, vals in per_role_vals.items():
        if vals:
            role_records.append({
                "Holding Period": hp,
                "Role": role,
                "Mean": mean(vals),
                "Std": stdev(vals) if len(vals)>1 else 0.0,
            })

if not records:
    print("No interior-equilibrium payoffs gathered.")
    sys.exit(0)

# -------------------------------------------------------------------------
plot_df = pd.DataFrame(records)
fig, axes = plt.subplots(1, 2, figsize=(15,6), sharey=True)
role_df = pd.DataFrame(role_records)

for ax, role in zip(axes, plot_df["Role"].unique()):
    sub = plot_df[plot_df["Role"]==role]
    for strat, g in sub.groupby("Strategy"):
        g_sorted = g.sort_values("Holding Period")
        ax.plot(g_sorted["Holding Period"], g_sorted["Mean"], marker='o', label=strat)
        ax.fill_between(g_sorted["Holding Period"],
                        g_sorted["Mean"]-g_sorted["Std"],
                        g_sorted["Mean"]+g_sorted["Std"], alpha=0.2)
    # overlay role-average line
    sub_r = role_df[role_df["Role"]==role].sort_values("Holding Period")
    if not sub_r.empty:
        ax.plot(sub_r["Holding Period"], sub_r["Mean"], color='black', lw=2, marker='s', label='V_r(x_int)')
        ax.fill_between(sub_r["Holding Period"],
                        sub_r["Mean"]-sub_r["Std"],
                        sub_r["Mean"]+sub_r["Std"], color='grey', alpha=0.2)

    ax.set_title(f"{role} – Payoff vs Interior EQs (≤5e-4)")
    ax.set_xlabel("Holding Period")
    ax.set_ylabel("Expected Payoff versus x_int (mean ±1σ)")
    ax.legend()
plt.tight_layout()
out = Path(__file__).with_name("payoffs_vs_interior_e4_by_hp.png")
fig.savefig(out, dpi=300)
print("Saved interior-equilibrium payoff plot to", out.relative_to(Path.cwd()))

# --------------- Relative gain plot --------------------
if gain_records:
    gain_df = pd.DataFrame(gain_records)
    figg, axesg = plt.subplots(1,2, figsize=(15,6), sharey=True)
    for ax, role in zip(axesg, gain_df["Role"].unique()):
        sub = gain_df[gain_df["Role"]==role]
        for strat, g in sub.groupby("Strategy"):
            g_sorted = g.groupby("Holding Period")["Gain"].agg(["mean","std"]).reset_index()
            ax.plot(g_sorted["Holding Period"], g_sorted["mean"], marker='o', label=strat)
            ax.fill_between(g_sorted["Holding Period"],
                            g_sorted["mean"]-g_sorted["std"],
                            g_sorted["mean"]+g_sorted["std"], alpha=0.2)
        ax.set_title(f"{role} – Deviation Gain vs Interior EQs (≤5e-4)")
        ax.set_xlabel("Holding Period"); ax.set_ylabel("Deviation Gain (mean ±1σ)")
        ax.legend()
    plt.tight_layout()
    outg = Path(__file__).with_name("payoffs_vs_interior_gain_e4_by_hp.png")
    figg.savefig(outg, dpi=300)
    print("Saved deviation-gain plot to", outg.relative_to(Path.cwd()))

# --------------- Mixture probability plot ---------------
mix_records = []
for hp, d in mix_accum.items():
    for (role,strat), lst in d.items():
        mix_records.append({
            "Holding Period": hp,
            "Role": role,
            "Strategy": strat,
            "Prob": sum(lst)/len(lst) if lst else 0.0,
        })

if mix_records:
    mix_df = pd.DataFrame(mix_records)
    figm, axesm = plt.subplots(1,2, figsize=(15,6), sharey=True)
    for ax, role in zip(axesm, mix_df["Role"].unique()):
        sub = mix_df[mix_df["Role"]==role]
        for strat, g in sub.groupby("Strategy"):
            g_sorted = g.sort_values("Holding Period")
            ax.plot(g_sorted["Holding Period"], g_sorted["Prob"], marker='o', label=strat)
        ax.set_title(f"{role} – Avg mix prob (Interior EQs ≤5e-4)")
        ax.set_xlabel("Holding Period"); ax.set_ylabel("Probability in mixture")
        ax.set_xlim(0, 500)
        ax.legend()
    plt.tight_layout()
    outm = Path(__file__).with_name("interior_mix_probs_e4.png")
    figm.savefig(outm, dpi=300)
    print("Saved mixture-probability plot to", outm.relative_to(Path.cwd())) 