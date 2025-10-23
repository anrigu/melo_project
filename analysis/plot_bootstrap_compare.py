#!/usr/bin/env python3
"""plot_bootstrap_compare.py

Generate ONLY the two bootstrap comparison figures:
    1. compare_pooled_bootstrap_all_lines_by_hp.png     – replicate-specific EQ (old Fig 9)
    2. compare_pooled_bootstrap_fixed_eq_all_lines_by_hp.png – fixed pooled EQ (old Fig 10)

Both figures are saved next to this script.

The script re-implements the minimum logic needed, avoiding the massive
`plot_payoff_vs_eq.py`.  Run it with:

    python analysis/plot_bootstrap_compare.py         # all HP values
    python analysis/plot_bootstrap_compare.py --hp 50 100  # subset of HPs

"""
from __future__ import annotations

import argparse, glob, json, itertools, os, sys, re, time
from tqdm.auto import tqdm
from collections import defaultdict, Counter
from pathlib import Path


from matplotlib.ticker import MaxNLocator
import matplotlib 
matplotlib.use('Agg')    # or 'Qt6Agg' if you have Qt installed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import gaussian_filter1d
from math import comb           # for multinomial coefficients


ROOT = Path(__file__).resolve().parent.parent 
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "marketsim" / "egta" / "gameanalysis-old"))

from gameanalysis import paygame, nash as ga_nash, regret as ga_regret
from gameanalysis.reduction import deviation_preserving



import gameanalysis.utils as _ga_utils
try:
    from scipy.special import comb as _sp_comb 
    _ga_utils.comb = _sp_comb 
except Exception:
    import math

    def _fallback_comb(n, k, exact=False): 
        """math.comb wrapper that ignores *exact* arg expected by SciPy API."""
        return math.comb(n, k)

    _ga_utils.comb = _fallback_comb  

from marketsim.egta.core.game import Game
from marketsim.egta.solvers.equilibria import replicator_dynamics as ms_rd
from marketsim.game.role_symmetric_game import RoleSymmetricGame
from marketsim.egta.simulators.melo_wrapper import MeloSimulator

SEED_ROOTS = [
    "result_two_role_still_role_symmetric_3",
    "result_two_role_still_role_symmetric_3_20k",
    "result_two_role_still_role_symmetric_4",
    "result_two_role_still_role_symmetric_4_20k",
    "result_two_role_still_role_symmetric_5",
    "result_two_role_still_role_symmetric_5_20k",
    "result_two_role_still_role_symmetric_6",
    "result_two_role_still_role_symmetric_6_20k", 
    "result_two_role_still_role_symmetric_7",
   # "result_two_role_still_role_symmetric_7_20k",
    # "result_two_role_still_role__1_hbl_symmetric_3",
    # "result_two_role_still_role__1_hbl_symmetric_4", clear
    # "result_two_role_still_role__1_hbl_symmetric_5",
    # "result_two_role_still_role__1_hbl_symmetric_6",
    # "result_two_role_still_role__1_hbl_symmetric_7", 
    # One-role experiments
#    "result_one_role_still_role_symmetric_3",
#    "result_one_role_still_role_symmetric_4",
#    "result_one_role_still_role_symmetric_5",
#    "result_one_role_still_role_symmetric_6",
#     "result_one_role_still_role_symmetric_7",

]
'''
SEED_ROOTS = [
    "result_two_role_still_role__1_hbl_symmetric_3",
    "result_two_role_still_role__1_hbl_symmetric_4",
    "result_two_role_still_role__1_hbl_symmetric_5",
    "result_two_role_still_role__1_hbl_symmetric_6",
    "result_two_role_still_role__1_hbl_symmetric_7",
]
'''

CANONICAL_ORDER = [
    ("MOBI", "MOBI_0_100_shade0_0"),
    ("MOBI", "MOBI_100_0_shade250_500"),
    ("ZI",   "ZI_0_100_shade250_500"),
    ("ZI",   "ZI_100_0_shade250_500"),
]

GAPFILL_ROOT = "gapfill_profiles2"
os.makedirs(GAPFILL_ROOT, exist_ok=True)

if GAPFILL_ROOT not in SEED_ROOTS:
    SEED_ROOTS.append(GAPFILL_ROOT)

GAPFILL_ROOT = "gapfill_profiles"
os.makedirs(GAPFILL_ROOT, exist_ok=True)

if GAPFILL_ROOT not in SEED_ROOTS:
    SEED_ROOTS.append(GAPFILL_ROOT)

GAPFILL_ROOT = "gapfill_profiles3"
os.makedirs(GAPFILL_ROOT, exist_ok=True)

if GAPFILL_ROOT not in SEED_ROOTS:
    SEED_ROOTS.append(GAPFILL_ROOT)

GAPFILL_ROOT = "gapfill_profiles6"
if GAPFILL_ROOT not in SEED_ROOTS:
    SEED_ROOTS.append(GAPFILL_ROOT)

GAPFILL_ROOT = "gapfill_profiles_tester"
if GAPFILL_ROOT not in SEED_ROOTS:
    SEED_ROOTS.append(GAPFILL_ROOT)







from pathlib import Path as _Path  

def _abs_path(p: str) -> str:
    """Return absolute path for *p* (relative paths are resolved against
    the project root *ROOT*)."""
    pp = _Path(p)
    return str((ROOT / pp).resolve()) if not pp.is_absolute() else str(pp.resolve())

SEED_ROOTS = list(dict.fromkeys(_abs_path(p) for p in SEED_ROOTS))  

# Normalise the active GAPFILL_ROOT for subsequent gap-fill writes
GAPFILL_ROOT = _abs_path(GAPFILL_ROOT)

# Mapping (role,strategy) → canonical index for fast lookup
IDX_CAN_MAP = {pair: i for i, pair in enumerate(CANONICAL_ORDER)}

# Will collect one record per HP containing the DPR equilibrium mixture
# and its expected payoff per role.  Written to CSV at program end.
DPR_EQ_ROWS: list[dict[str, float]] = []

N_BOOT = 100
STOP_REG = 1e-4
SMOOTH_CI = False
USE_SMOOTHING = False

# epsilon threshold for RSNE validity
EPS_RSNE = 1e-4

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def collect_profiles_by_seed(hp: int) -> dict[str, list]:
    """Return mapping seed_root -> list[profiles] for one HP."""
    seed_to_profiles: dict[str, list] = {}
    for root in SEED_ROOTS:
        profs = []
        p1 = glob.glob(f"{root}/**/holding_period_{hp}/**/raw_payoff_data.json", recursive=True)
        p2 = glob.glob(f"{root}/**/pilot_egta_run_{hp}_*/**/raw_payoff_data.json", recursive=True)
        for rf in p1 + p2:
            try:
                profs.extend(json.load(open(rf)))
            except Exception:
                continue
        if profs:
            seed_to_profiles[root] = profs
        if hp == 300 or hp == 320:
            print(f"Number of Profiles is: {len(profs)}") 
    return seed_to_profiles

# ------------------------------------------------------------------
# Bootstrap helper – draw until we have a *complete* 2×2 game
# ------------------------------------------------------------------
def resample_complete_game(
    hp: int,
    rng: np.random.Generator,
    *,
    max_attempts: int = 1_000,
):
    """
    Re-draw bootstrap samples until Game keeps both strategies for
    every role *and* its payoff table is fully populated (no NaNs).

    Parameters
    ----------
    hp : int
        Holding period whose profiles we are sampling.
    rng : np.random.Generator
        Random generator used for bootstrap sampling.
    max_attempts : int, default 1000
        How many times to try before giving up and returning (None, None).

    Returns
    -------
    game : Game | None
        A fully populated 2 × 2 RoleSymmetricGame, or None if we gave up.
    profs : list | None
        The list of profiles that generated `game`, or None on failure.
    """
    for _ in range(max_attempts):
        profs = draw_profiles_full_support(hp, rng)
        if not profs:
            continue

        game = build_game_from_profiles(profs)
        if game is None:
            continue

        if any(len(strats) < 2 for strats in game.strategy_names_per_role):
            continue

        if torch.isnan(game.game.rsg_payoff_table).any():
            continue

        return game, profs

    return None, None

def build_game_from_profiles(profiles: list) -> Game | None:
  

    cleaned: list = []

    for prof in profiles:
        if not isinstance(prof, (list, tuple)) or not prof:
            continue  

        new_prof = []
        for idx, row in enumerate(prof):
            if isinstance(row, (list, tuple)) and 3 <= len(row) <= 4:
                if len(row) == 3:
                    pid, strat, payoff = row
                    new_prof.append((pid, "Player", strat, payoff))
                else:
                    new_prof.append(tuple(row[:4]))
            elif isinstance(row, dict):
                role = row.get("role") or row.get("Role")
                strat = row.get("strategy") or row.get("Strategy")
                payoff = row.get("payoff") or row.get("Payoff")
                if role and strat and payoff is not None:
                    new_prof.append((f"p{idx}", role, strat, float(payoff)))

        if new_prof:
            cleaned.append(new_prof)

    if not cleaned:
        return None

    try:
        return Game.from_payoff_data(cleaned, normalize_payoffs=False)
    except Exception as exc:
        _log(f"[warning] Game.from_payoff_data failed: {exc}")
        return None


def draw_profiles_full_support(hp: int, rng: np.random.Generator, max_retry: int = 500):
    seed_map = collect_profiles_by_seed(hp)

    master = list(itertools.chain.from_iterable(seed_map.values()))
    if not master:
        return []
    n_tot = len(master)
    for _ in range(max_retry):
        sample = [master[i] for i in rng.integers(0, n_tot, size=n_tot)]
        support = set()
        for rec in sample:
            if isinstance(rec, (list, tuple)) and len(rec) >= 3:
                role, strat = rec[1], rec[2]
            elif isinstance(rec, dict):
                role = rec.get("role") or rec.get("Role")
                strat = rec.get("strategy") or rec.get("Strategy")
            else:
                continue
            if isinstance(role, str) and isinstance(strat, str):
                support.add((role, strat))
        if all(t in support for t in CANONICAL_ORDER):
            return sample
    return master 

def _social_welfare(game: Game, mix: torch.Tensor) -> float:
    
   
    try:
        # Fast path – obtain baseline payoff *per strategy* and apply mixture weights
        base_vals = game.game.mixture_values(mix)  # type: ignore[attr-defined]
        start, sw = 0, 0.0
        for strat_list in game.strategy_names_per_role:
            n = len(strat_list)
            seg = slice(start, start + n)
            sw += float((mix[seg] * base_vals[seg]).sum())
            start += n
        return sw
    except Exception:
        # Fallback – use deviation payoffs (already probability-weighted once
        # we multiply by *mix*).
        dev = game.deviation_payoffs(mix)
        idx, sw = 0, 0.0
        for strat_list in game.strategy_names_per_role:
            n = len(strat_list)
            seg = slice(idx, idx + n)
            sw += float((mix[seg] * dev[seg]).sum())
            idx += n
        return sw


# New helper: multinomial welfare expectation

def welfare_multinomial(ga_game, mix_vec):
    
   
    
    mix_np = mix_vec.detach().cpu().numpy() if hasattr(mix_vec, "detach") else np.asarray(mix_vec, dtype=float)
    probs_per_role = []
    idx = 0
    if hasattr(ga_game, "num_role_strats"):
        role_sizes = [int(n) for n in ga_game.num_role_strats]
    elif hasattr(ga_game, "strategy_names_per_role"):
        role_sizes = [len(lst) for lst in ga_game.strategy_names_per_role]
    elif hasattr(ga_game, "role_starts") and hasattr(ga_game, "num_strats"):
        # Compute sizes from start indices
        starts = list(map(int, ga_game.role_starts))
        starts.append(int(getattr(ga_game, "num_strats")))
        role_sizes = [starts[i+1] - starts[i] for i in range(len(starts)-1)]
    else:
        raise AttributeError("Cannot infer role sizes for game object passed to welfare_multinomial")
    for n in role_sizes:
        probs_per_role.append(mix_np[idx : idx + int(n)])
        idx += int(n)
    total_welfare = 0.0
    # Obtain iterable of (counts, payoff_vec)
    if hasattr(ga_game, "profiles") and hasattr(ga_game, "payoffs"):
        profile_iter = zip(ga_game.profiles(), ga_game.payoffs())
    elif hasattr(ga_game, "game") and hasattr(ga_game.game, "profiles"):
        profile_iter = zip(ga_game.game.profiles(), ga_game.game.payoffs())
    elif hasattr(ga_game, "rsg_config_table") and ga_game.rsg_config_table is not None:
        cfg = ga_game.rsg_config_table.cpu().numpy()
        pay = ga_game.rsg_payoff_table.cpu().numpy().T  # shape (n_cfg, n_strats)
        profile_iter = zip(cfg, pay)
    elif hasattr(ga_game, "game"):
        inner = ga_game.game
        if hasattr(inner, "profiles"):
            profile_iter = zip(inner.profiles(), inner.payoffs())
        elif hasattr(inner, "rsg_config_table") and inner.rsg_config_table is not None:
            cfg = inner.rsg_config_table.cpu().numpy()
            pay = inner.rsg_payoff_table.cpu().numpy().T
            profile_iter = zip(cfg, pay)
        else:
            raise AttributeError("Wrapped game object lacks profiles/payoffs for welfare_multinomial")
    else:
        raise AttributeError("Game object lacks profiles/payoffs accessors required for welfare_multinomial")

    for counts, payoff_vec in profile_iter:
        prob = 1.0
        idx = 0
        for r, n in enumerate(role_sizes):
            counts_r = counts[idx : idx + n]
            p_r      = probs_per_role[r]
            coeff = 1
            role_players = int(np.sum(counts_r))
            rest = role_players
            for k in counts_r:
                k_int = int(k)
                coeff *= comb(rest, k_int)
                rest  -= k_int
            prob_r = coeff * np.prod(p_r ** counts_r)
            prob  *= prob_r
            idx   += int(n)
        total_payoff_profile = np.dot(payoff_vec, counts)
        total_welfare += prob * total_payoff_profile
    return float(total_welfare)


def _is_duplicate(a: torch.Tensor, b: torch.Tensor, tol: float = 0.01) -> bool:
    """Return *True* if mixtures *a* and *b* are indistinguishable.

    We test the *per-role* L∞ distance w.r.t. the canonical two-by-two
    ordering (indices 0–1 = MOBI, 2–3 = ZI).  For our purposes that is
    equivalent to the plain vector L∞ distance because each role spans
    exactly two consecutive indices.
    """
    return bool(torch.max(torch.abs(a - b)) <= tol)


def find_pooled_equilibrium(
    hp: int,
    *,
    n_start: int = 30,
    iters: int = 1000,
    converge_threshold: float = 1e-7,
    dup_tol: float = 0.01,
    regret_max: float = 1e-4,
) -> torch.Tensor:
    """Compute a *fixed* equilibrium for the given holding period.

    Steps
    ------
    1.  Pool **all** payoff data across seeds for this *hp* and build the
        empirical game.
    2.  Run multi-population replicator dynamics from ``n_start`` random
        initial mixtures (each role independently uniform-random).
    3.  Deduplicate candidate equilibria using an L∞ distance threshold
        of ``dup_tol`` (per role).
    4.  Select the candidate with **median social welfare**.

    The result is converted to a length-4 tensor in *canonical* order
    (MOBI_A, MOBI_CDA, ZI_A, ZI_CDA) and renormalised per role.
    """
  
    candidates: list[torch.Tensor] = []
    welfare_vals: list[float] = []
    raw_files: list[str] = []
    for root in SEED_ROOTS:
        raw_files.extend(glob.glob(f"{root}/**/holding_period_{hp}/**/raw_payoff_data.json", recursive=True))
        raw_files.extend(glob.glob(f"{root}/**/pilot_egta_run_{hp}_*/**/raw_payoff_data.json", recursive=True))

    all_profiles: list = []
    for fp in raw_files:
        try:
            payload = json.load(open(fp))
            if payload:
                all_profiles.extend(payload)
        except Exception as exc:
            _log(f"[warning] Skipping corrupt payoff file {fp}: {exc}")
            continue

    game = None
    if all_profiles:
        try:
            game = build_game_from_profiles(all_profiles)
        except Exception as exc:
            _log(f"[warning] Could not build pooled game for HP={hp}: {exc}")

    if game is None:
        raise RuntimeError(f"Cannot build pooled game for HP={hp}")

    if all(len(lst) == 2 for lst in game.strategy_names_per_role):
        try:
            mix, _ = min(game.find_nash_equilibrium_2x2(), key=lambda x: x[1])
            print(game.role_names, game.strategy_names_per_role)
            print(f"regret =  {game.regret(mix)} for mix {mix}")
            print()
            if game.regret(mix) <= regret_max:
                candidates.append(_normalise_per_role(mix.clone()))
                welfare_vals.append(welfare_multinomial(game, mix))
        except Exception:
            pass 

    
    for _ in range(n_start): #rd from all the start 
        print(f"Running RD for iteration {_} in HP {hp}")
        # ----- restrict to observed columns ---------------------------------
        keep_idx = [
            s for s in range(game.num_strategies)
            if not torch.isnan(game.game.rsg_payoff_table[s]).all()
        ]

        # ensure at least one strategy per role survives
        role_ok = True
        for r_start, n in zip(game.role_starts, [2, 2]):
            if not any(i in keep_idx for i in range(int(r_start), int(r_start + n))):
                role_ok = False; break

        game_r = game.restrict(keep_idx) if role_ok else game

        # fresh random start in the *restricted* space
        init = _normalise_per_role(torch.rand(game_r.num_strategies))

        try:
            mix = replicator_dynamics(
                game_r,
                mixture=init,
                iters=iters,
                #num_random_starts=n_start,
                converge_threshold=converge_threshold,
                use_multiple_starts=True,
            )
            print(f"regret =  {game.regret(mix)} for mix {mix}")
        except Exception:
            continue

        # evaluate regret **in the restricted game** (this ignores missing columns)
        if game_r.regret(mix) > regret_max:
            continue

        # --- expand back to full-length canonical vector ---
        #full_mix = torch.zeros(game.num_strategies)
        #full_mix[torch.tensor(keep_idx)] = mix
       # mix = _normalise_per_role(full_mix.clone())

        # Deduplicate using per-role tolerance
       
        if any(_is_duplicate(mix, prev, dup_tol) for prev in candidates):
            continue

        candidates.append(mix)
        welfare_vals.append(welfare_multinomial(game, mix))

    
        idx_median = len(candidates) // 2
        mix = [m for _, m in sorted(zip(welfare_vals, candidates))][idx_median]

    # --------------------------------------------------------------

    out = [0.0] * 4
    idx_game = 0
    for role, strats in zip(game.role_names, game.strategy_names_per_role):
        for strat in strats:
            idx_can = next(i for i, (r, s) in enumerate(CANONICAL_ORDER) if r == role and s == strat)
            out[idx_can] = float(mix[idx_game])
            idx_game += 1

    mobi_sum = out[0] + out[1]
    zi_sum = out[2] + out[3]
    out[0] /= mobi_sum; out[1] /= mobi_sum
    out[2] /= zi_sum; out[3] /= zi_sum


    print(f"Equilibrium for {hp} is {out}")
    return torch.tensor(out)


def _normalise_per_role(vec: torch.Tensor) -> torch.Tensor:
    start = 0
    role_sizes = [2, 2]  # always 2×2 here
    for n in role_sizes:
        seg = vec[start:start + n]
        seg /= seg.sum()
        vec[start:start + n] = seg
        start += n
    return vec

def _counts_to_profile(counts: np.ndarray, role_names: list[str], strat_names_per_role: list[list[str]]):
    """Convert strategy-count array to explicit profile list for simulation."""
    profile = []
    idx = 0
    for role, strats in zip(role_names, strat_names_per_role):
        for strat in strats:
            c = int(counts[idx])
            profile.extend([(role, strat)] * c)
            idx += 1
    return profile


def _simulate_profiles_for_counts(
    hp: int,
    profiles_counts: np.ndarray,
    role_names: list[str],
    num_players_per_role: list[int],
    strat_names_per_role: list[list[str]],
):
    # NEW: force correct shape
    profiles_counts = np.asarray(profiles_counts, dtype=int).reshape(-1, 4)
    n_workers = int(os.environ.get("SLURM_CPUS_ON_NODE", 8))
    sim = MeloSimulator(
        num_strategic_mobi=num_players_per_role[0],
        num_strategic_zi=num_players_per_role[1],
        sim_time=10000,
        lam=6e-3,
        mean=1e6,
        r=0.001,
        shock_var=1e6,
        q_max=10,
        pv_var=5000000,
        shade=[250, 500], #same shading both agents
        holding_period=hp,
        mobi_strategies=strat_names_per_role[0],
        zi_strategies=strat_names_per_role[1],
        reps=10000,
        num_workers=n_workers,
        parallel=True,
        log_profile_details=True,
    )

    legacy_rows_all = []
    iterator = tqdm(profiles_counts, desc=f"Simulating gap profiles HP {hp}", unit="prof") if VERBOSE else profiles_counts
    for cnts in iterator:
        cnts_arr = np.asarray(cnts, dtype=int).flatten()
        if cnts_arr.size != 4:
            _log("[gapfill] skipping malformed counts", cnts)
            continue
        _log("[gapfill] Simulating missing full profile:", cnts_arr.tolist())
        prof_list = _counts_to_profile(cnts_arr, role_names, strat_names_per_role)
        obs = sim.simulate_profile(prof_list)
        legacy_rows = []
        # Observation.payoffs are ordered per player in the *original* profile list
        for idx, ((role, strat), payoff) in enumerate(zip(prof_list, obs.payoffs)):
            legacy_rows.append((f"p{idx}", role, strat, float(payoff)))
        legacy_rows_all.append(legacy_rows)
        _log("[gapfill]    → collected payoffs", [round(float(p),2) for p in obs.payoffs])
    return legacy_rows_all


def _fill_dpr_missing_payoffs(
    hp: int,
    ga_full,
    ga_red_full,
    game_full_ms,
    role_names,
    num_players_full,
    strat_names_per_role,
    *,
    print_only: bool = False,
    n_red: int = 4,
):
    """Identify NaNs in reduced game, simulate missing full profiles, and update the master game.

    Returns True iff new data were added (i.e., another reduction pass may help)."""
    import os
    if print_only or os.getenv("DPR_PRINT_ONLY"):
        return False  # skip simulation entirely

    red_profiles = ga_red_full.profiles()
    red_payoffs = ga_red_full.payoffs()

    # 1) rows that still contain any NaN deviation payoff
    nan_rows = np.isnan(red_payoffs).any(axis=1)
    missing_red_list: list[list[int]] = red_profiles[nan_rows].tolist()

    # 2) baseline rows that are *completely absent*  (only n_red+1 for 2 roles)
    if len(role_names) == 2:
        expected_baselines = n_red + 1  # 0..n_red MELO counts for MOBI implies same for ZI
    else:
        from math import comb as _comb
        expected_baselines = _comb(n_red + len(role_names) - 1, len(role_names) - 1)

    if ga_red_full.num_profiles < expected_baselines:
        present = {tuple(p.tolist()) for p in red_profiles}
        for mobi_a in range(n_red + 1):
            for zi_a in range(n_red + 1):
                prof = [mobi_a, n_red - mobi_a, zi_a, n_red - zi_a]
                if tuple(prof) not in present:
                    missing_red_list.append(prof)

    # --- deduplicate so each reduced profile appears once -----------
    missing_red_list = list({tuple(p) for p in missing_red_list})
 
    if not missing_red_list:
        return False
    missing_red = np.asarray(missing_red_list, dtype=int)

    _log(f"[gapfill] {missing_red.shape[0]} reduced profiles have missing payoffs – expanding …")

    # Expand to required full-game profiles (baselines + deviations)
    full_needed = deviation_preserving.expand_profiles(ga_full, missing_red)

    # Determine which of these full profiles still have ANY NaN payoff entry
    pay_full = ga_full.get_payoffs(full_needed)

    # Baseline rows completely absent (all NaNs)
    baseline_mask = np.all(np.isnan(pay_full), axis=1)

    # Rows present but with at least one NaN value
    mask_missing_full = np.isnan(pay_full).any(axis=1)

    # NEW: rows whose payoffs are all zeros (this happens when the profile
    #      has not been simulated and earlier code zero-filled NaNs). Treat
    #      those exactly like missing baselines so they will be scheduled
    #      for simulation.
    mask_all_zero = np.all(pay_full == 0, axis=1)

    # Profiles to simulate: missing baseline rows, zero rows, or rows with partial NaNs
    mask_need_sim = baseline_mask | mask_all_zero | mask_missing_full

    full_to_sim = full_needed[mask_need_sim]

    _log(f"[gapfill] → {full_to_sim.shape[0]} unique 68-player profiles still missing payoffs (including baselines)")

    if full_to_sim.size == 0:
        return False

    # Simulate and update (baselines + deviations)
    legacy_rows_nested = _simulate_profiles_for_counts(
        hp,
        full_to_sim,
        role_names,
        num_players_full,
        strat_names_per_role,
    )
    # Pass the nested list (one entry per simulated profile) to the RSG helper
    # so it can correctly interpret each profile as a separate observation.
    game_full_ms.game.update_with_new_data(legacy_rows_nested, normalize_payoffs=False)

    # ------------------------------------------------------------------
    # Persist the newly simulated profiles so future runs can reuse them.
    #
    # IMPORTANT: we store **one list per 68-agent profile** (nested list),
    # mirroring the structure expected by `build_game_from_profiles`.
    # Flattening the rows (old behaviour) meant a subsequent run could not
    # reconstruct the profile structure and therefore silently ignored the
    # freshly simulated data.
    # ------------------------------------------------------------------

    gap_dir = os.path.join(GAPFILL_ROOT, f"holding_period_{hp}")
    os.makedirs(gap_dir, exist_ok=True)
    out_path = os.path.join(gap_dir, "raw_payoff_data.json")

    try:
        if os.path.exists(out_path):
            existing_raw = json.load(open(out_path))

            # keep only *nested* profiles (first element is itself list / tuple / dict)
            def _is_nested(prof):
                return bool(prof) and isinstance(prof[0], (list, tuple, dict))

            existing_nested = [prof for prof in existing_raw if _is_nested(prof)]
            # de-dup at profile granularity
            seen = {tuple(map(tuple, p)) for p in existing_nested}
            for prof in legacy_rows_nested:
                key = tuple(map(tuple, prof))
                if key not in seen:
                    existing_nested.append(prof)
                    seen.add(key)
            payload_to_write = existing_nested
        else:
            payload_to_write = legacy_rows_nested

        with open(out_path, "w") as f:
            json.dump(payload_to_write, f, indent=2)
    except Exception as exc:
        print(f"[warning] could not write gap-fill data to {out_path}: {exc}")

    _log("[gapfill] Added", len(legacy_rows_nested), "profiles to RoleSymmetricGame table and persisted to disk →", out_path)

    return True

# ------------------------------------------------------------------
# Seed-level bootstrap helper
# ------------------------------------------------------------------
def sample_seed_bootstrap(hp: int, rng: np.random.Generator, seed_to_profiles: dict[str, list]) -> list:
    """Concatenate profile lists from a bootstrap resample of seeds.

    Each replicate draws *len(seeds)* seeds with replacement and
    concatenates all their profiles, preserving within-seed correlation.
    Returns an empty list if *seed_to_profiles* is empty.
    """
    if not seed_to_profiles:
        return []
    seeds = list(seed_to_profiles.keys())
    draws = rng.integers(0, len(seeds), size=len(seeds))
    profs: list = []
    for idx in draws:
        profs.extend(seed_to_profiles[seeds[idx]])
    return profs

# ------------------------------------------------------------------
# Enhanced bootstrap sampling: ensure both strategies per role present
# ------------------------------------------------------------------
# (Keeping the existing _sample_profiles_full_cols for optional future use.)

# ------------------------------------------------------------------
# Bootstrap loops
# ------------------------------------------------------------------

# ------------------------------------------------------------------
# DPR equilibrium helper  (n_red = 4 players per role)  NEW
# ------------------------------------------------------------------

def find_pooled_equilibrium_dpr(hp: int, *, n_red: int = 4, print_only: bool = False, **kwargs) -> torch.Tensor:
    """Return a 4-entry canonical equilibrium mixture for the *true* DPR game.

    Steps
    -----
    1. Aggregate *all* payoff data for the requested holding-period and build a
       full `gameanalysis` paygame.
    2. Apply deviation-preserving reduction to `red_players = [n_red, …]`.
    3. Run replicator dynamics on the reduced game to obtain a low-regret
       mixture.
    4. Convert that mixture to the canonical ordering
       (MOBI_MELO, MOBI_CDA, ZI_MELO, ZI_CDA) and renormalise per role.
    """

  
    profs_full = list(itertools.chain.from_iterable(collect_profiles_by_seed(hp).values()))
    _log(f"[load] HP {hp}: loaded {len(profs_full)} raw profiles (before dedup)")
    if not profs_full:
        raise RuntimeError(f"No payoff data found for HP={hp}")

    game_full_ms = build_game_from_profiles(profs_full)
    if game_full_ms is None:
        raise RuntimeError(f"Cannot construct full RoleSymmetricGame for HP={hp}")
    # Diagnostic: count unique 68-player profiles in the aggregated data
    cfg_arr = game_full_ms.game.rsg_config_table.cpu().numpy().astype(int)
    n_profiles_total  = cfg_arr.shape[0]
    n_profiles_unique = np.unique(cfg_arr, axis=0).shape[0]
    _log(f"[load] HP {hp}: {n_profiles_unique} unique full-game profiles (from {n_profiles_total} rows)")
    # ------------------------------------------------------------------
    # Sanity-check: verify baseline 68-player profile and its single-agent
    # deviations are present in the aggregated data. The baseline counts are
    # (28, 0, 30, 10) following the canonical ordering (MOBI_MELO, MOBI_CDA,
    # ZI_MELO, ZI_CDA). Deviations move exactly one agent between MELO↔CDA for
    # either population.
    # ------------------------------------------------------------------
    def _check_full_profiles(game_obj):
        """Emit a diagnostic line for each required profile. Does *not*
        abort the run – it only prints a warning when a profile is missing so
        that the user can investigate further."""
        # Build mapping (role, strategy) → column index in the config table
        idx_map: dict[tuple[str,str], int] = {}
        idx = 0
        for role, strats in zip(game_obj.role_names, game_obj.strategy_names_per_role):
            for strat in strats:
                idx_map[(role, strat)] = idx
                idx += 1
        # Ensure the four canonical strategies are present
        missing_strats = [pair for pair in CANONICAL_ORDER if pair not in idx_map]
        if missing_strats:
            _log(f"[check] HP {hp}: missing expected strategies {missing_strats}; skipping profile check")
            return

        idx0 = idx_map[CANONICAL_ORDER[0]]  # MOBI_MELO
        idx1 = idx_map[CANONICAL_ORDER[1]]  # MOBI_CDA
        idx2 = idx_map[CANONICAL_ORDER[2]]  # ZI_MELO
        idx3 = idx_map[CANONICAL_ORDER[3]]  # ZI_CDA

        # desired_profiles = [ 
        #     (28, 0, 30, 10),
        #     (27, 1, 30, 10),
        #     (28, 0, 29, 11),
        #     (28, 0, 31, 9),
        # ]

        # cfg_table = game_obj.game.rsg_config_table.cpu().numpy().astype(int)

        # for counts in desired_profiles:
        #     found = ((cfg_table[:, idx0] == counts[0]) &
        #              (cfg_table[:, idx1] == counts[1]) &
        #              (cfg_table[:, idx2] == counts[2]) &
        #              (cfg_table[:, idx3] == counts[3])).any()
        #     status = "✓" if found else "✗ MISSING"
        #     _log(f"[check] HP {hp}: profile {counts} {status}")
    # Run the check immediately
    _check_full_profiles(game_full_ms)
    # Extra: print full-game payoffs for profile (28, 0, 10, 30)
    def _print_payoffs_for_counts(game_obj, counts_tup):
        idx_map: dict[tuple[str,str], int] = {}
        idx = 0
        for role, strats in zip(game_obj.role_names, game_obj.strategy_names_per_role):
            for strat in strats:
                idx_map[(role, strat)] = idx
                idx += 1
        try:
            i0 = idx_map[CANONICAL_ORDER[0]]
            i1 = idx_map[CANONICAL_ORDER[1]]
            i2 = idx_map[CANONICAL_ORDER[2]]
            i3 = idx_map[CANONICAL_ORDER[3]]
        except KeyError:
            _log(f"[check] HP {hp}: canonical strategies missing – cannot print payoffs for {counts_tup}")
            return
        cfg = game_obj.game.rsg_config_table.cpu().numpy().astype(int)
        mask = ((cfg[:, i0] == counts_tup[0]) &
                (cfg[:, i1] == counts_tup[1]) &
                (cfg[:, i2] == counts_tup[2]) &
                (cfg[:, i3] == counts_tup[3]))
        cols = np.where(mask)[0]
        if cols.size == 0:
            _log(f"[check] HP {hp}: profile {counts_tup} not present – no payoffs to show")
            return
        pay_tab = game_obj.game.rsg_payoff_table[:, cols].cpu().numpy()
        vec = np.nanmean(pay_tab, axis=1)
        vec = [round(float(x), 4) for x in vec]
        # Pretty label order (canonical order per role)
        labels = [f"{r}:{s}" for r,s in CANONICAL_ORDER]
        _log(f"[check] HP {hp}: mean payoffs for profile {counts_tup} → {{" + ", ".join(f"{lab}={val}" for lab,val in zip(labels, vec)) + "}}")
    _print_payoffs_for_counts(game_full_ms, (28, 0, 10, 30))

    role_names = game_full_ms.role_names
    num_players_full = [int(x) for x in game_full_ms.num_players_per_role]
    strat_names_per_role = game_full_ms.strategy_names_per_role

    profiles_arr = game_full_ms.game.rsg_config_table.cpu().numpy().astype(int)
    payoffs_arr = game_full_ms.game.rsg_payoff_table.cpu().numpy().T  # shape (num_profiles, num_strats)

  
    zero_mask = profiles_arr == 0
    payoffs_arr[zero_mask] = 0.0

    ga_full = paygame.game_names(
        role_names,
        num_players_full,
        strat_names_per_role,
        profiles_arr,
        payoffs_arr,
    )
    _log(f"[DPR] Building initial paygame for HP {hp} with {profiles_arr.shape[0]} profiles …")
   
    red_players = np.array([n_red] * len(role_names))

    MAX_REFILL = 3  # at most three rounds of gap filling
    for refill_round in range(1, MAX_REFILL + 1):
        _log(f"[DPR] Reduction round {refill_round} – applying DPR reduction …")
        ga_red_full = deviation_preserving.reduce_game(ga_full, red_players)

        n_nan = np.isnan(ga_red_full.payoffs()).sum()
        missing_baselines = ga_red_full.num_profiles < (n_red + 1) ** len(role_names)
        _log(f"[DPR]   reduced game: {ga_red_full.num_profiles} profiles, NaN entries={n_nan}, missing_baselines={missing_baselines}")
        p_raw = ga_red_full.payoffs()
        prof_counts = ga_red_full.profiles()
        pay_clean = p_raw.copy()
        pay_clean[prof_counts == 0] = 0.0
        min_pay = ga_red_full.min_strat_payoffs()
        keep_mask = ~np.isnan(min_pay)

        need_fill = (n_nan > 0) or missing_baselines
        if not need_fill:
            break

        filled = _fill_dpr_missing_payoffs(
            hp,
            ga_full,
            ga_red_full,
            game_full_ms,
            role_names,
            num_players_full,
            strat_names_per_role,
            n_red=n_red,
            print_only=print_only,
        )

        if not filled:
            _log("[DPR]   No further profiles could be filled – stopping gapfill loop")
            break

        # Re-build *ga_full* from the now-updated RoleSymmetricGame tables
        profiles_arr = game_full_ms.game.rsg_config_table.cpu().numpy().astype(int)
        payoffs_arr = game_full_ms.game.rsg_payoff_table.cpu().numpy().T
        zero_mask = profiles_arr == 0
        payoffs_arr[zero_mask] = 0.0
        ga_full = paygame.game_names(
            role_names,
            num_players_full,
            strat_names_per_role,
            profiles_arr,
            payoffs_arr,
        )

   

    #p_raw = np.nan_to_num(ga_red_full.payoffs(), nan=0.0) 
    p_raw = ga_red_full.payoffs()
    prof_counts = ga_red_full.profiles() 
    pay_clean = p_raw.copy()
    pay_clean[prof_counts == 0] = 0.0

   
    #pay_clean[pay_clean < 0] = 0.0

    #if not np.array_equal(pay_clean, ga_red_full.payoffs()):
        #ga_red_full = paygame.game_replace(ga_red_full, ga_red_full.profiles(), pay_clean)



    min_pay = ga_red_full.min_strat_payoffs()
    keep_mask = ~np.isnan(min_pay)


    # for r, (start, size) in enumerate(zip(ga_red_full.role_starts, ga_red_full.num_role_strats)):
    #     if not keep_mask[start:start + size].any():
    #         raise RuntimeError(
    #             f"HP {hp}: all strategies missing data for role {role_names[r]} even after refill"
    #         )

    #if not keep_mask.all():
    #ga_red = ga_red_full.restrict(keep_mask)
    #else:
    ga_red = ga_red_full


    N_START = 30
    REGRET_MAX = 1e-4
    CANDIDATES: list[torch.Tensor] = []
    WELFARE: list[float] = []


  
    corner_eps = 0.05 
    pre_starts: list[np.ndarray] = []

   
    mob_start, zi_start = int(ga_red.role_starts[0]), int(ga_red.role_starts[1])

    for mob_choice in (0, 1):      
        for zi_choice in (0, 1):
            vec = np.zeros(ga_red.num_strats)
       
            vec[mob_start + mob_choice] = 1 - corner_eps
            vec[mob_start + (1 - mob_choice)] = corner_eps
      
            vec[zi_start + zi_choice] = 1 - corner_eps
            vec[zi_start + (1 - zi_choice)] = corner_eps
            pre_starts.append(vec)

    
    centre = np.zeros(ga_red.num_strats)
    centre[mob_start:mob_start+2] = 0.5
    centre[zi_start:zi_start+2]  = 0.5
    pre_starts.append(centre)

    rng_loop = np.random.default_rng(hp + 123)

    total_iters = pre_starts + [None] * N_START  
    for j, start_seed in enumerate(total_iters):
        if start_seed is None:
            mix0 = rng_loop.random(ga_red.num_strats) + 1e-3
            for start, size in zip(ga_red.role_starts, ga_red.num_role_strats):
                seg = slice(int(start), int(start+size))
                mix0[seg] /= mix0[seg].sum()
        else:
            mix0 = start_seed.copy()

        try:
            mix_np_j = ga_nash.replicator_dynamics(ga_red, mix0)
        except ValueError:
            try:
                mix_np_j = ga_nash.replicator_dynamics(ga_red, mix0)
            except ValueError:
                continue

        # Compute regret and print mixture details
        reg_val = ga_regret.mixture_regret(ga_red, mix_np_j)


        print(f"HP {hp} | RD start {j:02d} with starting mix {mix0} → regret = {reg_val:.6g} for mix {np.round(mix_np_j, 3).tolist()}")
        # skip if regret too high
        if reg_val > REGRET_MAX:
            continue

        mix_torch = torch.tensor(mix_np_j, dtype=torch.float32)
        # deduplicate
        if any(_is_duplicate(mix_torch, prev, 0.01) for prev in CANDIDATES):
            continue

        CANDIDATES.append(mix_torch)
        WELFARE.append(welfare_multinomial(ga_red, mix_np_j))
    #base_mix = np.array([1.0, 0.0, 0.4, 0.6])
    special_mix = mix_np_j.copy()  
    start_mobi, start_zi = ga_red.role_starts
    size_mobi , size_zi = ga_red.num_role_strats
                 
    base_mix = np.array([1, 0,  0.75, 0.25])

    # --- convert canonical 4-entry vector to game-order vector ---------
    base_mix_game: list[float] = []
    for role, start, n in zip(ga_red.role_names, ga_red.role_starts, ga_red.num_role_strats):
        for off in range(int(n)):
            strat = ga_red.strat_name(int(start + off))
            base_mix_game.append(float(base_mix[IDX_CAN_MAP[(role, strat)]]))
    base_mix_game = np.asarray(base_mix_game, dtype=float)
    # renormalise per role (required by multi-pop SRD)
    for start, n in zip(ga_red.role_starts, ga_red.num_role_strats):
        seg = slice(int(start), int(start + n))
        base_mix_game[seg] /= base_mix_game[seg].sum()

    # --- deterministic RD check (legacy) ------------------------------
    print(f"test with rd that {base_mix} is an eq: regret = {ga_regret.mixture_regret(ga_red, base_mix_game)}")

    # --- RSNE check via mixture_regret (deviation–payoff based) ---------
    reg_base = ga_regret.mixture_regret(ga_red, base_mix_game)
    print("canonical mix", base_mix.tolist(), "→ regret =", reg_base)

     # --- exact support-enumeration equilibrium search ------------------
    try:
         eq_arr = ga_nash.mixed_nash(
             ga_red,
             regret_thresh=1e-3,      # stop as soon as ε<=1e-3
             grid_points=0,           # skip coarse grid; do pure support enum
             random_restarts=0,
         )
         if eq_arr.size > 0:
             print("mixed_nash returned", eq_arr.shape[0], "equilibria: ↑")
             for k, eqm in enumerate(eq_arr):
                 print(f"  eq {k}: mix =", np.round(eqm, 3).tolist(),
                       "| regret =", ga_regret.mixture_regret(ga_red, eqm))
         else:
             print("[mixed_nash] no equilibrium found within ε≤1e-3")
    except Exception as exc:
         print("[mixed_nash] support-enumeration failed:", exc)

     # --- per-role best-response diagnostic -----------------------------
    dev_pay = ga_red.deviation_payoffs(torch.tensor(base_mix_game, dtype=torch.float32))
    idx = 0
    for r, n in enumerate(ga_red.num_role_strats):
         seg = slice(idx, idx + int(n))
         role_val = float((torch.tensor(base_mix_game[seg]) * dev_pay[seg]).sum())
         best_dev = float(dev_pay[seg].max())
         print(f"role {ga_red.role_names[r]}:  V_r(x̂) = {role_val:.2f}   best-dev = {best_dev:.2f}   regret = {best_dev - role_val:.2f}")
         idx += int(n)

    # NOTE: gameanalysis < 0.26 does not expose stochastic_replicator_dynamics.
    # If you upgrade gameanalysis you can uncomment the block below to run the
    # deviation-payoff SRD which respects RSNE.  Until then we rely on the
    # regret value above.
    #
    # from gameanalysis.nash import stochastic_replicator_dynamics as _srd
    # mix_srd = _srd(
    #     ga_red,
    #     init_mix=base_mix_game.copy(),
    #     step=0.01,
    #     beta=0.0,
    #     max_iters=200_000,
    #     converge_eps=1e-9,
    #     rng=np.random.default_rng(123),
    # )
    # print("SRD converged mix:", np.round(mix_srd, 3).tolist(),
    #       "| regret =", ga_regret.mixture_regret(ga_red, mix_srd))
    eps = 1e-6            # step size  (e.g. 0.0001)
    tau = 1e-4            # stop when regret > tau


 

    # def scan_direction(sign: int):
    #     """sign = +1 to push component-2 up, −1 to push it down."""
    #     step_num = 0
    #     val_2 = base_mix[2]
    #     while 0.0 < val_2 < 1.0:
    #         mix = base_mix.copy()
    #         mix[2] = val_2
    #         mix[3] = 1.0 - mix[2]
    #         reg = ga_regret.mixture_regret(ga_red, mix)
    #         print(f"{'+' if sign>0 else '-'} step {step_num:03d} | mix {np.round(mix,4)} | reg={reg:.6g}")
    #         # if reg > tau:
    #         #     print("    ↳ stopping (regret above τ)")
    #         #     break
    #         # # next step
    #         step_num += 1
    #         val_2 += sign * eps

    # print("\n### Scanning upward (component-2 ↑) ###")
    # scan_direction(+1)
    # print("\n### Scanning downward (component-2 ↓) ###")
    # #scan_direction(-1)

    profiles  = list(itertools.chain.from_iterable(collect_profiles_by_seed(hp).values()))
    game_full = build_game_from_profiles(profiles)
    assert game_full is not None, "couldn’t build full game"

 
    mix = torch.tensor(special_mix, dtype=torch.float32)

    
    mobi_slice = slice(int(start_mobi), int(start_mobi + size_mobi))
    zi_slice   = slice(int(start_zi)  , int(start_zi   + size_zi))

    mixes_to_check = [[1, 0, 0, 1], [0, 1, 0, 1], [1, 0, 1, 0], [1.0, 0.0, 0.097, 0.903], [1, 0, .75, .25]]
    for mix in mixes_to_check:
        print(f"checking welfare for {mix}")

        mix_full_vals: list[float] = []
        for role, strats in zip(game_full.role_names, game_full.strategy_names_per_role):
            for strat in strats:
                mix_full_vals.append(float(mix[IDX_CAN_MAP[(role, strat)]]))
        mix_full = torch.tensor(mix_full_vals, dtype=torch.float32)
        print(f"mix full {mix_full}")
        start = 0
        for n in [len(s) for s in game_full.strategy_names_per_role]:
            seg = slice(start, start + n)
            mix_full[seg] = mix_full[seg] / mix_full[seg].sum()
            start += n

        reg_val = game_full.regret(mix_full)
        print(f"welfare: {welfare_multinomial(game_full, mix_full)} | regret: {reg_val:.6g}")
        dv_vec = game_full.deviation_payoffs(mix_full)

        start = 0
        for role_name, strats in zip(game_full.role_names, game_full.strategy_names_per_role):
            seg = slice(start, start + len(strats))
            V_r = float((mix_full[seg] * dv_vec[seg]).sum())
            print(f"   {role_name} equilibrium value  V_{role_name}(x̂) = {V_r:.3f}")
            start += len(strats)

    # --------------------------------------------------------------
    # Simple grid-search: vary t in [0, t_target] for mix [1,0,t,1-t]
    # --------------------------------------------------------------
    # t_target = 0.097  # final ZI_MELO share
    # n_steps = 10      # number of intermediate points (inclusive)
    # print("\nGrid-search along ZI mixing axis (mix = [1, 0, t, 1−t])")
    # for step in range(n_steps + 1):
    #     t = (t_target / n_steps) * step
    #     mix_vec = [1.0, 0.0, t, 1.0 - t]

    #     # Build full-length mixture vector in game order
    #     mix_full_vals = []
    #     for role, strats in zip(game_full.role_names, game_full.strategy_names_per_role):
    #         for strat in strats:
    #             mix_full_vals.append(float(mix_vec[IDX_CAN_MAP[(role, strat)]]))
    #     mix_full = torch.tensor(mix_full_vals, dtype=torch.float32)

    #     # Renormalise per role (probabilities must sum to 1 for each role)
    #     start_idx = 0
    #     for n in [len(s) for s in game_full.strategy_names_per_role]:
    #         seg = slice(start_idx, start_idx + n)
    #         mix_full[seg] = mix_full[seg] / mix_full[seg].sum()
    #         start_idx += n

    #     w = _social_welfare(game_full, mix_full)
    #     dv_vec = game_full.deviation_payoffs(mix_full)

    #     # Per-role baseline values
    #     idx_dv = 0
    #     role_vals = {}
    #     for role_name, strats in zip(game_full.role_names, game_full.strategy_names_per_role):
    #         seg = slice(idx_dv, idx_dv + len(strats))
    #         role_vals[role_name] = float((mix_full[seg] * dv_vec[seg]).sum())
    #         idx_dv += len(strats)


    mixes_to_check = [[1, 0, 0, 1], [0, 1, 0, 1], [1, 0, 1, 0], [1.0, 0.0, 0.097, 0.903], [1, 0, .75, .25]]
    for mix in mixes_to_check:
        print(f"checking welfare for {mix} in DPR GAME")

        # Build mixture vector in *ga_red* order (DPR reduced game)
        mix_red_vals: list[float] = []
        # Build list of strategies per role directly from ga_red indices
        for role, start, n_str in zip(ga_red.role_names, ga_red.role_starts, ga_red.num_role_strats):
            for offset in range(int(n_str)):
                strat = ga_red.strat_name(int(start + offset))
                mix_red_vals.append(float(mix[IDX_CAN_MAP[(role, strat)]]))
        mix_red = torch.tensor(mix_red_vals, dtype=torch.float32)

        # Renormalise per role
        start = 0
        for n in ga_red.num_role_strats:
            seg = slice(start, start + n)
            mix_red[seg] = mix_red[seg] / mix_red[seg].sum()
            start += n

        # Compute welfare in the DPR game
        dev_vec = ga_red.deviation_payoffs(mix_red)
        start = 0
        welfare = 0.0
        role_vals = {}
        for role_name, n in zip(ga_red.role_names, ga_red.num_role_strats):
            seg = slice(start, start + n)
            V_r = float((mix_red[seg] * dev_vec[seg]).sum())
            role_vals[role_name] = V_r
            welfare += V_r
            start += n

        pay_full = ga_red_full.payoffs()
        keep_str = ~(pay_full == 0).all(axis=0)      # shape (num_strats,)

        ga_reg   = ga_red_full.restrict(keep_str)
        print(f"mix (DPR order) {mix_red.numpy()}")
        
        
        reg_red = ga_regret.mixture_regret(ga_red, mix_red.numpy())
        print(f"welfare (DPR): {welfare:.3f} | regret (DPR): {reg_red:.6g}")
        for role_name, V_r in role_vals.items():
            print(f"   {role_name} equilibrium value  V_{role_name}(x̂) = {V_r:.3f}")

    
    exit()
   
    # special_mix = mix_np_j.copy()
    # special_mix = [1, 0, 0, 1]
    # special_mix = [1, 0, 0, 1]

    # n_red = int(ga_red.num_role_players[0])     
    # role_names = ga_red.role_names              
    # start_mobi, start_zi = ga_red.role_starts
    # size_mobi , size_zi = ga_red.num_role_strats

    # profiles  = list(itertools.chain.from_iterable(collect_profiles_by_seed(hp).values()))
    # game_full = build_game_from_profiles(profiles)
    # assert game_full is not None, "couldn’t build full game"

    
    # mix = torch.tensor(special_mix, dtype=torch.float32)

    # start = 0
    # for n in [len(s) for s in game_full.strategy_names_per_role]:
    #     seg = slice(start, start+n); mix[seg] /= mix[seg].sum(); start += n
    # print(mix)
    # print(f"HP {hp} – full-game regret = {game_full.regret(mix):.9f}")

    # mobi_slice = slice(int(start_mobi), int(start_mobi + size_mobi))   
    # zi_slice   = slice(int(start_zi)  , int(start_zi   + size_zi))
    # mixes_to_check = [[1, 0, 0, 1], [0, 1, 0, 1], [1, 0, 1, 0]]
    # for mix in mixes_to_check:
    #     print(f"checking welfare for {mix}")
    #     print(f"welfare: {_ga_social_welfare(ga_red, mix)}")
    #     mix_t  = torch.tensor(mix, dtype=torch.float32)
    #     dv_vec = ga_red.deviation_payoffs(mix_t)       
    #     start  = 0
    #     for role_name, n_strat in zip(ga_red.role_names, ga_red.num_role_strats):
    #         seg = slice(start, start + n_strat)
    #         V_r = float((mix_t[seg] * dv_vec[seg]).sum())
    #         print(f"   {role_name} equilibrium value  V_{role_name}(x̂) = {V_r:.3f}")
    #         start += n_strat





    # def role_value(row_idx, role_slice):
    #     counts = ga_red.profiles()[row_idx, role_slice]
    #     pay    = ga_red.payoffs()[row_idx, role_slice]
    #     return (counts * pay).sum() / n_red     

    # grid_x, grid_y, val_mobi, val_zi = [], [], [], []
    # for i, prof in enumerate(ga_red.profiles()):
    #     x = int(prof[mobi_slice.start])      
    #     y = int(prof[zi_slice.start])            
    #     grid_x.append(x)
    #     grid_y.append(y)
    #     val_mobi.append(role_value(i, mobi_slice))
    #     val_zi  .append(role_value(i, zi_slice ))

    # grid_x = np.array(grid_x)
    # grid_y = np.array(grid_y)
    # val_mobi = np.array(val_mobi)
    # val_zi   = np.array(val_zi)
    # val_sw   = val_mobi + val_zi                
    # eq_mixes = []
    
    # for mix in CANDIDATES:
    #     print(mix)
    #     #eq_mix = ga_nash.replicator_dynamics(ga_red, np.ones(ga_red.num_strats)/ga_red.num_strats)
    #     x_eq = n_red * float(mix[mobi_slice.start])     #
    #     y_eq = n_red * float(mix[zi_slice.start])    

    #     eq_mixes.append([x_eq, y_eq])


    # # --- plotting --------------------------------------------------------------
    # fig = plt.figure(figsize=(16, 5))

    # def add_surface(ax_idx, title, z_vals, cmap='viridis'):
    #     ax = fig.add_subplot(1, 3, ax_idx, projection='3d')
    #     ax.plot_trisurf(grid_x, grid_y, z_vals, cmap=cmap, alpha=0.85)
    #     #ax.scatter([x_eq], [y_eq], [np.interp((x_eq,y_eq), (grid_x,grid_y), z_vals)],
    #             #c='red', marker='x', s=80, label='equilibrium')

     
    #     for x_eq, y_eq in eq_mixes:
    #         dist2 = (grid_x - x_eq)**2 + (grid_y - y_eq)**2
    #         idx   = np.argmin(dist2)
    #         z_eq  = z_vals[idx]
    #         ax.scatter([x_eq], [y_eq], [z_eq], c='red', marker='x', s=80, label='equilibrium')
       

     
    #     ax.set_xlabel('# MOBI MELO')
    #     ax.set_ylabel('# ZI MELO')
    #     ax.set_zlabel('payoff')
    #     ax.set_title(title)
    #     ax.set_xticks(range(0, n_red+1))
    #     ax.set_yticks(range(0, n_red+1))
    #     ax.view_init(elev=25, azim=45)
    #     ax.legend()

    # add_surface(1, f'{role_names[0]} value', val_mobi, cmap='Blues')
    # add_surface(2, f'{role_names[1]} value', val_zi,   cmap='Oranges')
    # add_surface(3, 'social welfare',         val_sw,   cmap='Greens')

    # plt.tight_layout()
    # plt.show()
    

    print(f"checking special regret {special_mix} regret is {ga_regret.mixture_regret(ga_red, special_mix):.6g}")
   
    
    def _is_pure(mix_arr, tol=1e-2):
        # considers mix pure if, for each role, one strategy mass >1-tol
        idx = 0
        for sz in ga_red.num_role_strats:
            seg = slice(idx, idx+sz)
            if not (mix_arr[seg] > 1-tol).any():
                return False
            idx += sz
        return True

    # if not CANDIDATES or all(_is_pure(m.cpu().numpy()) for m in CANDIDATES):
    #     print(f"HP {hp}: RD only found pure strategies – trying mixed_nash …")
    #     try:
    #         eqm_arr = ga_nash.mixed_nash(ga_red, regret_thresh=1e-3, grid_points=15, random_restarts=100)
    #         if eqm_arr.size > 0:
    #             mix_np = eqm_arr[0]
    #         else:
    #             mix_np = None
    #     except Exception as exc:
    #         print(f"[warning] mixed_nash failed: {exc}")
    #         mix_np = None

    #     # --- If still no interior equilibrium, try marketsim replicator_dynamics on an RSG wrapper ---
    #     if mix_np is None:
    #         try:
    #             cfg_tensor = torch.tensor(ga_red.profiles(), dtype=torch.float32)
    #             pay_tensor = torch.tensor(ga_red.payoffs().T, dtype=torch.float32)
    #             rsg_tmp = RoleSymmetricGame(
    #                 role_names,
    #                 [n_red] * len(role_names),
    #                 strat_names_per_role,
    #                 rsg_config_table=cfg_tensor,
    #                 rsg_payoff_table=pay_tensor,
    #             )

    #             game_ms = Game(rsg_tmp)

    #             best_mix: Optional[torch.Tensor] = None
    #             best_reg = float("inf")
    #             rng_ms = np.random.default_rng(hp + 777)

    #             for k in range(30):
    #                 # fresh random start (role-normalised)
    #                 start_vec = torch.rand(game_ms.num_strategies)
    #                 start = 0
    #                 for sz in game_ms.num_players_per_role:
    #                     seg = slice(start, start + int(sz))
    #                     start_vec[seg] /= start_vec[seg].sum()
    #                     start += int(sz)

    #                 try:
    #                     cand = ms_rd(game_ms, mixture=start_vec, iters=3000, converge_threshold=1e-7,
    #                                  use_multiple_starts=False)
    #                     cand_reg = game_ms.regret(cand)
    #                     print(
    #                         f"HP {hp} | MS-RD {k:02d} → regret={cand_reg:.3g} mix {np.round(cand.numpy(),3).tolist()}"
    #                     )
    #                     if cand_reg < best_reg:
    #                         best_reg = cand_reg
    #                         best_mix = cand
    #                 except Exception as exc:
    #                     print(f"[warning] MS-RD start {k} failed: {exc}")

    #             if best_mix is not None:
    #                 mix_np = best_mix.numpy()
    #                 print(
    #                     f"HP {hp}: marketsim RD best mix (regret {best_reg:.3g}) = {np.round(mix_np,3).tolist()}"
    #                 )
    #             else:
    #                 mix_np = mix_np_j if 'mix_np_j' in locals() else np.ones(ga_red.num_strats) / ga_red.num_strats
    #         except Exception as exc:
    #             print(f"[warning] marketsim RD setup failed: {exc}")
    #             mix_np = mix_np_j if 'mix_np_j' in locals() else np.ones(ga_red.num_strats) / ga_red.num_strats
    # else:
        # pick median-welfare candidate
    idx_med = len(CANDIDATES)//2
    mix_np = CANDIDATES[sorted(range(len(CANDIDATES)), key=lambda k: WELFARE[k])[idx_med]].numpy()

    # ------------------------------------------------------------------
    # 4. Map to canonical 4-entry tensor and renormalise per-role
    # ------------------------------------------------------------------
    # If some strategies were dropped we need to expand back to full length
    full_mix_vec = np.zeros(len(keep_mask), dtype=float)
    if keep_mask.all():
        full_mix_vec = mix_np
    else:
        full_mix_vec[keep_mask] = mix_np

    out = [0.0] * 4
    idx = 0
    for role, strats in zip(role_names, strat_names_per_role):
        for strat in strats:
            can_idx = IDX_CAN_MAP[(role, strat)]
            out[can_idx] = float(full_mix_vec[idx]) if idx < len(full_mix_vec) else 0.0
            idx += 1

    out[0:2] = (np.array(out[0:2]) / np.sum(out[0:2])).tolist()
    out[2:4] = (np.array(out[2:4]) / np.sum(out[2:4])).tolist()


    mix_torch = torch.tensor(mix_np, dtype=torch.float32)
    dev_vec = ga_red.deviation_payoffs(mix_torch)
    idx_seg = 0
    exp_vals: list[float] = []
    for sz in ga_red.num_role_strats:
        seg = slice(idx_seg, idx_seg + sz)
        exp_vals.append(float((mix_torch[seg] * dev_vec[seg]).sum()))
        idx_seg += sz

    DPR_EQ_ROWS.append({
        "HP": hp,
        "MOBI_MELO": out[0],
        "MOBI_CDA": out[1],
        "ZI_MELO": out[2],
        "ZI_CDA": out[3],
        "Payoff_MOBI": exp_vals[0],
        "Payoff_ZI": exp_vals[1],
    })

 
    if kwargs.get("debug_print", True):
        #_print_dpr_game(ga_red, hp)
        _print_dpr_game(ga_red, hp, show_baseline=True, game_full=game_full_ms)


    return torch.tensor(out)


def bootstrap(hp_list: list[int], fixed_eq: bool = True, debug_hps: Iterable[int] = (), *, use_hand_eq: bool = False):
    from typing import Iterable

    def _debug_dump(game: Game, mix_full: torch.Tensor, hp_val: int):
        """Print mixture + per-role payoff diagnostics for one HP."""
        print(f"\n🔎 DEBUG HP {hp_val}")
        present_probs = []
        for role, strats in zip(game.role_names, game.strategy_names_per_role):
            for strat in strats:
                present_probs.append(float(mix_full[IDX_CAN_MAP[(role, strat)]]))
        mix_t = torch.tensor(present_probs, dtype=torch.float32)
        start = 0
        for strats in game.strategy_names_per_role:
            seg = slice(start, start+len(strats))
            mix_seg = mix_t[seg]
            mix_t[seg] = mix_seg / mix_seg.sum()
            start += len(strats)

        dev = game.deviation_payoffs(mix_t)
        idx = 0
        for role, strats in zip(game.role_names, game.strategy_names_per_role):
            seg = slice(idx, idx+len(strats))
            Vr = float((mix_t[seg]*dev[seg]).sum())
            pay_melo = pay_cda = None
            for j, strat in enumerate(strats):
                val = float(dev[idx+j])
                if "_0_100" in strat:
                    pay_melo = val
                elif "_100_0" in strat:
                    pay_cda = val
            mix_list = [f"{strats[j]}={mix_t[idx+j]:.3f}" for j in range(len(strats))]
            print(f"  {role}: V={Vr:.4f} | MELO={pay_melo:.4f}  CDA={pay_cda:.4f} | mix: {'; '.join(mix_list)}")
            idx += len(strats)

    rng = np.random.default_rng()
    records = []

    SEED_MAP = {hp: collect_profiles_by_seed(hp) for hp in hp_list}

    if fixed_eq:
        if use_hand_eq:
            print("Using user-supplied historical equilibrium map …")
            pooled_map = {hp: HAND_EQ_DICT[hp] for hp in hp_list if hp in HAND_EQ_DICT}
        else:
            print("Computing fixed pooled *DPR* equilibria for the requested HP values …")
            pooled_map: dict[int, torch.Tensor] = {
                hp: find_pooled_equilibrium_dpr(hp, print_only=False) for hp in hp_list
            }
        # Echo the computed equilibria so they are visible in the console
        print("\n=== Fixed pooled equilibria used in bootstrap ===")
        for hp_val, eq in sorted(pooled_map.items()):
            eq_str = ", ".join(f"{p:.3f}" for p in eq.tolist())
            print(f"HP {hp_val:>3}  →  [ {eq_str} ]")
        print("===============================================\n")
        print("…done.\n")
    else:
        pooled_map = {}

    # ------------------------------------------------------------------
    # Optional debug print *once* per requested HP using pooled game
    # ------------------------------------------------------------------
    for hp_dbg in debug_hps:
        if hp_dbg in pooled_map and hp_dbg in POOLED_GAMES:
            _debug_dump(POOLED_GAMES[hp_dbg], pooled_map[hp_dbg], hp_dbg)
    
    for hp in hp_list:
        for _ in range(N_BOOT):
            #game = resample_complete_game(hp, rng)
           # if game is None:
             #   continue
            print(f"Running Bootstrap {_}")
            profs = sample_seed_bootstrap(hp, rng, SEED_MAP[hp])
             
            if not profs:
                continue
            game = build_game_from_profiles(profs)
            if game is None:
                continue

         
            ref_game = POOLED_GAMES[hp]
            ref_cfgs = {tuple(c.tolist()): j for j, c in enumerate(ref_game.game.rsg_config_table.t())}
            for col_idx, cfg in enumerate(game.game.rsg_config_table.t()):
                key = tuple(cfg.tolist())
                if key in ref_cfgs:
                    ref_col = ref_cfgs[key]
                    mask_col = torch.isnan(game.game.rsg_payoff_table[:, col_idx])
                    if mask_col.any():
                        game.game.rsg_payoff_table[mask_col, col_idx] = ref_game.game.rsg_payoff_table[mask_col, ref_col]  # type: ignore

            # Skip replicate if a strategy column was dropped
            if any(len(strats) < 2 for strats in game.strategy_names_per_role):
                continue

            
            if fixed_eq:
                mix = pooled_map[hp]
                present = []
                for role, strats in zip(game.role_names, game.strategy_names_per_role):
                    for strat in strats:
                        present.append(float(mix[IDX_CAN_MAP[(role, strat)]]))
                # Build mixture vector in game order and normalise per role (Figure-9 convention)
                mix_t = _normalise_per_role(torch.tensor(present, dtype=torch.float32))
            else:
                mix_t = None
                if all(len(lst) == 2 for lst in game.strategy_names_per_role):
                    try:
                        mix_t, _ = min(game.find_nash_equilibrium_2x2(), key=lambda x: x[1])
                    except Exception:
                        pass
                if mix_t is None:
                    try:
                        mix_t = replicator_dynamics(
                            game,
                            iters=3000,
                            converge_threshold=1e-7,
                            use_multiple_starts=True,
                        )
                    except Exception:
                        mix_t = torch.ones(game.num_strategies) / game.num_strategies
                mix_t = _normalise_per_role(mix_t.clone())

                # ensure epsilon-RSNE; if regret too high try fallback or skip
                if game.regret(mix_t) > EPS_RSNE:
                    # second pass with more iterations
                    try:
                        mix_t = replicator_dynamics(
                            game,
                            mix_t,
                            iters=10000,
                            converge_threshold=1e-8,
                            use_multiple_starts=False,
                        )
                    except Exception:
                        pass
                    mix_t = _normalise_per_role(mix_t.clone())
                    if game.regret(mix_t) > EPS_RSNE:
                        # skip this replicate
                        continue

            dev = game.deviation_payoffs(mix_t)
            idx = 0
            for role, strats in zip(game.role_names, game.strategy_names_per_role):
                seg = slice(idx, idx + len(strats)) 
                Vr = float((mix_t[seg] * dev[seg]).sum())
                records.append({"HP": hp, "Role": role, "Label": "V", "Val": Vr})

                # --- NEW: compute CDA–MELO payoff gap for ZI role -----------
                gap_vals = {}
                for strat in strats:
                    val = dev[idx].item()
                    records.append({"HP": hp, "Role": role, "Label": strat, "Val": val})

                    if role == "ZI":
                        if "_0_100" in strat:      # MELO variant
                            gap_vals["MELO"] = val
                        elif "_100_0" in strat:    # CDA variant
                            gap_vals["CDA"] = val
                    idx += 1

                # Store the gap once both values seen
                if role == "ZI" and gap_vals.keys() >= {"MELO", "CDA"}:
                    records.append({"HP": hp, "Role": role, "Label": "Gap", "Val": gap_vals["CDA"] - gap_vals["MELO"]})
    return pd.DataFrame(records)


def summarise_with_z(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (hp, role, label), grp in df.groupby(["HP", "Role", "Label"]):
        vals = grp["Val"].to_numpy()
        mu = float(vals.mean())
        sd = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        if sd > 0 and len(vals) > 1:
            z = (vals - mu) / sd
            lo_z, hi_z = np.percentile(z, [2.5, 97.5])
            lo = lo_z * sd + mu
            hi = hi_z * sd + mu
        else:
            lo = hi = mu
        rows.append({"HP": hp, "Role": role, "Label": label, "Mean": mu, "Lo": lo, "Hi": hi})
    return pd.DataFrame(rows)

# ------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------

def plot_figure(df_sum: pd.DataFrame, title_suffix: str, out_png: Path):
    roles = sorted(df_sum["Role"].unique())
    fig, axes = plt.subplots(1, len(roles), figsize=(15, 6), sharey=False)
    if len(roles) == 1:
        axes = [axes]

    for ax, role in zip(axes, roles):
        sub = df_sum[df_sum["Role"] == role]
        hp_sorted = sorted(sub["HP"].unique())
        ax.set_xticks(hp_sorted)
        ax.set_xticklabels([str(int(t)) for t in hp_sorted], rotation=45, ha="right")
        if hp_sorted:
            # Set x-axis to span from 0 to max HP to use full width
            ax.set_xlim(0, max(hp_sorted))

        label_map = {
            (CANONICAL_ORDER[1][1] if role == "MOBI" else CANONICAL_ORDER[3][1]): "Dev: CDA",
            (CANONICAL_ORDER[0][1] if role == "MOBI" else CANONICAL_ORDER[2][1]): "Dev: MELO",
            "V": "Equilibrium Value",
            "Gap": "CDA − MELO Gap",
        }

        for label, g in sub.groupby("Label"):
            g_sorted = g.sort_values("HP")
            x = g_sorted["HP"].to_numpy()
            y = g_sorted["Mean"].to_numpy()
            lo = g_sorted["Lo"].to_numpy()
            hi = g_sorted["Hi"].to_numpy()

            # optional smoothing (same flags as Fig-9)
            if USE_SMOOTHING and label in label_map:
                y_plot = gaussian_filter1d(y, sigma=.25)
            else:
                y_plot = y

            if SMOOTH_CI and not np.isnan(lo).all():
                lo_plot = gaussian_filter1d(lo, sigma=.25)
                hi_plot = gaussian_filter1d(hi, sigma=.25)
            else:
                lo_plot, hi_plot = lo, hi

            if label == "V":
                ax.plot(x, y_plot, color="black", lw=2, marker="s", label=label_map.get(label, label))
                ax.fill_between(x, lo_plot, hi_plot, color="grey", alpha=0.25)
            else:
                ax.plot(x, y_plot, marker="o", label=label_map.get(label, label))
                ax.fill_between(x, lo_plot, hi_plot, alpha=0.15)

        ax.set_title(f"{role} – Deviations vs Vᵣ(x̂) (bootstrap {N_BOOT})")
        ax.set_xlabel("Holding Period")
        ax.set_ylabel("Payoff")
        # Remove the hardcoded xlim since we set it above based on data
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    fig.savefig(out_png, dpi=300)
    print("Saved", out_png.relative_to(Path.cwd()))

# ------------------------------------------------------------------
# Bump-chart helper – rank payoffs per HP and show ordering evolution
# ------------------------------------------------------------------

def plot_bump_chart(df_sum: pd.DataFrame, title_suffix: str, out_png: Path):
    """Draw a bump chart of rank-only payoff ordering.

    Parameters
    ----------
    df_sum : pd.DataFrame
        Summary table returned by ``summarise_with_z``.
    title_suffix : str
        Extra title information (e.g. data source).
    out_png : Path
        Output PNG path.
    """
    roles = sorted(df_sum["Role"].unique())
    fig, axes = plt.subplots(1, len(roles), figsize=(7 * len(roles), 6), sharey=True)
    if len(roles) == 1:
        axes = [axes]

    # Colour palette (consistent across roles)
    colour_map = {
        "equilibrium": "black",
        "melo": "royalblue",
        "cda": "darkorange",
    }

    for ax, role in zip(axes, roles):
        sub = df_sum[df_sum["Role"] == role]
        if sub.empty:
            continue

        # Map strategy labels to canonical short names
        lab_eq = "V"  # equilibrium value label in df_sum
        lab_melo = CANONICAL_ORDER[0][1] if role == "MOBI" else CANONICAL_ORDER[2][1]
        lab_cda  = CANONICAL_ORDER[1][1] if role == "MOBI" else CANONICAL_ORDER[3][1]

        nice_name = {
            lab_eq: "equilibrium",
            lab_melo: "melo",
            lab_cda: "cda",
        }

        # Build tidy table (HP, series, payoff)
        rows = []
        for label in [lab_eq, lab_melo, lab_cda]:
            grp = sub[sub["Label"] == label]
            for hp, payoff in zip(grp["HP"], grp["Mean"]):
                rows.append({"holding_per": hp, "series": nice_name[label], "payoff": payoff})
        if not rows:
            continue
        df = pd.DataFrame(rows)

        # Rank within each HP (1 = highest payoff)
        df["rank"] = df.groupby("holding_per")["payoff"].rank(ascending=False)
        bump = df.pivot(index="holding_per", columns="series", values="rank")
        bump = bump.sort_index()

        # Draw bump chart
        for series in ["equilibrium", "melo", "cda"]:
            if series not in bump.columns:
                continue
            ax.plot(
                bump.index,
                bump[series],
                label=series.title(),
                marker="o",
                linewidth=3,
                color=colour_map[series],
                zorder=10,
            )
            ax.fill_between(
                bump.index,
                bump[series] + 0.4,
                bump[series] - 0.4,
                color=colour_map[series],
                alpha=0.15,
            )

        # Cosmetics
        ax.set_ylim(3.6, 0.4)
        ax.set_yticks([1, 2, 3])
        ax.set_yticklabels(["1st", "2nd", "3rd"])
        ax.set_xlabel("Holding Period")
        ax.set_title(f"{role} – Rank-only view {title_suffix}")
        ax.grid(axis="x", linestyle=":", linewidth=0.4)
        ax.legend(loc="upper center", ncol=3, frameon=False)

    plt.tight_layout()
    fig.savefig(out_png, dpi=300)
    print("Saved", out_png.relative_to(Path.cwd()))


# ------------------------------------------------------------------
# Pretty-print helpers for DPR games
# ------------------------------------------------------------------

def _print_dpr_game(ga_game, hp_val, *, show_baseline=False, game_full=None):
    """Print all reduced profiles with their payoffs (one line each).

    Output format:
        HP XX | PROFILE  ->  [pay1, pay2, pay3, pay4]

    If *show_baseline* is *False* (default) the numbers are the deviation
    payoffs taken directly from *ga_game* (as before).

    If *show_baseline* is *True* you must also pass *game_full* (the
    corresponding 68-player ``RoleSymmetricGame`` wrapper).  In that mode the
    vector shows **baseline expected payoffs per strategy**, averaged over all
    full-game profiles that reduce to the DPR profile and are present in
    *game_full*.
    """

    if show_baseline and game_full is None:
        raise ValueError("When show_baseline=True you must supply game_full")

    print(f"\n=== DPR reduced game for HP {hp_val} ({ga_game.num_profiles} profiles) ===")

    rnames     = ga_game.role_names
    s_per_role = ga_game.num_role_strats

    # Prepare list to collect CSV rows for this HP's DPR table
    rows_data: list[list] = []

    def _nice_profile(p):
        segs, idx = [], 0
        for r, n in zip(rnames, s_per_role):
            counts = p[idx:idx+n]
            segs.append(
                f"{r}(" + ", ".join(
                    f"{ga_game.strat_name(idx + s)}={counts[s]}" for s in range(n)
                ) + ")"
            )
            idx += n
        return "; ".join(segs)

    if show_baseline:
        # pre-compute mapping from full profiles to column index
        cfg_full = game_full.game.rsg_config_table.cpu().numpy().astype(int)

    for prof, pay in zip(ga_game.profiles(), ga_game.payoffs()):
        if not show_baseline:
            vec = np.round(pay, 4).tolist()
            contrib = []
        else:
            # # average baseline payoffs over contributing full-game profiles
            # from analysis.plot_bootstrap_compare import list_full_profiles_from_dpr

            # red_tup = tuple(int(x) for x in prof)
            # full_list = list_full_profiles_from_dpr(red_tup)
            # present_mask = [(cfg_full == flp).all(1) for flp in full_list]
            # cols = np.where(np.any(present_mask, 0))[0]

            # if cols.size == 0:
            #     vec = [0.0] * ga_game.num_strats
            #     contrib = []
            # else:
            #     pays_full = game_full.game.rsg_payoff_table[:, cols]
            #     vec = np.nanmean(pays_full, 1)
            #     vec = [round(float(x), 4) if not np.isnan(x) else 0.0 for x in vec]
            #     contrib = [full_list[i] for i, m in enumerate(present_mask) if m.any()]

            #     # Detailed per-profile payoffs (one line per contributing column)
            #     for flp, mask in zip(full_list, present_mask):
            #         if not mask.any():
            #             continue
            #         col_idx = np.where(mask)[0][0]
            #         pay_prof = game_full.game.rsg_payoff_table[:, col_idx].cpu().numpy()
            #         pay_prof = [round(float(x), 4) if not np.isnan(x) else 0.0 for x in pay_prof]
            #         print(f"        full {flp} → {pay_prof}")
            
            from analysis.plot_bootstrap_compare import list_full_profiles_from_dpr

            red_tup = tuple(int(x) for x in prof)
            full_list = list_full_profiles_from_dpr(red_tup)
            present_mask = [(cfg_full == flp).all(1) for flp in full_list]
            cols = np.where(np.any(present_mask, 0))[0]

            if cols.size == 0:
                vec = [0.0] * ga_game.num_strats
                contrib = []
            else:
                # pays_full = game_full.game.rsg_payoff_table[:, cols]
                # vec = np.nanmean(pays_full, 1)
                # vec = [round(float(x), 4) if not np.isnan(x) else 0.0 for x in vec]
                # contrib = [full_list[i] for i, m in enumerate(present_mask) if m.any()]

                # # Detailed per-profile payoffs (one line per contributing column)
                # for flp, mask in zip(full_list, present_mask):
                #     if not mask.any():
                #         continue
                #     col_idx = np.where(mask)[0][0]
                #     pay_prof = game_full.game.rsg_payoff_table[:, col_idx].cpu().numpy()
                #     pay_prof = [round(float(x), 4) if not np.isnan(x) else 0.0 for x in pay_prof]
                #     print(f"        full {flp} → {pay_prof}")
                from analysis.plot_bootstrap_compare import list_full_profiles_from_dpr

                import itertools

                red_tup = tuple(int(x) for x in prof)

                # Per-strategy mapping table (unique 68-agent profile for each strategy)
                mapping_dict = list_full_profiles_from_dpr(red_tup, return_mapping=True)

                key_order = ["MOBI_MELO", "MOBI_CDA", "ZI_MELO", "ZI_CDA"]

                vec = [0.0] * ga_game.num_strats  # will fill one value per strategy
                contrib: list[tuple[int, int, int, int]] = []

                # Build quick lookup → column index
                row_to_col = {tuple(cfg_full[i]): i for i in range(cfg_full.shape[0])}

                strat_idx_global = 0
                role_starts = ga_game.role_starts
                role_sizes = ga_game.num_role_strats

                for r_idx, role in enumerate(ga_game.role_names):
                    start = int(role_starts[r_idx])
                    size = int(role_sizes[r_idx])
                    for offset in range(size):
                        global_idx = start + offset
                        strat = ga_game.strat_name(global_idx)

                        idx_can = IDX_CAN_MAP.get((role, strat))
                        if idx_can is None:
                            continue  # skip unknown strategy

                        prof_counts = mapping_dict.get(key_order[idx_can])
                        if prof_counts is None:
                            continue  # unmapped strategy should not occur

                        col_idx = row_to_col.get(tuple(prof_counts))

                        if col_idx is None:
                            # profile not present yet – mark as NaN so gap-fill can detect
                            vec[global_idx] = float("nan")
                            # record missing profile for diagnostics (but avoid raising)
                            if prof_counts not in contrib:
                                contrib.append(prof_counts)
                        else:
                            payoff_val = float(game_full.game.rsg_payoff_table[global_idx, col_idx])
                            vec[global_idx] = round(payoff_val, 4)
                            if prof_counts not in contrib:
                                contrib.append(prof_counts)

                # Detailed print per contributing profile (one line each)
                for prof_counts in contrib:
                    col_idx = row_to_col[tuple(prof_counts)]
                    pay_prof = game_full.game.rsg_payoff_table[:, col_idx].cpu().numpy()
                    pay_prof = [round(float(x), 4) if not np.isnan(x) else 0.0 for x in pay_prof]
                    print(f"        full {prof_counts} → {pay_prof}")

            print(f"HP {hp_val:>3} | {_nice_profile(prof)}  ->  {vec}   full={contrib}")

            # Collect counts and payoffs in canonical order for CSV export
            idx_g = 0
            counts_can = [0] * 4
            payoffs_can = [0.0] * 4
            for role_name, role_size in zip(rnames, s_per_role):
                for off in range(role_size):
                    strat_nm = ga_game.strat_name(idx_g)
                    can_idx = IDX_CAN_MAP.get((role_name, strat_nm))
                    if can_idx is not None:
                        counts_can[can_idx] = int(prof[idx_g])
                        # Use vec value if available else 0.0
                        val_pay = vec[idx_g]
                        try:
                            payoffs_can[can_idx] = float(val_pay) if not (isinstance(val_pay, float) and np.isnan(val_pay)) else 0.0
                        except Exception:
                            payoffs_can[can_idx] = 0.0
                    idx_g += 1
            rows_data.append(counts_can + payoffs_can + [""])

            # Export collected DPR table to CSV
    if rows_data:
        import pandas as _pd
        header_cols = ["#_MOBI_MELO", "#_MOBI_CDA", "#_ZI_MELO", "#_ZI_CDA", "Pay_MOBI_MELO", "Pay_MOBI_CDA", "Pay_ZI_MELO", "Pay_ZI_CDA", "Comment"]
        df_out = _pd.DataFrame(rows_data, columns=header_cols)
        out_csv_path = Path(__file__).with_name(f"dpr_table_hp_{hp_val}.csv")
        try:
            df_out.to_csv(out_csv_path, index=False)
            print("Saved DPR table to", out_csv_path.relative_to(Path.cwd()))
        except Exception as exc:
            print(f"[warning] Could not write DPR table CSV: {exc}")

    # ------------------------------------------------------------------
    # Summary printout (concise) plus DPR mapping table
    # ------------------------------------------------------------------

    print(f"HP {hp_val:>3} | {_nice_profile(prof)}  ->  {vec}")

    if show_baseline and 'mapping_dict' in locals():
        # DPR mapping table
        for strat_key, full_prof in mapping_dict.items():
                print(f"        DPR-map {strat_key:10} → {full_prof}")

        # Which full profiles actually contributed to the baseline average
        if contrib:
            print(f"        contributing full profiles: {contrib}")


# ------------------------------------------------------------------
# Utility: enumerate full-game profiles that map to a DPR profile
# ------------------------------------------------------------------

def list_full_profiles_from_dpr(
    dpr_profile,
    n_mobi: int = 28,
    n_zi: int = 40,
    *,
    n_red: int = 4,
    return_mapping: bool = False,
):
    """Enumerate or *directly construct* 68-player profiles that reduce to
    ``dpr_profile`` under deviation-preserving reduction (DPR).

    The default behaviour (``return_mapping=False``) **preserves the original
    interface**: we brute-force over all feasible count combinations and return
    *every* 68-agent profile that maps to the reduced profile.

    When ``return_mapping=True`` the helper instead follows the analytical
    recipe described in the discussion ‒ *one focal agent at a time* – and
    returns a dictionary that maps **each strategy present in the reduced
    game** to the *unique* full-game profile that the corresponding focal
    agent "sees".  Keys are the canonical strategy labels

        ``{"MOBI_MELO", "MOBI_CDA", "ZI_MELO", "ZI_CDA"}``

    and values are 4-tuples ``(MM, MC, ZM, ZC)`` summing to
    ``(n_mobi, n_zi)``.

    This DPR table is exactly the one illustrated in the worked example:

        >>> list_full_profiles_from_dpr((2, 2, 1, 3), return_mapping=True)
        {
            'MOBI_MELO': (10, 18, 10, 30),
            'MOBI_CDA' : (18, 10, 10, 30),
            'ZI_MELO'  : (14, 14, 1, 39),
            'ZI_CDA'   : (14, 14, 13, 27)
        }

    The analytical path is >100× faster than the brute-force enumeration and
    avoids pulling in *gameanalysis* when you only need the focal-mapping.
    """
    import itertools
    import numpy as np

    # ------------------------------------------------------------------
    # Fast analytical construction (return_mapping=True)
    # ------------------------------------------------------------------

    if return_mapping:
        dpr_profile = tuple(int(x) for x in dpr_profile)

        if len(dpr_profile) != 4:
            raise ValueError("dpr_profile must have length 4 (MOBI_MELO, MOBI_CDA, ZI_MELO, ZI_CDA)")

        if (dpr_profile[0] + dpr_profile[1] != n_red) or (dpr_profile[2] + dpr_profile[3] != n_red):
            raise ValueError("Each role in dpr_profile must sum to n_red")

        mm, mc, zm, zc = dpr_profile  # unpack once

        # Scaling factors (integers by construction)
        scale_same_mobi = (n_mobi - 1) // (n_red - 1)  #  (28-1)/(4-1) = 9
        scale_same_zi   = (n_zi   - 1) // (n_red - 1)  #  (40-1)/(4-1) = 13

        scale_cross_mobi = n_mobi // n_red  # 28/4 = 7
        scale_cross_zi   = n_zi   // n_red  # 40/4 = 10

        mapping = {}

        # Helper to build dict entries succinctly
        def _add(key, vals):
            mapping[key] = tuple(int(v) for v in vals)

        # --------------------------------------------------------------
        # MOBI focal agents
        # --------------------------------------------------------------
        if mm > 0:
            mm_minus1 = mm - 1
            mm_final = mm_minus1 * scale_same_mobi + 1
            mc_final = mc * scale_same_mobi
            zm_final = zm * scale_cross_zi
            zc_final = zc * scale_cross_zi
            _add("MOBI_MELO", (mm_final, mc_final, zm_final, zc_final))

        if mc > 0:
            mc_minus1 = mc - 1
            mm_final = mm * scale_same_mobi
            mc_final = mc_minus1 * scale_same_mobi + 1
            zm_final = zm * scale_cross_zi
            zc_final = zc * scale_cross_zi
            _add("MOBI_CDA", (mm_final, mc_final, zm_final, zc_final))

        # --------------------------------------------------------------
        # ZI focal agents
        # --------------------------------------------------------------
        if zm > 0:
            zm_minus1 = zm - 1
            zm_final = zm_minus1 * scale_same_zi + 1
            zc_final = zc * scale_same_zi
            mm_final = mm * scale_cross_mobi
            mc_final = mc * scale_cross_mobi
            _add("ZI_MELO", (mm_final, mc_final, zm_final, zc_final))

        if zc > 0:
            zc_minus1 = zc - 1
            zm_final = zm * scale_same_zi
            zc_final = zc_minus1 * scale_same_zi + 1
            mm_final = mm * scale_cross_mobi
            mc_final = mc * scale_cross_mobi
            _add("ZI_CDA", (mm_final, mc_final, zm_final, zc_final))

        return mapping

    # ------------------------------------------------------------------
    # Legacy brute-force enumeration (return_mapping=False)
    # ------------------------------------------------------------------

    from gameanalysis.reduction import deviation_preserving as _dpr
    from gameanalysis import rsgame as _rsgame

    # --- sanity checks ----------------------------------------------------
    dpr_profile = tuple(int(x) for x in dpr_profile)
    if len(dpr_profile) != 4:
        raise ValueError("dpr_profile must have length 4 (MOBI_MELO, MOBI_CDA, ZI_MELO, ZI_CDA)")
    if (dpr_profile[0] + dpr_profile[1] != n_red) or (dpr_profile[2] + dpr_profile[3] != n_red):
        raise ValueError("Each role in dpr_profile must sum to n_red")

    # Build a *reduced* RsGame template (4 players/role) used solely by the
    # DPR helper to perform the reduction.
    role_names = ("MOBI", "ZI")
    strat_names = (["MOBI_CDA", "MOBI_MELO"], ["ZI_CDA", "ZI_MELO"])
    red_game = _rsgame.empty_names(role_names, [n_red, n_red], strat_names)

    matches: list[tuple[int, int, int, int]] = []

    # Enumerate all feasible count combinations (cheap – only 1189 cases).
    for mobi_melo in range(n_mobi + 1):
        mobi_cda = n_mobi - mobi_melo

        # Quick pre-filter: ignore combinations that cannot possibly map to the
        # desired reduced counts based on simple scaling heuristics.
        scaled = round(mobi_melo * n_red / n_mobi) if n_mobi > 0 else 0
        if scaled not in (dpr_profile[0], dpr_profile[0] - 1, dpr_profile[0] + 1):
            continue

        for zi_melo in range(n_zi + 1):
            zi_cda = n_zi - zi_melo
            full_prof = np.array([[mobi_melo, mobi_cda, zi_melo, zi_cda]])

            # Run the actual DPR mapping using gameanalysis logic
            red_prof = _dpr.reduce_profiles(red_game, full_prof)
            if red_prof.size == 0:
                continue  # profile not reducible (should not happen here)
            if tuple(int(x) for x in red_prof[0]) == dpr_profile:
                matches.append((mobi_melo, mobi_cda, zi_melo, zi_cda))

    # ------------------------------------------------------------------
    # Optional verbose diagnostics
    # ------------------------------------------------------------------
    if VERBOSE:
        print(f"\n[DPR-lookup] Target reduced profile {dpr_profile}  →  {len(matches)} full-game profiles found")
        if matches:
            mm = [p[0] for p in matches]
            mc = [p[1] for p in matches]
            zm = [p[2] for p in matches]
            zc = [p[3] for p in matches]
            print("   MOBI_MELO: min", min(mm), "max", max(mm))
            print("   MOBI_CDA : min", min(mc), "max", max(mc))
            print("   ZI_MELO  : min", min(zm), "max", max(zm))
            print("   ZI_CDA   : min", min(zc), "max", max(zc))
    return matches


# ------------------------------------------------------------------
# Simple logging helper so we can toggle all messages in one place
# ------------------------------------------------------------------

VERBOSE = True  # flip to False for a quieter run


def _log(*args, **kwargs):
    """Print only when VERBOSE is True."""
    if VERBOSE:
        print(*args, **kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hp", type=int, nargs="*", help="specific HP values to plot", default=[160])
    parser.add_argument("--debug-hp", type=int, nargs="*", default=[],
                    help="holding-period values for which to print detailed equilibrium/ payoff diagnostics")
    parser.add_argument("--hand-eq",
                    help="use the hard-coded historical equilibrium map instead of recomputing pooled equilibria")
    parser.add_argument("--dpr", action="store_true", default=True,
                        help="Generate additional DPR-equilibrium figure (default: on)")
    parser.add_argument("--fill-only", action="store_true", default=True,
                        help="Only perform DPR gap-filling; skip bootstrap and plotting (default: on)")
    parser.add_argument("--print-only", action="store_true", help="Print DPR payoff tables without simulating new profiles")
    parser.add_argument("--check-zero", action="store_true", help="Diagnose profiles with zero or negative payoffs and exit")
    args = parser.parse_args()

    # infer HP set
    def discover_hps() -> list[int]:
        hp_set = set()
        pattern = re.compile(r"holding_period_(\d+)|pilot_egta_run_(\d+)_")
        for root in SEED_ROOTS:
            for fp in glob.glob(f"{root}/**/raw_payoff_data.json", recursive=True):
                m = pattern.search(fp)
                if m:
                    hp = m.group(1) or m.group(2)
                    hp_set.add(int(hp))
        return sorted(hp_set)

    hp_all = set(discover_hps())
    hp_list = sorted(args.hp) if args.hp else sorted(hp_all)
    debug_hps = set(args.debug_hp)
    use_hand_eq = args.hand_eq
    use_dpr = args.dpr
    fill_only = args.fill_only
    print_only = args.print_only
    check_zero = args.check_zero
    if not hp_list:
        print("No holding-period data found – nothing to plot.")
        sys.exit(0)

    print("HP values:", hp_list)

    # --------------------------------------------------------------
    # Build pooled reference games once (complete empirical tables)
    # – but only for HPs we actually need. This avoids unnecessary
    #   work when running with the default single-HP configuration.
    # --------------------------------------------------------------

    if not fill_only:
        print("Building pooled reference games …")

        hp_needed = set(hp_list) | debug_hps  # include any HPs requested for debug dumps

        POOLED_GAMES = {
            hp: build_game_from_profiles(
                list(itertools.chain.from_iterable(
                    collect_profiles_by_seed(hp).values()))
            )
            for hp in hp_needed
        }

        print("…done.")

        # --- temporary sanity check for new welfare metric -------------
        if hp_list:
            g = POOLED_GAMES[hp_list[0]]
            pure = torch.tensor([1,0,1,0], dtype=torch.float32)   # adjust order if needed
            mix  = torch.tensor([0.5,0.5,0.5,0.5])
            print("Sanity check – MOBI/ZI pure welfare:",
                  welfare_multinomial(g, pure))
            print("Sanity check – 50-50 mix welfare:",
                  welfare_multinomial(g, mix))
    else:
        # Dummy placeholder so references to POOLED_GAMES are still valid if
        # any later code path (unexpectedly) needs it, but we avoid the heavy
        # construction.
        POOLED_GAMES = {}

    if fill_only and not print_only:
        print("Gap-fill mode: running DPR reduction and filling missing profiles only …")
        import concurrent.futures, os
        max_workers = min( os.cpu_count() or 1, len(hp_list) )

        def _worker(hp_val):
            try:
                find_pooled_equilibrium_dpr(hp_val, debug_print=False)
            except Exception as exc:
                print(f"[gapfill] HP {hp_val}: {exc}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            list(pool.map(_worker, hp_list))

        print("Gap-fill complete.  Re-run without --fill-only to generate figures.")
        sys.exit(0)

    # pass print_only flag when building DPR map

    #     Figure A: replicate-specific equilibrium
    #    dfA = bootstrap(hp_list, fixed_eq=False)
    #    dfA_sum = summarise_with_z(dfA)
    #    plot_figure(
    #        dfA_sum,
    #        "Deviations vs Vᵣ(x) (bootstrap)",
    #       Path(__file__).with_name("compare_pooled_bootstrap_all_lines_by_hp.png"),
    #    )
    # print("Running Figure 2")
    # # Figure B: fixed pooled equilibrium (full-game)
    # dfB = bootstrap(hp_list, fixed_eq=True, debug_hps=debug_hps, use_hand_eq=use_hand_eq)
    # dfB_sum = summarise_with_z(dfB)
    # plot_figure(
    #     dfB_sum,
    #     "Deviations vs Vᵣ(x̂) (fixed pooled EQ, bootstrap)",
    #     Path(__file__).with_name("compare_pooled_bootstrap_fixed_eq_all_lines_by_hp_full_smooth_final.png"),
    # )

    # -------- NEW Figure C: fixed DPR equilibrium ----------------------
    if use_dpr:
        print("Computing DPR equilibria for figure C …")
        pooled_dpr_map = {hp: find_pooled_equilibrium_dpr(hp, print_only=print_only) for hp in hp_list}
        # Monkey-patch HAND_EQ_DICT style override for bootstrap
        HAND_EQ_DICT_DPR = pooled_dpr_map
        def bootstrap_dpr(hp_list_local):
            # reuse bootstrap() but pass pooled map via use_hand_eq
            global HAND_EQ_DICT
            old = HAND_EQ_DICT
            HAND_EQ_DICT = {hp: t for hp,t in HAND_EQ_DICT_DPR.items()}
            df = bootstrap(hp_list_local, fixed_eq=True, debug_hps=(), use_hand_eq=False)
            HAND_EQ_DICT = old
            return df
        dfC = bootstrap_dpr(hp_list)
        dfC_sum = summarise_with_z(dfC)
        plot_figure(
            dfC_sum,
            "Deviations vs Vᵣ(x̃) (DPR EQ, bootstrap)",
            Path(__file__).with_name("compare_pooled_bootstrap_dpr_eq_all_lines_by_hp.png"),
        )

        # --- NEW quick plot: ZI payoff gap vs HP -----------------------
        gap_df = dfC_sum[(dfC_sum["Role"] == "ZI") & (dfC_sum["Label"] == "Gap")]
        if not gap_df.empty:
            plt.figure(figsize=(7, 4))
            hp_vals = gap_df["HP"]
            y = gap_df["Mean"]
            lo = gap_df["Lo"]
            hi = gap_df["Hi"]
            plt.plot(hp_vals, y, marker="o", color="purple", label="dev_CDA − dev_MELO")
            plt.fill_between(hp_vals, lo, hi, color="purple", alpha=0.2)
            plt.axhline(0, color="black", lw=1)
            plt.xlabel("Holding Period")
            plt.ylabel("Payoff gap (CDA − MELO)")
            plt.title("Bootstrap 95 % CI of ZI payoff gap")
            plt.legend()
            plt.tight_layout()
            gap_png = Path(__file__).with_name("zi_gap_vs_hp.png")
            plt.savefig(gap_png, dpi=300)
            print("Saved", gap_png.relative_to(Path.cwd()))

        # # Print reduced DPR game for each HP
        # for hp_val in hp_list:
        #     if hp_val in pooled_dpr_map:
        #         _print_dpr_game(pooled_dpr_map[hp_val], hp_val)

# ------------------------------------------------------------------
# Save DPR equilibria CSV if rows were collected
# ------------------------------------------------------------------
if DPR_EQ_ROWS:
    out_csv = Path(__file__).with_name("dpr_equilibria.csv")
    pd.DataFrame(DPR_EQ_ROWS).sort_values("HP").to_csv(out_csv, index=False)
    print("Saved", out_csv.relative_to(Path.cwd()))

    # Bump chart for fixed equilibrium
    # plot_bump_chart(dfB_sum, "fixed pooled EQ", Path(__file__).with_name("compare_pooled_bootstrap_fixed_eq_bump_chart_test.png"))

    # ------------------------------------------------------------------
    # Optional diagnostic: count zero / negative payoffs directly in the
    # raw profile data before any aggregation.  Useful to spot systematic
    # patterns.
    # ------------------------------------------------------------------

    if check_zero:
        print("\n=== Zero / negative payoff diagnostic ===")
        for hp_val in hp_list:
            zero_ctr: Counter = Counter()
            neg_ctr: Counter = Counter()
            total_ctr: Counter = Counter()

            # Gather all raw payoff rows for this HP across seed roots
            all_profiles = list(itertools.chain.from_iterable(collect_profiles_by_seed(hp_val).values()))
            for row in all_profiles:
                # Row can be legacy tuple or dict – normalise first
                if isinstance(row, (list, tuple)):
                    if len(row) < 4:
                        continue
                    _, role, strat, payoff = row[:4]
                elif isinstance(row, dict):
                    role = row.get("role") or row.get("Role")
                    strat = row.get("strategy") or row.get("Strategy")
                    payoff = row.get("payoff") or row.get("Payoff")
                else:
                    continue

                if role is None or strat is None or payoff is None:
                    continue

                key = (role, strat)
                total_ctr[key] += 1
                try:
                    val = float(payoff)
                except Exception:
                    continue
                if val == 0:
                    zero_ctr[key] += 1
                elif val < 0:
                    neg_ctr[key] += 1

            print(f"HP {hp_val}")
            for key in sorted(total_ctr.keys()):
                tot = total_ctr[key]
                zc = zero_ctr.get(key, 0)
                nc = neg_ctr.get(key, 0)
                if zc or nc:
                    role, strat = key
                    print(f"  {role}:{strat}  zeros={zc}  negatives={nc}  total={tot}  pct_zero={zc/tot:.3%}  pct_neg={nc/tot:.3%}")
            print()

        print("Diagnostic complete – exiting as requested by --check-zero flag.")
        sys.exit(0)

