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
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import gaussian_filter1d

# Remove early premature GameAnalysis imports (they will be re-imported after sys.path tweaks)

ROOT = Path(__file__).resolve().parent.parent 
sys.path.insert(0, str(ROOT))
# Ensure the bundled GameAnalysis package (marketsim/egta/gameanalysis-old) is importable
sys.path.insert(0, str(ROOT / "marketsim" / "egta" / "gameanalysis-old"))

# Imports that depend on the gameanalysis package must come *after* the path adjustment
from gameanalysis import paygame, nash as ga_nash
from gameanalysis.reduction import deviation_preserving

# ------------------------------------------------------------------
# Patch gameanalysis.utils to work even when SciPy is absent
# ------------------------------------------------------------------
import gameanalysis.utils as _ga_utils
try:
    from scipy.special import comb as _sp_comb  # type: ignore
    _ga_utils.comb = _sp_comb  # type: ignore[attr-defined]
except Exception:
    import math

    def _fallback_comb(n, k, exact=False):  # noqa: D401
        """math.comb wrapper that ignores *exact* arg expected by SciPy API."""
        return math.comb(n, k)

    _ga_utils.comb = _fallback_comb  # type: ignore[attr-defined]


from marketsim.egta.core.game import Game
from marketsim.egta.solvers.equilibria import replicator_dynamics
from marketsim.egta.simulators.melo_wrapper import MeloSimulator

SEED_ROOTS = [
    "result_two_role_still_role_symmetric_3",
   # "result_two_role_still_role_symmetric_3_20k",
    "result_two_role_still_role_symmetric_4",
   # "result_two_role_still_role_symmetric_4_20k",
    "result_two_role_still_role_symmetric_5",
   # "result_two_role_still_role_symmetric_5_20k",
    "result_two_role_still_role_symmetric_6",
   # "result_two_role_still_role_symmetric_6_20k",
    "result_two_role_still_role_symmetric_7",
    #"result_two_role_still_role_symmetric_7_20k",
    # "result_two_role_still_role__1_hbl_symmetric_3",
    # "result_two_role_still_role__1_hbl_symmetric_4", 
    # "result_two_role_still_role__1_hbl_symmetric_5",
    # "result_two_role_still_role__1_hbl_symmetric_6",
    # "result_two_role_still_role__1_hbl_symmetric_7",
    # One-role experiments
   "result_one_role_still_role_symmetric_3",
   "result_one_role_still_role_symmetric_4",
   "result_one_role_still_role_symmetric_5",
   "result_one_role_still_role_symmetric_6",
   "result_one_role_still_role_symmetric_7",

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

# Directory where on-the-fly gap-filled profiles will be stored so subsequent
# runs can reuse them without re-simulating.
GAPFILL_ROOT = "gapfill_profiles"
os.makedirs(GAPFILL_ROOT, exist_ok=True)

# Ensure the gap-fill root is scanned the next time we build the seed→profile map.
if GAPFILL_ROOT not in SEED_ROOTS:
    SEED_ROOTS.append(GAPFILL_ROOT)

# Mapping (role,strategy) → canonical index for fast lookup
IDX_CAN_MAP = {pair: i for i, pair in enumerate(CANONICAL_ORDER)}

N_BOOT = 500
STOP_REG = 1e-4
SMOOTH_CI = False
USE_SMOOTHING = False

# epsilon threshold for RSNE validity
EPS_RSNE = 1e-4

# === NEW: default repetitions for MELO simulation during gap-fill ===
DEFAULT_GAPFILL_REPS: int = 10000  # higher reps → lower variance in deviation payoffs

# Store at module level so helper functions can access/update
SIM_REPS: int = DEFAULT_GAPFILL_REPS

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
    try:
        return Game.from_payoff_data(profiles, normalize_payoffs=False)
    except Exception:
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
    return master  # fallback – guaranteed full support across all pref

def _social_welfare(game: Game, mix: torch.Tensor) -> float:
    """Return total (summed across roles) expected payoff of *mix*.

    We first try the dedicated ``mixture_values`` helper exposed by the
    underlying :class:`RoleSymmetricGame`.  If that is unavailable we
    fall back to the definition via deviation payoffs.
    """
    try:
        # Preferred path – fast and precise
        return float(game.game.mixture_values(mix).sum())  # type: ignore[attr-defined]
    except Exception:
        # Generic (slower) fall-back
        dev = game.deviation_payoffs(mix)
        idx, sw = 0, 0.0
        for strat_list in game.strategy_names_per_role:
            n = len(strat_list)
            seg = slice(idx, idx + n)
            sw += float((mix[seg] * dev[seg]).sum())
            idx += n
        return sw


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
    # --------------------------------------------------------------
    # 1.  Build pooled game
    # --------------------------------------------------------------
    candidates: list[torch.Tensor] = []
    welfare_vals: list[float] = []
    raw_files: list[str] = []
    for root in SEED_ROOTS:
        raw_files.extend(glob.glob(f"{root}/**/holding_period_{hp}/**/raw_payoff_data.json", recursive=True))
        raw_files.extend(glob.glob(f"{root}/**/pilot_egta_run_{hp}_*/**/raw_payoff_data.json", recursive=True))

    try:
        game = build_game_from_profiles(list(itertools.chain.from_iterable(json.load(open(f)) for f in raw_files)))
    except Exception:
        game = None
    if game is None:
        raise RuntimeError(f"Cannot build pooled game for HP={hp}")

    if all(len(lst) == 2 for lst in game.strategy_names_per_role):
        try:
            mix, _ = min(game.find_nash_equilibrium_2x2(), key=lambda x: x[1])
            print(game.role_names, game.strategy_names_per_role)
            print(f"regret =  {game.regret(mix)} for mix {mix}")
            if game.regret(mix) <= regret_max:
                candidates.append(_normalise_per_role(mix.clone()))
                welfare_vals.append(_social_welfare(game, mix))
        except Exception:
            pass 


    for _ in range(n_start): #rd from all the start 
        print(f"Running RD for iteration {_}")
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
        welfare_vals.append(_social_welfare(game, mix))

    
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
    """Simulate each full-game profile and return legacy rows suitable for RoleSymmetricGame.update_with_new_data."""
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
        shade=[250, 500],
        holding_period=hp,
        mobi_strategies=strat_names_per_role[0],
        zi_strategies=strat_names_per_role[1],
        reps=SIM_REPS,
        parallel=True,
        log_profile_details=True,
    )

    legacy_rows_all = []
    iterator = tqdm(profiles_counts, desc=f"Simulating gap profiles HP {hp}", unit="prof") if VERBOSE else profiles_counts
    for cnts in iterator:
        _log("[gapfill] Simulating missing full profile:", cnts.tolist())
        prof_list = _counts_to_profile(cnts, role_names, strat_names_per_role)
        obs = sim.simulate_profile(prof_list)
        legacy_rows = []
        for idx, ((role, strat), payoff) in enumerate(zip(obs.profile_key, obs.payoffs)):
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
):
    """Identify NaNs in reduced game, simulate missing full profiles, and update the master game.

    Returns True iff new data were added (i.e., another reduction pass may help)."""
    import os
    if print_only or os.getenv("DPR_PRINT_ONLY"):
        return False  # skip simulation entirely

    red_profiles = ga_red_full.profiles()
    red_payoffs = ga_red_full.payoffs()
    nan_rows = np.isnan(red_payoffs).any(axis=1)
    if not nan_rows.any():
        return False

    missing_red = red_profiles[nan_rows]
    _log(f"[gapfill] {missing_red.shape[0]} reduced profiles have NaN payoffs – expanding …")

    # Expand to required full-game profiles (baselines + deviations)
    full_needed = deviation_preserving.expand_profiles(ga_full, missing_red)

    # Determine which of these full profiles still have ANY NaN payoff entry
    pay_full = ga_full.get_payoffs(full_needed)

    # Baseline rows = every payoff NaN (no data at all)
    baseline_mask = np.all(np.isnan(pay_full), axis=1)
    # Deviation rows with some missing payoffs
    mask_missing_full = np.isnan(pay_full).any(axis=1)

    full_to_sim = full_needed[baseline_mask | mask_missing_full]
    _log(f"[gapfill] → {full_to_sim.shape[0]} unique 68-player profiles still missing payoffs (including baselines)")

    if full_to_sim.size == 0:
        return False

    # Simulate and update (baselines + deviations)
    legacy_rows = _simulate_profiles_for_counts(
        hp,
        full_to_sim,
        role_names,
        num_players_full,
        strat_names_per_role,
    )
    game_full_ms.game.update_with_new_data(legacy_rows, normalize_payoffs=False)

    # ------------------------------------------------------------------
    # Persist the newly simulated profiles so future runs can reuse them
    # ------------------------------------------------------------------
    gap_dir = os.path.join(GAPFILL_ROOT, f"holding_period_{hp}")
    os.makedirs(gap_dir, exist_ok=True)
    out_path = os.path.join(gap_dir, "raw_payoff_data.json")
    try:
        if os.path.exists(out_path):
            existing = json.load(open(out_path))
            existing.extend(legacy_rows)
            with open(out_path, "w") as f:
                json.dump(existing, f, indent=2)
        else:
            with open(out_path, "w") as f:
                json.dump(legacy_rows, f, indent=2)
    except Exception as exc:
        print(f"[warning] could not write gap-fill data to {out_path}: {exc}")

    _log("[gapfill] Added", len(legacy_rows), "profiles to RoleSymmetricGame table and persisted to disk →", out_path)

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

HAND_EQ_DICT = { #actual
    0:   torch.tensor([0.827, 0.173, 0.000, 1.000]),
    20:  torch.tensor([0.472, 0.528, 0.000, 1.000]),
    40:  torch.tensor([0.573, 0.427, 1.000, 0.000]),
    60:  torch.tensor([0.773, 0.227, 0.000, 1.000]),
    80:  torch.tensor([0.203, 0.797, 1.000, 0.000]),
    100: torch.tensor([0.760, 0.240, 0.000, 1.000]),
    120: torch.tensor([0.208, 0.792, 0.000, 1.000]), #flipped
    140: torch.tensor([1.000, 0.000, 0.000, 1.000]),
    160: torch.tensor([1.000, 0.000, 0.000, 1.000]),
    180: torch.tensor([0.253, 0.747, 0.000, 1.000]),
    200: torch.tensor([1.000, 0.000, 0.000, 1.000]),
    220: torch.tensor([0.724, 0.276, 0.000, 1.000]),
    240: torch.tensor([0.197, 0.803, 0.000, 1.000]),
    260: torch.tensor([0.708, 0.292, 0.000, 1.000]),
    280: torch.tensor([0.485, 0.515, 0.000, 1.000]),
    300: torch.tensor([0.000, 1.000, 0.000, 1.000]),
    320: torch.tensor([0.000, 1.000, 0.000, 1.000]),
}
'''
HAND_EQ_DICT = {
    0:   torch.tensor([1.000, 0.000, 0.000, 1.000]),
    20:  torch.tensor([1.000, 0.000, 0.000, 1.000]),
    40:  torch.tensor([1.000, 0.000, 0.000, 1.000]),
    60:  torch.tensor([1.000, 0.000, 0.000, 1.000]),
    80:  torch.tensor([1.000, 0.000, 0.000, 1.000]),
    100: torch.tensor([0.760, 0.240, 0.000, 1.000]),
    120: torch.tensor([0.1, 0.79, 0.000, 1.000]),
    140: torch.tensor([1.000, 0.000, 1.000, 0.000]),
    160: torch.tensor([1.000, 0.000, 0.000, 1.000]),
    180: torch.tensor([.30, 0.70, 0.000, 1.000]),
    200: torch.tensor([.48, .52, 0.000, 1.000]),
    220: torch.tensor([0.22, 0.78, 0.000, 1.000]),
    240: torch.tensor([0.000, 1.000, 0.000, 1.000]),
    260: torch.tensor([0.000, 1.000, 0.000, 1.000]),
    280: torch.tensor([0.000, 1.000, 0.000, 1.000]),
    300: torch.tensor([0.000, 1.000, 0.000, 1.000]),
    320: torch.tensor([0.000, 1.000, 0.000, 1.000]),
}
'''

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

    # ------------------------------------------------------------------
    # 1. Build *full* GameAnalysis paygame from raw profiles
    # ------------------------------------------------------------------
    profs_full = list(itertools.chain.from_iterable(collect_profiles_by_seed(hp).values()))
    if not profs_full:
        raise RuntimeError(f"No payoff data found for HP={hp}")

    game_full_ms = build_game_from_profiles(profs_full)
    if game_full_ms is None:
        raise RuntimeError(f"Cannot construct full RoleSymmetricGame for HP={hp}")

    role_names = game_full_ms.role_names
    num_players_full = [int(x) for x in game_full_ms.num_players_per_role]
    strat_names_per_role = game_full_ms.strategy_names_per_role

    profiles_arr = game_full_ms.game.rsg_config_table.cpu().numpy().astype(int)
    payoffs_arr = game_full_ms.game.rsg_payoff_table.cpu().numpy().T  # shape (num_profiles, num_strats)

    # Zero-out payoffs for strategies that have zero players in a profile –
    # required by gameanalysis.paygame validation.
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

    # ------------------------------------------------------------------
    # 2. Deviation-preserving reduction *with* auto-filling of missing
    #    profiles by calling the simulator on-demand.
    # ------------------------------------------------------------------
    red_players = np.array([n_red] * len(role_names))

    MAX_REFILL = 3  # at most three rounds of gap filling
    for refill_round in range(1, MAX_REFILL + 1):
        _log(f"[DPR] Reduction round {refill_round} – applying DPR reduction …")
        ga_red_full = deviation_preserving.reduce_game(ga_full, red_players)
        n_nan = np.isnan(ga_red_full.payoffs()).sum()
        _log(f"[DPR]   reduced game: {ga_red_full.num_profiles} profiles, NaN entries={n_nan}")
        # If no NaNs remain, we're done
        if not np.isnan(ga_red_full.payoffs()).any():
            break

        filled = _fill_dpr_missing_payoffs(
            hp,
            ga_full,
            ga_red_full,
            game_full_ms,
            role_names,
            num_players_full,
            strat_names_per_role,
            print_only=print_only,
        )

        if not filled:
            # Could not add new data – give up on further filling
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

    # ------------------------------------------------------------------
    # After gap filling, compute the *proper* DPR game using the library
    # function so that deviation payoffs include the scaling factor
    #   (N_r − 1) / (n_red − 1).
    # ------------------------------------------------------------------

    ga_red_full = deviation_preserving.reduce_game(ga_full, red_players)

    # Library call may still drop rows whose deviation payoffs are all NaN
    # (e.g. corner cases with impossible deviations).  Replace remaining
    # NaNs with 0 so that we always keep a rectangular 25-row table.
    if ga_red_full.num_profiles < n_red**2:  # n_red per role, 2 roles => 25
        pay = ga_red_full.payoffs()
        nan_mask = np.isnan(pay)
        if nan_mask.any():
            pay[nan_mask] = 0.0
            ga_red_full = paygame.game_replace(ga_red_full, ga_red_full.profiles(), pay)

        # --- NEW: ensure that *all* 25 baseline profiles are present ---------
        # Enumerate all combinations of counts for the two–strategy, two–role
        # 4-player-per-role DPR baseline game. Missing rows are appended with
        # zero payoffs so that printing and downstream analysis always see a
        # complete rectangular 25-row payoff table.
        total_needed = (n_red + 1) ** 2  # (0..4) counts per role → 25
        if ga_red_full.num_profiles < total_needed:
            # Build a fast lookup set of already present baseline profiles
            present = {tuple(p.tolist()) for p in ga_red_full.profiles()}
            missing_profiles = []
            for mobi_a in range(n_red + 1):
                zi_a_range = range(n_red + 1)
                for zi_a in zi_a_range:
                    prof = [
                        mobi_a,                       # MOBI strategy A
                        n_red - mobi_a,               # MOBI strategy B
                        zi_a,                         # ZI strategy A
                        n_red - zi_a,                 # ZI strategy B
                    ]
                    if tuple(prof) not in present:
                        missing_profiles.append(prof)

            if missing_profiles:
                # Append zero-payoff rows for the missing baselines
                add_profs = np.array(missing_profiles, dtype=int)
                zero_pays = np.zeros((add_profs.shape[0], pay.shape[1]), dtype=float)
                new_profs = np.vstack([ga_red_full.profiles(), add_profs])
                new_pays = np.vstack([ga_red_full.payoffs(), zero_pays])
                ga_red_full = paygame.game_replace(ga_red_full, new_profs, new_pays)

    ga_red = ga_red_full

    # ------------------------------------------------------------------
    # 3. Solve the reduced game – replicator dynamics (GameAnalysis version)
    # ------------------------------------------------------------------
    keep_mask = np.ones(ga_red.num_strats, dtype=bool)  # all strategies kept
    rng = np.random.default_rng(hp + 42)
    init_mix = rng.random(ga_red.num_strats) + 1e-3  # strictly positive
    for start, size in zip(ga_red.role_starts, ga_red.num_role_strats):
        seg = slice(start, start + size)
        init_mix[seg] /= init_mix[seg].sum()

    # Run replicator dynamics; fall back gracefully if it fails
    try:
        mix_np = ga_nash.replicator_dynamics(ga_red, init_mix, slack=1e-3)
    except ValueError:
        try:
            mix_np = ga_nash.replicator_dynamics(ga_red, init_mix, slack=0.05)
        except ValueError:
            # last numerical fallback – try closed-form 2×2 solver
            try:
                mix_np, _ = ga_red.find_nash_equilibrium_2x2()[0]
                mix_np = mix_np.astype(float)
            except Exception:
                # Fall back to uniform per-role mixture
                mix_np = np.ones(ga_red.num_strats)
                for start, size in zip(ga_red.role_starts, ga_red.num_role_strats):
                    mix_np[start:start+size] /= size


    full_mix_vec = mix_np.copy()

    out = [0.0] * 4
    idx = 0
    for role, strats in zip(role_names, strat_names_per_role):
        for strat in strats:
            can_idx = IDX_CAN_MAP[(role, strat)]
            out[can_idx] = float(full_mix_vec[idx]) if idx < len(full_mix_vec) else 0.0
            idx += 1

    # per-role renorm (guard against numerical drift)
    out[0:2] = (np.array(out[0:2]) / np.sum(out[0:2])).tolist()
    out[2:4] = (np.array(out[2:4]) / np.sum(out[2:4])).tolist()

    # -- optional debug print of the reduced game ---------------------
    if kwargs.get("debug_print", True):
        _print_dpr_game(ga_red, hp)

    return torch.tensor(out)


def bootstrap(hp_list: list[int], fixed_eq: bool = True, debug_hps: Iterable[int] = (), *, use_hand_eq: bool = False):
    from typing import Iterable

    def _debug_dump(game: Game, mix_full: torch.Tensor, hp_val: int):
        """Print mixture + per-role payoff diagnostics for one HP."""
        print(f" DEBUG HP {hp_val}")
        # Build mixture aligned with *game* order
        present_probs = []
        for role, strats in zip(game.role_names, game.strategy_names_per_role):
            for strat in strats:
                present_probs.append(float(mix_full[IDX_CAN_MAP[(role, strat)]]))
        mix_t = torch.tensor(present_probs, dtype=torch.float32)
        # Renormalise per role
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
            # identify MELO/CDA indices within this role
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

    # Build SEED_MAP once for seed-level sampling
    SEED_MAP = {hp: collect_profiles_by_seed(hp) for hp in hp_list}

    # ------------------------------------------------------------------
    # Determine the fixed equilibrium mixture for each HP (if requested)
    # ------------------------------------------------------------------
    if fixed_eq:
        if use_hand_eq:
            print("Using user-supplied historical equilibrium map …")
            pooled_map = {hp: HAND_EQ_DICT[hp] for hp in hp_list if hp in HAND_EQ_DICT}
        else:
            print("Computing fixed pooled equilibria for the requested HP values …")
            pooled_map: dict[int, torch.Tensor] = {
                hp: find_pooled_equilibrium(hp) for hp in hp_list
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
            # game = resample_complete_game(hp, rng)
        #    if game is None:
            #    continue
            print(f"Running Bootstrap {_}")
            # --- seed-level bootstrap sample --------------------------------
            profs = sample_seed_bootstrap(hp, rng, SEED_MAP[hp])
             
            if not profs:
                continue
            game = build_game_from_profiles(profs)
            if game is None:
                continue

            # --------------------------------------------------
            # Fill missing payoff cells from the pooled reference
            # (handle mismatched column counts safely)
            # --------------------------------------------------
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
            CANONICAL_ORDER[1][1] if role == "MOBI" else CANONICAL_ORDER[3][1]: "Dev: CDA",
            CANONICAL_ORDER[0][1] if role == "MOBI" else CANONICAL_ORDER[2][1]: "Dev: MELO",
            "V": "Equilibrium Value",
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
                ax.plot(x, y_plot, color="black", lw=2, marker="s", label=label_map[label])
                ax.fill_between(x, lo_plot, hi_plot, color="grey", alpha=0.25)
            else:
                ax.plot(x, y_plot, marker="o", label=label_map[label])
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


# Helper to pretty print a DPR game (reduced 4-player/role)

def _print_dpr_game(ga_game, hp_val):
    """Print all reduced profiles with their payoffs (one line each).

    Output format:
        HP XX | PROFILE  ->  [pay1, pay2, pay3, pay4]
    where pay* are the deviation payoffs in the game ordering.
    """
    print(f"\n=== DPR reduced game for HP {hp_val} ({ga_game.num_profiles} profiles) ===")

    # convenience look-ups
    rnames = ga_game.role_names
    s_per_role = ga_game.num_role_strats
    rstarts = ga_game.role_starts

    def _nice_profile(p):
        segs = []
        idx = 0
        for r, n in zip(rnames, s_per_role):
            counts = p[idx:idx+n]
            pieces = []
            for s_idx in range(n):
                s_name = ga_game.strat_name(idx + s_idx)
                pieces.append(f"{s_name}={counts[s_idx]}")
            segs.append(f"{r}({', '.join(pieces)})")
            idx += n
        return "; ".join(segs)

    for prof, pay in zip(ga_game.profiles(), ga_game.payoffs()):
        print(f"HP {hp_val:>3} | {_nice_profile(prof)}  ->  {np.round(pay, 4).tolist()}")


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
    parser.add_argument("--hp", type=int, nargs="*", help="specific HP values to plot")
    parser.add_argument("--debug-hp", type=int, nargs="*", default=[],
                    help="holding-period values for which to print detailed equilibrium/ payoff diagnostics")
    parser.add_argument("--hand-eq", action="store_true",
                    help="use the hard-coded historical equilibrium map instead of recomputing pooled equilibria")
    parser.add_argument("--dpr", action="store_true", help="Generate additional DPR-equilibrium figure")
    parser.add_argument("--fill-only", action="store_true", help="Only perform DPR gap-filling; skip bootstrap and plotting")
    parser.add_argument("--print-only", action="store_true", help="Print DPR payoff tables without simulating new profiles")
    parser.add_argument("--reps", type=int, default=DEFAULT_GAPFILL_REPS,
                    help="Number of Monte-Carlo repetitions when simulating missing profiles (default: %(default)s)")
    args = parser.parse_args()

    # Propagate to module-level so helpers can see the user override
    SIM_REPS = args.reps

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
    if not hp_list:
        print("No holding-period data found – nothing to plot.")
        sys.exit(0)

    print("HP values:", hp_list)

    # --------------------------------------------------------------
    # Build pooled reference games once (complete empirical tables)
    # --------------------------------------------------------------
    print("Building pooled reference games …")
    POOLED_GAMES = {
        hp: build_game_from_profiles(
                list(itertools.chain.from_iterable(
                    collect_profiles_by_seed(hp).values()))
            )
        for hp in hp_all
    }
    print("…done.")

    if fill_only:
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

        # Figure A: replicate-specific equilibrium
    #    dfA = bootstrap(hp_list, fixed_eq=False)
    #    dfA_sum = summarise_with_z(dfA)
    #    plot_figure(
        #    dfA_sum,
        #    "Deviations vs Vᵣ(x) (bootstrap)",
        #   Path(__file__).with_name("compare_pooled_bootstrap_all_lines_by_hp.png"),
    #    )
    # print("Running Figure 2")
    Figure B: fixed pooled equilibrium (full-game)
    # dfB = bootstrap(hp_list, fixed_eq=True, debug_hps=debug_hps, use_hand_eq=use_hand_eq)
    # dfB_sum = summarise_with_z(dfB)
    # plot_figure(
        # dfB_sum,
        # "Deviations vs Vᵣ(x̂) (fixed pooled EQ, bootstrap)",
        # Path(__file__).with_name("compare_pooled_bootstrap_fixed_eq_all_lines_by_hp_full_smooth_final.png"),
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
            df = bootstrap(hp_list_local, fixed_eq=True, debug_hps=(), use_hand_eq=True)
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

        # Print reduced DPR game for each HP
        for hp_val in hp_list:
            if hp_val in pooled_dpr_map:
                _print_dpr_game(pooled_dpr_map[hp_val], hp_val)
    # Bump chart for fixed equilibrium
    # plot_bump_chart(dfB_sum, "fixed pooled EQ", Path(__file__).with_name("compare_pooled_bootstrap_fixed_eq_bump_chart_test.png"))

