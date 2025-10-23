# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import json, glob, re, os, sys
from pathlib import Path
from collections import defaultdict
from statistics import mean, stdev
from functools import lru_cache

import numpy as np  # for bootstrap resampling
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch

from multiprocessing import Pool, cpu_count, set_start_method
from scipy.ndimage import gaussian_filter1d
import itertools

# Toggle for optional smoothing of the MEAN curves (deviation & V_r). Only
# affects the visual plot – bootstrap statistics are untouched.
USE_SMOOTHING = True
# Optional: smooth the confidence-interval (Lo/Hi) envelopes in bootstrap
# figures (Figure 9 & 10).  Set to False if you prefer the raw, unsmoothed
# step-like CIs.
SMOOTH_CI = True

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from marketsim.game.role_symmetric_game import RoleSymmetricGame
from marketsim.egta.core.game import Game
'''
SEED_ROOTS = [
    "result_two_role_still_role_symmetric_3",
    "result_two_role_still_role_symmetric_4",
    "result_two_role_still_role_symmetric_5",
    "result_two_role_still_role_symmetric_6",
    "result_two_role_still_role_symmetric_7",
]

'''
SEED_ROOTS = [
    "result_two_role_still_role_symmetric_3",
    "result_two_role_still_role_symmetric_4",
    "result_two_role_still_role_symmetric_5",
    "result_two_role_still_role_symmetric_6",
    "result_two_role_still_role_symmetric_7",
    "result_2_role_still_role_symmetric_2_strategies_test_price_based_updates_false",
    "result_2_role_still_role_symmetric_2_strategies_test_price_based_updates_false_normalize_payoffs_false_2"
    #"result_two_role_still_role_symmetric_3_20k",
    #"result_two_role_still_role_symmetric_4_20k",
    #"result_two_role_still_role_symmetric_5_20k",
   # "result_two_role_still_role_symmetric_6_20k",
   # "result_two_role_still_role_symmetric_7_20k",
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


# helper --------------------------------------------------------------------
def _interior_2x2(game, eps=1e-12):
    """Analytic interior Nash for a 2×2 game using get_payoff_matrix()."""
    mat0 = game.get_payoff_matrix(0).flatten()
    mat1 = game.get_payoff_matrix(1).flatten()
    a11,a12,a21,a22 = mat0
    b11,b12,b21,b22 = mat1  # row-major after flatten
    denom_p = b22 - b21 - b12 + b11
    denom_q = a22 - a12 - a21 + a11
    if abs(denom_p) > eps and abs(denom_q) > eps:
        p = (b22 - b21) / denom_p       # MOBI-plays-CDA prob
        q = (a22 - a12) / denom_q       # ZI-plays-CDA prob
        if eps < p < 1-eps and eps < q < 1-eps:
            return torch.tensor([p, 1-p, q, 1-q])
    return None

def extract_minreg_mixture(eq_file: str):
    with open(eq_file) as fh:
        data = json.load(fh)
    best = min(data, key=lambda x: x["regret"])
    return torch.tensor(best["mixture"], dtype=torch.float32)


def build_rsg_from_obs(obs_files):
    """Create RoleSymmetricGame from list of observations.json paths."""
    if not obs_files:
        return None
    config_rows = []
    payoff_rows = None  # will build later
    role_names = None
    strategy_names_per_role = None
    num_players_per_role = None

    try:
        for f in obs_files:
            with open(f) as fh:
                obs_list = json.load(fh)
            for obs in obs_list:
                prof = obs["profile"] if isinstance(obs, dict) else obs[0]
                payoffs = obs["payoffs"] if isinstance(obs, dict) else None
                # Build role/strategy metadata from first profile
                if role_names is None:
                    role_names = []
                    strat_per_role = defaultdict(list)
                    cnt_per_role = defaultdict(int)
                    for role, strat in prof:
                        if role not in role_names:
                            role_names.append(role)
                        if strat not in strat_per_role[role]:
                            strat_per_role[role].append(strat)
                        cnt_per_role[role] += 1
                    strategy_names_per_role = [strat_per_role[r] for r in role_names]
                    num_players_per_role = [cnt_per_role[r] for r in role_names]
                    num_strats = sum(len(lst) for lst in strategy_names_per_role)
                    payoff_rows = [[] for _ in range(num_strats)]

                # count players per strategy to form config row
                counts = [0] * sum(len(lst) for lst in strategy_names_per_role)
                strat_index = {}
                idx = 0
                for lst in strategy_names_per_role:
                    for s in lst:
                        strat_index[s] = idx
                        idx += 1
                # Skip profiles containing unseen strategies
                unseen = False
                for role, strat in prof:
                    if strat not in strat_index:
                        unseen = True; break
                if unseen:
                    continue
                for role, strat in prof:
                    counts[strat_index[strat]] += 1
                config_rows.append(counts)

                # payoffs list is same order as prof; aggregate per strategy
                if payoffs is None:
                    continue
                # average payoff per strategy in this profile
                payoff_sum = [0.0] * len(counts)
                payoff_cnt = [0] * len(counts)
                for (role, strat), pf in zip(prof, payoffs):
                    idx = strat_index[strat]
                    payoff_sum[idx] += pf
                    payoff_cnt[idx] += 1
                for i in range(len(counts)):
                    if payoff_cnt[i]:
                        payoff_rows[i].append(payoff_sum[i] / payoff_cnt[i])
                    else:
                        payoff_rows[i].append(float("nan"))
    except KeyError as e:
        print(f"build_rsg_from_obs: unseen strategy {e}; falling back to raw data")
        return None

    cfg_tensor = torch.tensor(np.array(config_rows, dtype=np.float32))
    pay_tensor = torch.tensor(np.array([np.array(r, dtype=np.float32) for r in payoff_rows]))

    rsg = RoleSymmetricGame(
        role_names=role_names,
        num_players_per_role=num_players_per_role,
        strategy_names_per_role=strategy_names_per_role,
        rsg_config_table=cfg_tensor,
        rsg_payoff_table=pay_tensor,
    )
    return Game(rsg)

# ----------------------------- NEW ---------------------------------------
# Fallback loader for raw_payoff_data.json which contains the underlying
# list-of-profiles format produced by the simulator.  Each profile is a list
# of 4-tuples (player_id, role, strategy, payoff).

def build_game_from_raw(raw_files):
    if not raw_files:
        return None
    payoff_data = []  # list of profiles (list[list[tuple]])
    for rf in raw_files:
        try:
            with open(rf) as fh:
                data = json.load(fh)
            # data is a list of profiles; extend directly
            payoff_data.extend(data)
        except (json.JSONDecodeError, OSError) as e:
            print(f"Warning: could not parse {rf}: {e}")
            continue
    if not payoff_data:
        return None
    try:
        return Game.from_payoff_data(payoff_data, normalize_payoffs=False)
    except Exception as exc:
        print(f"Failed to build game from raw payoff data (len={len(payoff_data)}): {exc}")
        return None

# ---------------------------------------------------------------------------

# Per-strategy records (deviation payoffs)
strat_records = []          # Figure 1 – vs each seed's equilibrium mixture
avg_strat_records = []      # Figure 2 – vs HP-wise average mixture

# Role-level records (expected payoff in equilibrium)
# Role-level records (single payoff per role)
role_records = []  # first figure (equilibria of individual seeds)
avg_role_records = []  # second figure (average mixture across seeds)
agg_records = []  # rows for aggregated-game equilibria figure

hp_set = set()
for root in SEED_ROOTS:
    for eq_file in glob.glob(f"{root}/**/equilibria*.json", recursive=True):
        m = re.search(r"holding_period_(\d+)", eq_file) or re.search(r"pilot_egta_run_(\d+)", eq_file)
        if m:
            hp_set.add(int(m.group(1)))

# Keep only holding periods in 20-second increments (0,20,…,320)
hp_set = {hp for hp in hp_set if hp % 20 == 0 and hp <= 320}

# ---------------------- BOOTSTRAP OPTIONS ----------------------------------
DO_BOOT   = True        # set False to skip bootstrap pass
N_BOOT    = 1        # bootstrap iterations per HP (for Figure 9)
ALPHA_CI  = 0.05        # 95% confidence interval
STOP_REG  = 1e-4        # regret tolerance for replicator on raw payoffs
FIGURE_9_ONLY = True   # run only the bootstrap required for Figure 9

# ---------------------------------------------------------------------------

# ----- data helpers -------------------------------------------------------

# Return mapping seed_root -> list[profiles]
def collect_profiles_by_seed(hp:int):
    seed_to_profiles = {}
    for root in SEED_ROOTS:
        profs=[]
        rp1=glob.glob(f"{root}/**/holding_period_{hp}/**/raw_payoff_data.json",recursive=True)
        rp2=glob.glob(f"{root}/**/pilot_egta_run_{hp}_*/**/raw_payoff_data.json",recursive=True)
        for rf in rp1+rp2:
            try:
                profs.extend(json.load(open(rf)))
            except Exception:
                continue
        if profs:
            seed_to_profiles[root]=profs
    return seed_to_profiles

# bootstrap function returning list of dict records per HP
def bootstrap_pooled_hp(hp: int):
    """Bootstrap over *profiles* from raw payoff data while guaranteeing full 2×2 support.

    For each iteration we draw |P| profiles with replacement (|P| = total profiles in the
    pooled data). If the resample happens to omit any of the four (role,strategy) pairs we
    simply redraw.  With a modest number of retries this reliably produces valid games and
    avoids the earlier zero-success problem.
    """

    seed_map = collect_profiles_by_seed(hp)
    if not seed_map:
        return []

    seed_keys = list(seed_map.keys())
    n_seeds = len(seed_keys)

    vals = defaultdict(list)
    successes = 0
    rng = np.random.default_rng()

    while successes < N_BOOT:
        
        chosen = rng.integers(0, n_seeds, size=n_seeds)  
        sample_profiles = []
        for idx in chosen:
            sample_profiles.extend(seed_map[seed_keys[idx]])

        # Attempt to construct a Game from the resampled profiles (outside the for-loop)
        try:
            game = Game.from_payoff_data(sample_profiles, normalize_payoffs=False)
        except Exception as e:
            if successes == 0:
                print(f"Bootstrap HP {hp}: Game construction failed – {e}")
            continue

        mix = None
        try:
            if all(len(lst) == 2 for lst in game.strategy_names_per_role):
                eqs = game.find_nash_equilibrium_2x2()
                if eqs:
                    mix = eqs[0][0]
        except Exception:
            mix = None

        # ---- Otherwise run a slim replicator --------------------------
        if mix is None:
            from marketsim.egta.solvers.equilibria import replicator_dynamics
            init = torch.ones(game.num_strategies) / game.num_strategies
            try:
                mix = replicator_dynamics(
                    game,
                    init,
                    iters=1000,
                    use_multiple_starts=False,
                    converge_threshold=STOP_REG,
                )
            except Exception as e:
                if successes == 0:
                    print(f"Bootstrap HP {hp}: Replicator failed – {e}")
                mix = None

        if mix is None:
            # fallback: empirical frequency mixture
            mix_counts = [0.0] * game.num_strategies
            strat_index = {}
            idxg = 0
            for role, strats in zip(game.role_names, game.strategy_names_per_role):
                for s in strats:
                    strat_index[(role, s)] = idxg
                    idxg += 1
            for prof in sample_profiles:
                for _, role, strat, _ in prof:
                    if (role, strat) in strat_index:
                        mix_counts[strat_index[(role, strat)]] += 1
            # normalise per role
            start = 0
            for role, strats in zip(game.role_names, game.strategy_names_per_role):
                role_total = sum(mix_counts[start : start + len(strats)])
                if role_total > 0:
                    for j in range(len(strats)):
                        mix_counts[start + j] /= role_total
                else:
                    for j in range(len(strats)):
                        mix_counts[start + j] = 1.0 / len(strats)
                start += len(strats)
            mix = torch.tensor(mix_counts, dtype=torch.float32)

        # ---- Record deviation payoffs --------------------------------
        dev = game.deviation_payoffs(mix)
        idx = 0
        for role, strats in zip(game.role_names, game.strategy_names_per_role):
            seg = slice(idx, idx + len(strats))
            Vr = float((mix[seg] * dev[seg]).sum())
            vals[(role, 'V')].append(Vr)
            for strat in strats:
                vals[(role, strat)].append(float(dev[idx]))
                idx += 1

        successes += 1  # count this bootstrap replicate

    print(f"Bootstrap HP {hp}: successes={successes}/{N_BOOT}, curves={len(vals)}")

    records=[]
    for (role,label), lst in vals.items():
        if lst:
            mu=float(np.mean(lst)); lo,hi=np.quantile(lst,[ALPHA_CI/2,1-ALPHA_CI/2])
            records.append({'Holding Period':hp,'Role':role,'Label':label,'Mean':mu,'Lo':float(lo),'Hi':float(hi)})
    return records

# --------------------- GUARD AGAINST WORKER EXECUTION -----------------------

import multiprocessing as _mp
# When this module is imported inside a multiprocessing worker (Pool), the
# process name is not 'MainProcess'.  Exit early so that only definitions are
# loaded and heavy top-level code below is skipped.  The parent process
# controls execution via explicit calls (e.g. Pool.map).

if _mp.current_process().name != "MainProcess":
    # Prevent accidental recursive pool creation and duplicate plotting in
    # worker processes.  The parent just needs the functions.
    raise SystemExit

# Define canonical order for MOBI and ZI strategies globally so helpers can reuse it

CANONICAL_ORDER = [
    ("MOBI", "MOBI_0_100_shade0_0"),
    ("MOBI", "MOBI_100_0_shade250_500"),
    ("ZI",   "ZI_0_100_shade250_500"),
    ("ZI",   "ZI_100_0_shade250_500"),
]

from functools import lru_cache

@lru_cache(maxsize=None)
def find_pooled_equilibrium(hp:int, n_starts:int=100, tol:float=1e-4, rd_iters:int=1000):
    """Return a pooled-game equilibrium mixture for one holding period (HP).
    Steps:
      1. Build the aggregated game from all seeds' raw payoff data.
      2. Run replicator dynamics from many random starts to collect candidate ε-RSNE.
      3. Cluster candidates by an absolute 0.01 tolerance per component.
      4. Return the *average* of cluster representatives (smoother than picking one).
    The result is cached per HP so subsequent calls are cheap."""
    # ------------------------------------------------------------------
    # Build aggregated game across all seeds for this HP
    agg_raw_files = []
    for root in SEED_ROOTS:
        agg_raw_files.extend(glob.glob(f"{root}/**/holding_period_{hp}/**/raw_payoff_data.json", recursive=True))
        agg_raw_files.extend(glob.glob(f"{root}/**/pilot_egta_run_{hp}_*/**/raw_payoff_data.json", recursive=True))
    game = build_game_from_raw(agg_raw_files)
    if game is None:
        raise RuntimeError(f"Could not build pooled game for HP={hp}")

    from marketsim.egta.solvers.equilibria import replicator_dynamics

    # ------------------------------------------------------------------
    # Helper: per-role normalisation + floor to avoid rounding a prob to 0
    # ------------------------------------------------------------------

    def _normalise_per_role(vec: torch.Tensor, floor: float = 0.005):
        """Ensure each role's probs sum to 1 and each component ≥ floor."""
        start = 0
        role_sizes = getattr(game, 'num_strategies_per_role', None)
        if role_sizes is None:
            role_sizes = getattr(game, 'num_strats_per_role', None)
        if role_sizes is None:
            role_sizes = [len(lst) for lst in game.strategy_names_per_role]

        for n in role_sizes:
            seg = vec[start:start + n]
            seg = torch.clamp(seg, min=floor)
            seg /= seg.sum()
            vec[start:start + n] = seg
            start += n
        return vec

    # ------------------------------------------------------------------
    # Collect candidate equilibria
    #   • all analytic 2×2 solutions (if applicable)
    #   • n_starts replicator-dynamics runs
    # ------------------------------------------------------------------

    cand: list[tuple[torch.Tensor, float]] = []  # (mix, social_welfare)

    def _social_welfare(mix: torch.Tensor) -> float:
        dev = game.deviation_payoffs(mix)
        sw = 0.0; idx = 0
        for strats in game.strategy_names_per_role:
            seg = slice(idx, idx + len(strats))
            sw += float((mix[seg] * dev[seg]).sum())
            idx += len(strats)
        return sw

    # ---- analytic 2×2 equilibria ------------------------------------
    if all(len(lst) == 2 for lst in game.strategy_names_per_role):
        try:
            for mix, reg in game.find_nash_equilibrium_2x2():
                if reg <= STOP_REG:
                    mix = _normalise_per_role(mix.clone())
                    cand.append((mix, _social_welfare(mix)))
        except Exception:
            pass  # continue with RD starts

    # ---- replicator-dynamics starts ---------------------------------
    for _ in range(n_starts):
        rand_init = torch.rand(game.num_strategies)
        rand_init = _normalise_per_role(rand_init)
        try:
            mix = replicator_dynamics(
                game,
                rand_init,
                iters=rd_iters,
                converge_threshold=tol,
                use_multiple_starts=True
            )
            mix = _normalise_per_role(mix)
            #if float(game.regret(mix)) <= 1e-3:
               # cand.append((mix, _social_welfare(mix)))
        except Exception:
            continue

    # ------------------------------------------------------------------
    # Ensure we have at least one candidate – use per-role uniform mix as
    # last-chance fallback (should be very rare but avoids empty list).
    # ------------------------------------------------------------------
   # if not cand:
      #  mix = torch.ones(game.num_strategies)
      #  mix = _normalise_per_role(mix)
      #  cand.append((mix, _social_welfare(mix)))

    # Deduplicate with absolute tolerance 0.01 per strategy component (per-role).
    clusters: list[list[tuple[torch.Tensor, float]]] = []
    for m, sw in cand:
        placed = False
        for cluster in clusters:
            rep_mix = cluster[0][0]
            if torch.max(torch.abs(m - rep_mix)) <= 0.005 + 1e-12:
                cluster.append((m, sw))
                placed = True
                break
        if not placed:
            clusters.append([(m, sw)])

    uniq = [(cluster[0][0], np.median([sw for _mix, sw in cluster])) for cluster in clusters]
    # Sort by SW and pick median
    uniq_sorted = sorted(uniq, key=lambda x: x[1])
    # If more than one representative pick the median-SW one; if many reps
    # tie, average them (smoother).
    mid_idx = len(uniq_sorted) // 2
    median_sw = uniq_sorted[mid_idx][1]
    reps_same_sw = [m for m, sw in uniq_sorted if np.isclose(sw, median_sw)]
    if len(reps_same_sw) == 1:
        chosen_mix = reps_same_sw[0]
    else:
        reps_same_sw.sort(key=lambda m: float(m[0]))   # sort by MELO share
        chosen_mix = reps_same_sw[len(reps_same_sw)//2]
    return chosen_mix

for hp in sorted(hp_set):
    per_seed_role_vals = defaultdict(list)  # role -> list[float] over seeds
    per_seed_payoffs = defaultdict(list)    # (role,strat) kept for third-figure use
    seed_games_and_mixes = []  # store tuples (game, canonical_mix_vec)

    for root in SEED_ROOTS:
        eq_files = glob.glob(
            f"{root}/**/holding_period_{hp}/**/equilibria*.json", recursive=True)
        if not eq_files:
            eq_files = glob.glob(
                f"{root}/**/pilot_egta_run_{hp}_*/**/equilibria*.json", recursive=True)
        if not eq_files:
            continue
        # Extract mixture from the first *non-empty* equilibria file we find
        eq_path = None
        eq_data = []
        for cand in eq_files:
            try:
                with open(cand) as _fh:
                    data_candidate = json.load(_fh)
                if data_candidate:
                    eq_path = cand
                    eq_data = data_candidate
                    break
            except (json.JSONDecodeError, OSError):
                continue
        if not eq_data:
            continue  # no usable equilibria file in this seed/HP
        best_entry = min(eq_data, key=lambda x: x.get("regret", 0.0))

        # Prefer the *non-corner* equilibrium with the lowest regret.
        # A corner equilibrium is one that is pure for *all* roles (prob≈0 or 1).
        def is_corner(mix_list, role_sizes, tol=4):
            """Return True if every role plays a single strategy with prob≥1-tol."""
            start = 0
            for sz in role_sizes:
                seg = mix_list[start : start + sz]
                start += sz
                if len(seg) == 0:
                    return True  # degenerate – treat as corner
                # role is corner if exactly one component carries (1-tol) mass or more
                corner_this_role = sum(p >= 1.0 - tol for p in seg) == 1
                if not corner_this_role:
                    return False
            return True

        # Build the game first so we know the number of strategies per role.
        obs_files = glob.glob(
            f"{root}/**/holding_period_{hp}/**/observations.json", recursive=True
        )
        if not obs_files:
            obs_files = glob.glob(
                f"{root}/**/pilot_egta_run_{hp}_*/**/observations.json", recursive=True
            )
        # If observation files are missing, fall back to raw_payoff_data.json
        game = None
        if obs_files:
            try:
                game = build_rsg_from_obs(obs_files)
            except Exception as e:
                print(f"Failed to build game from observations for HP={hp} seed {root}: {e}")
                game = None
        if game is None:
            raw_files = glob.glob(
                f"{root}/**/holding_period_{hp}/**/raw_payoff_data.json", recursive=True
            )
            if not raw_files:
                raw_files = glob.glob(
                    f"{root}/**/pilot_egta_run_{hp}_*/**/raw_payoff_data.json", recursive=True
                )
            game = build_game_from_raw(raw_files)
        if game is None:
            continue

        # Number of strategies per role
        if hasattr(game, "num_strategies_per_role"):
            role_sizes = list(game.num_strategies_per_role)
        else:
            # Derive from strategy_names_per_role if not exposed by wrapper
            role_sizes = [len(lst) for lst in game.strategy_names_per_role]

        # Select the first non-corner equilibrium; if none, fall back to min-regret.
        selected_entry = None
        for entry in sorted(eq_data, key=lambda x: x.get("regret", 0.0)):
            mv = entry.get("mixture") or entry.get("mixture_vector")
            if mv is None:
                continue
            if not is_corner(mv, role_sizes):
                selected_entry = entry
                break
        if selected_entry is None:
            selected_entry = best_entry

        # --- Build mixture aligned with the strategies present in *game* ---
        mix_dict_by_role = {}  # will fill for canonical refs
        canonical_mix_vec = [0.0, 0.0, 0.0, 0.0]  # in CANONICAL_ORDER
        if "mixture_by_role" in selected_entry:
            mb = selected_entry["mixture_by_role"]
            mix_vals = []
            for role, strats in zip(game.role_names, game.strategy_names_per_role):
                role_probs = mb.get(role, {})
                for strat in strats:
                    mix_vals.append(float(role_probs.get(strat, 0.0)))
            # Also fill canonical dict
            for idx, (role, strat) in enumerate(CANONICAL_ORDER):
                p = float(mb.get(role, {}).get(strat, 0.0))
                mix_dict_by_role.setdefault(role, {})[strat] = p
                canonical_mix_vec[idx] = p
            mix = torch.tensor(mix_vals, dtype=torch.float32)
            # Renormalise per role (some mass may have been lost if we dropped unseen strategies)
            start = 0
            for r_idx, strats in enumerate(game.strategy_names_per_role):
                seg = mix[start : start + len(strats)]
                s = seg.sum()
                if s > 1e-12:
                    mix[start : start + len(strats)] = seg / s
                else:  # all zeros – default to uniform
                    mix[start : start + len(strats)] = 1.0 / len(strats)
                start += len(strats)
        else:
            mix_vec = selected_entry.get("mixture") or selected_entry.get("mixture_vector")
            if mix_vec is None:
                continue
            mix_full = torch.tensor(mix_vec, dtype=torch.float32)
            if mix_full.numel() == game.num_strategies:
                mix = mix_full
            else:
                # Slice: assume original order role-wise; keep positions for present strats
                present_mix = []
                pos = 0
                for role, strats in zip(game.role_names, game.strategy_names_per_role):
                    needed = len(strats)
                    if pos + needed > mix_full.numel():
                        present_mix.extend([0.0] * needed)
                    else:
                        present_mix.extend(mix_full[pos : pos + needed].tolist())
                    pos += needed
                mix = torch.tensor(present_mix, dtype=torch.float32)
                # Renormalise per role
                start = 0
                for strats in game.strategy_names_per_role:
                    seg = mix[start : start + len(strats)]
                    s = seg.sum()
                    if s > 1e-12:
                        mix[start : start + len(strats)] = seg / s
                    else:
                        mix[start : start + len(strats)] = 1.0 / len(strats)
                    start += len(strats)

            # build mix_dict_by_role from mix_full using canonical ordering
            mix_dict_by_role = {}
            for idx, (role, strat) in enumerate(CANONICAL_ORDER):
                prob = mix_full[idx] if idx < len(mix_full) else 0.0
                mix_dict_by_role.setdefault(role, {})[strat] = float(prob)
                canonical_mix_vec[idx] = float(prob)

        # Compute deviation payoffs versus the selected equilibrium mixture
        dev_pay = game.deviation_payoffs(mix)

        # 1) store deviation payoffs per-strategy (still used by third figure)
        idx = 0
        for role_name, strats in zip(game.role_names, game.strategy_names_per_role):
            seg_size = len(strats)
            # deviation payoffs for all strategies in this role
            for s_name in strats:
                per_seed_payoffs[(role_name, s_name)].append(float(dev_pay[idx]))
                idx += 1

        # 2) compute role-average payoff V_r(x) = Σ_k x_{rk} u_{rk}(x)
        idx_r = 0
        for role_name, strats in zip(game.role_names, game.strategy_names_per_role):
            seg_size = len(strats)
            mix_seg = mix[idx_r : idx_r + seg_size]
            dev_seg = dev_pay[idx_r : idx_r + seg_size]
            role_value = float((mix_seg * dev_seg).sum().item())
            per_seed_role_vals[role_name].append(role_value)
            idx_r += seg_size

        # store for average-mixture pass later
        seed_games_and_mixes.append((game, canonical_mix_vec))

    # Aggregate role-level payoff over seeds and store
    for role, lst in per_seed_role_vals.items():
        if lst:
            mu = mean(lst)
            sd = stdev(lst) if len(lst) > 1 else 0.0
            role_records.append({
                "Holding Period": hp,
                "Role": role,
                "Mean": mu,
                "Std": sd,
            })

    # Aggregate per-strategy deviation payoffs for Figure 1
    for (role, strat), lst in per_seed_payoffs.items():
        if lst:
            mu = mean(lst)
            sd = stdev(lst) if len(lst) > 1 else 0.0
            strat_records.append({
                "Holding Period": hp,
                "Role": role,
                "Strategy": strat,
                "Mean": mu,
                "Std": sd,
            })

    # ------- second figure: average mixture for this HP --------
    if seed_games_and_mixes:
        # 1) Compute average canonical mixture over seeds
        count_seeds = len(seed_games_and_mixes)
        avg_vec = [0.0, 0.0, 0.0, 0.0]
        for _, vec in seed_games_and_mixes:
            for i in range(4):
                avg_vec[i] += vec[i]
        avg_vec = [v / count_seeds for v in avg_vec]

        # Renormalise per role (first two entries MOBI, last two ZI)
        mobi_sum = avg_vec[0] + avg_vec[1]
        zi_sum   = avg_vec[2] + avg_vec[3]
        if mobi_sum > 1e-12:
            avg_vec[0] /= mobi_sum; avg_vec[1] /= mobi_sum
        else:
            avg_vec[0] = avg_vec[1] = 0.5
        if zi_sum > 1e-12:
            avg_vec[2] /= zi_sum; avg_vec[3] /= zi_sum
        else:
            avg_vec[2] = avg_vec[3] = 0.5

        # 2) Evaluate each seed's game versus this average mixture
        per_seed_role_vals_avg = defaultdict(list)  # role → list[float]
        per_seed_payoffs_avg   = defaultdict(list)  # (role,strat) → list[float]

        for game, _vec in seed_games_and_mixes:
            # Build mixture aligned with this game
            present_probs = []
            for role, strats in zip(game.role_names, game.strategy_names_per_role):
                for strat in strats:
                    # find index in canonical order
                    idx = None
                    for j,(r,s) in enumerate(CANONICAL_ORDER):
                        if r==role and s==strat:
                            idx=j; break
                    p = avg_vec[idx] if idx is not None else 0.0
                    present_probs.append(p)
            mix_t = torch.tensor(present_probs, dtype=torch.float32)
            # renormalise per role
            start = 0
            for strats in game.strategy_names_per_role:
                seg = mix_t[start:start+len(strats)]
                s_sum = seg.sum()
                if s_sum>1e-12:
                    mix_t[start:start+len(strats)] = seg/ s_sum
                else:
                    mix_t[start:start+len(strats)] = 1.0/len(strats)
                start += len(strats)

            dev_avg = game.deviation_payoffs(mix_t)

            # accumulate per-strategy deviation payoffs (for Figure 2)
            idxg2 = 0
            for role_name, strats in zip(game.role_names, game.strategy_names_per_role):
                for strat in strats:
                    per_seed_payoffs_avg[(role_name, strat)].append(float(dev_avg[idxg2]))
                    idxg2 += 1

            # accumulate role values V_r(¯x)
            idxg = 0
            for role_name, strats in zip(game.role_names, game.strategy_names_per_role):
                seg_size = len(strats)
                mix_seg = mix_t[idxg : idxg + seg_size]
                dev_seg = dev_avg[idxg : idxg + seg_size]
                role_val = float((mix_seg * dev_seg).sum().item())
                per_seed_role_vals_avg[role_name].append(role_val)
                idxg += seg_size

        # Aggregate per-strategy avg-mix deviation payoffs
        for (role, strat), lst in per_seed_payoffs_avg.items():
            if lst:
                mu = mean(lst)
                sd = stdev(lst) if len(lst) > 1 else 0.0
                avg_strat_records.append({
                    "Holding Period": hp,
                    "Role": role,
                    "Strategy": strat,
                    "Mean": mu,
                    "Std": sd,
                })

        # Aggregate role-level avg-mix payoffs
        for role, lst in per_seed_role_vals_avg.items():
            if lst:
                mu = mean(lst)
                sd = stdev(lst) if len(lst) > 1 else 0.0
                avg_role_records.append({
                    "Holding Period": hp,
                    "Role": role,
                    "Mean": mu,
                    "Std": sd,
                })

    # ------- third figure: equilibria of aggregated game --------
    # Build aggregated game from *all* raw payoff files across seeds for this HP
    agg_raw_files = []
    for root in SEED_ROOTS:
        agg_raw_files.extend(glob.glob(f"{root}/**/holding_period_{hp}/**/raw_payoff_data.json", recursive=True))
        agg_raw_files.extend(glob.glob(f"{root}/**/pilot_egta_run_{hp}_*/**/raw_payoff_data.json", recursive=True))

    agg_game = build_game_from_raw(agg_raw_files)
    if agg_game is not None:
        import torch
        from marketsim.egta.solvers.equilibria import replicator_dynamics

        eq_list = []
        try:
            # Fast analytic path for 2×2 games
            if all(len(lst) == 2 for lst in agg_game.strategy_names_per_role):
                eq_list = [(m, r) for m, r in agg_game.find_nash_equilibrium_2x2() if r <= 1e-3]
        except Exception:
            # fall through to replicator dynamics below
            eq_list = []

        # If analytic path didn't yield anything, always run a quick replicator-dynamics solve
        if not eq_list:
            init_mixture = torch.ones(agg_game.num_strategies) / agg_game.num_strategies
            try:
                mix_fast = replicator_dynamics(agg_game, init_mixture, iters=1000, converge_threshold=STOP_REG)
                eq_list = [(mix_fast, float(agg_game.regret(mix_fast)))]
            except Exception as _e:
                print(f"Warning: replicator dynamics failed for HP={hp}: {_e}")
                eq_list = []

        # Final fallback: use avg mixture from seeds if still empty
        if not eq_list and seed_games_and_mixes:
            eq_list = [(torch.tensor(avg_vec, dtype=torch.float32), 0.0)]

        # Evaluate payoffs for each equilibrium mixture present
        for mix_full, reg_val in eq_list:
            # Align mixture to agg_game strategies
            present_probs = []
            for role, strats in zip(agg_game.role_names, agg_game.strategy_names_per_role):
                for strat in strats:
                    idx = None
                    for j,(r,s) in enumerate(CANONICAL_ORDER):
                        if r==role and s==strat:
                            idx=j; break
                    p = mix_full[idx] if idx is not None and idx < len(mix_full) else 0.0
                    present_probs.append(float(p))
            mix_t = torch.tensor(present_probs, dtype=torch.float32)
            # renormalise per role
            start=0
            for strats in agg_game.strategy_names_per_role:
                seg = mix_t[start:start+len(strats)]
                s_sum = seg.sum()
                if s_sum>1e-12:
                    mix_t[start:start+len(strats)] = seg/s_sum
                else:
                    mix_t[start:start+len(strats)] = 1.0/len(strats)
                start+=len(strats)

            dev_agg = agg_game.deviation_payoffs(mix_t)
            idxg=0
            # compute role-average
            role_val_map = {}
            for role_name,strats in zip(agg_game.role_names, agg_game.strategy_names_per_role):
                for strat in strats:
                    agg_records.append({
                        "Holding Period": hp,
                        "Role": role_name,
                        "Strategy": strat,
                        "Mean": float(dev_agg[idxg]),
                        "Std": 0.0,
                    })
                    idxg+=1
            # second pass to compute role averages now that we know per-strategy devs
            idxg2 = 0
            for role_name,strats in zip(agg_game.role_names, agg_game.strategy_names_per_role):
                seg_size = len(strats)
                mix_seg = mix_t[idxg2:idxg2+seg_size]
                dev_seg = dev_agg[idxg2:idxg2+seg_size]
                role_val = float((mix_seg*dev_seg).sum().item())
                role_val_map[role_name] = role_val
                idxg2 += seg_size
            for role_name,val in role_val_map.items():
                agg_records.append({
                    "Holding Period": hp,
                    "Role": role_name,
                    "Strategy": "_ROLE_AVG_",
                    "Mean": val,
                    "Std": 0.0,
                })

if not strat_records:  
    print("No equilibrium-based payoff data collected.")
    exit()

# ---------------- Bootstrap pass -------------------------------------------
boot_records: list[dict] = []
# NEW: storage for DPR-scaled bootstrap results (n_r = 4 per role)
boot_records_dpr: list[dict] = []


# ---------------- Parallel bootstrap across replicates ------------------

if DO_BOOT and __name__ == "__main__":
    # Prefer 'fork' start method on POSIX; avoids re-importing this module in workers.
    try:
        set_start_method('fork', force=True)
    except RuntimeError:
        pass  # already set or not available (e.g., Windows)

    hp_list = sorted(hp_set)
    # ---------------------------------------------------------------
    # Pre-compute the single pooled equilibrium for each HP using
    # multiple random RD starts and the median-SW selection rule.
    # The helper `find_pooled_equilibrium` already performs that.
    # ---------------------------------------------------------------

    # NEW: Build a global mapping HP -> pooled equilibrium mixture (as list)
    pooled_eq_map = {}
    if not FIGURE_9_ONLY:
        for _hp in hp_list:
            try:
                pooled_eq_map[_hp] = find_pooled_equilibrium(_hp).tolist()
            except Exception as _e:
                print(f"Warning: could not compute pooled equilibrium for HP={_hp}: {_e}")
                # Fallback: uniform mixture over canonical strategies
                pooled_eq_map[_hp] = [0.5, 0.5, 0.5, 0.5]

    # tasks for both bootstrap passes
    tasks = [(hp, i) for hp in hp_list for i in range(N_BOOT)]

    @lru_cache(maxsize=None)
    def _seed_profiles_cached(hp_val: int):
        return collect_profiles_by_seed(hp_val)

    def _one_boot(task):
        hp_val, _idx = task
        seed_map_local = _seed_profiles_cached(hp_val)

        # quick local bootstrap (one replicate)
        rng_local = np.random.default_rng()
        seed_keys_local = list(seed_map_local.keys())
        chosen = rng_local.integers(0, len(seed_keys_local), size=len(seed_keys_local))
        sample_profiles = []
        for cid in chosen:
            sample_profiles.extend(seed_map_local[seed_keys_local[cid]])

        try:
            game = Game.from_payoff_data(sample_profiles, normalize_payoffs=False)
        except Exception:
            return []  # skip malformed replicate

        # ---------------- equilibrium for THIS bootstrap replicate ------------------
        #   Solve for an ε-RSNE (ε = 1e-4) inside the bootstrap sample.
        mix = None
        try:
            # Fast analytic Nash for 2×2 games
            if all(len(lst) == 2 for lst in game.strategy_names_per_role):
                eqs = game.find_nash_equilibrium_2x2()
                if eqs:
                    mix = eqs[0][0]
        except Exception:
            mix = None

        if mix is None:
            from marketsim.egta.solvers.equilibria import replicator_dynamics
            init = torch.ones(game.num_strategies) / game.num_strategies
            try:
                mix = replicator_dynamics(
                    game,
                    init,
                    iters=1000,
                    use_multiple_starts=False,
                    converge_threshold=STOP_REG,
                )
            except Exception:
                mix = None

        if mix is None:
            # Fallback: uniform mixture
            mix = torch.ones(game.num_strategies) / game.num_strategies

        # compute deviation payoffs
        dev = game.deviation_payoffs(mix)
        records_local = []
        idx_loc = 0
        for role, strats in zip(game.role_names, game.strategy_names_per_role):
            seg = slice(idx_loc, idx_loc + len(strats))
            Vr = float((mix[seg] * dev[seg]).sum())
            records_local.append({
                'Holding Period': hp_val,
                'Role': role,
                'Label': 'V',
                'Mean': Vr
            })
            for strat in strats:
                records_local.append({
                    'Holding Period': hp_val,
                    'Role': role,
                    'Label': strat,
                    'Mean': float(dev[idx_loc])
                })
                idx_loc += 1

        # (obsolete share_* rows removed)
 
        # ------------- replicate-level implied mix summary --------------

        # p_i rows obsolete – skip
 
        return records_local

    with Pool(processes=cpu_count()) as pool:
        for rec_list in pool.imap_unordered(_one_boot, tasks, chunksize=50):
            boot_records.extend(rec_list)

    # ----- compute CI summary (Lo / Hi) ---------------------------------
    if boot_records:
        _dfb_all = pd.DataFrame(boot_records)
        summary_boot = []
        for (hp_val, role_val, label_val), grp in _dfb_all.groupby(['Holding Period','Role','Label']):
            lo, hi = np.quantile(grp['Mean'], [ALPHA_CI/2, 1-ALPHA_CI/2])
            summary_boot.append({
                'Holding Period': hp_val,
                'Role': role_val,
                'Label': label_val,
                'Mean': grp['Mean'].mean(),
                'Lo':   float(lo),
                'Hi':   float(hi),
            })
        boot_records = summary_boot

        # ------------------------------------------------------------------
        # DPR BOOTSTRAP PASS – build reduced game (n_red = 4 players/role)
        # ------------------------------------------------------------------

        n_red = 4  # target players per role for DPR

        # Lightweight wrapper implementing DPR scaling and reporting correct num_players_per_role
        class _DPRGame:
            __slots__ = ("base", "scale_per_role", "num_strategies", "role_names",
                         "strategy_names_per_role", "is_role_symmetric", "game",
                         "num_players_per_role")
            def __init__(self, base):
                self.base = base  # original full game (RoleSymmetricGame wrapped by Game)
                self.role_names = base.role_names
                self.strategy_names_per_role = base.strategy_names_per_role
                self.is_role_symmetric = base.is_role_symmetric
                self.game = base.game  # expose device, tensors, etc.
                self.num_strategies = base.num_strategies
                # Compute per-role DPR scaling factor  f_r = (N_r - 1)/(n_red - 1)
                self.scale_per_role = []
                self.num_players_per_role = []
                for N_raw in base.num_players_per_role:
                    N_r = int(N_raw.item() if hasattr(N_raw, "item") else N_raw)
                    self.num_players_per_role.append(n_red)
                    if N_r <= 1 or N_r == n_red:
                        self.scale_per_role.append(1.0)
                    else:
                        self.scale_per_role.append((N_r - 1) / (n_red - 1))

            # Deviation payoffs scaled per DPR
            def deviation_payoffs(self, mix):
                dev = self.base.deviation_payoffs(mix).clone()
                idx = 0
                for role_idx, strats in enumerate(self.strategy_names_per_role):
                    k = len(strats)
                    factor = self.scale_per_role[role_idx]
                    dev[idx : idx + k] *= factor
                    idx += k
                return dev

            def regret(self, mix):
                dev = self.deviation_payoffs(mix)
                return torch.max(dev) - torch.dot(mix, dev)

            def best_responses(self, mix, atol=1e-3):
                dev = self.deviation_payoffs(mix)
                max_pay = torch.max(dev)
                return (dev >= max_pay - atol)

        # ---------------------------------------------------------------
        def _one_boot_dpr(task):
            hp_val, _idx = task
            seed_map_local = _seed_profiles_cached(hp_val)
            rng_local = np.random.default_rng()
            seed_keys_local = list(seed_map_local.keys())
            if not seed_keys_local:
                return []
            chosen = rng_local.integers(0, len(seed_keys_local), size=len(seed_keys_local))
            sample_profiles = []
            for cid in chosen:
                sample_profiles.extend(seed_map_local[seed_keys_local[cid]])
            try:
                full_game = Game.from_payoff_data(sample_profiles, normalize_payoffs=False)
            except Exception:
                return []

            dpr_game = _DPRGame(full_game)

            # Solve equilibrium in DPR game using replicator dynamics
            mix = None
            try:
                if all(len(lst) == 2 for lst in dpr_game.strategy_names_per_role):
                    # analytic path uses full_game.payoffs but equiv under scaling
                    eqs = full_game.find_nash_equilibrium_2x2()
                    if eqs:
                        mix = eqs[0][0]
            except Exception:
                mix = None

            if mix is None:
                from marketsim.egta.solvers.equilibria import replicator_dynamics
                init = torch.ones(dpr_game.num_strategies) / dpr_game.num_strategies
                try:
                    mix = replicator_dynamics(dpr_game, init, iters=1000, use_multiple_starts=False,
                                              converge_threshold=STOP_REG)
                except Exception:
                    mix = init

            dev = dpr_game.deviation_payoffs(mix)
            rec_local = []
            idx_loc = 0
            for role, strats in zip(dpr_game.role_names, dpr_game.strategy_names_per_role):
                seg = slice(idx_loc, idx_loc + len(strats))
                Vr = float((mix[seg] * dev[seg]).sum())
                rec_local.append({'Holding Period': hp_val, 'Role': role, 'Label': 'V', 'Mean': Vr})
                for strat in strats:
                    rec_local.append({'Holding Period': hp_val, 'Role': role, 'Label': strat,
                                       'Mean': float(dev[idx_loc])})
                    idx_loc += 1
            return rec_local

        with Pool(processes=cpu_count()) as _pool_dpr:
            for _rec in _pool_dpr.imap_unordered(_one_boot_dpr, tasks, chunksize=50):
                boot_records_dpr.extend(_rec)

        if boot_records_dpr:
            _df_dpr = pd.DataFrame(boot_records_dpr)
            summary_dpr = []
            for (hp_val, role_val, label_val), grp in _df_dpr.groupby(['Holding Period', 'Role', 'Label']):
                lo, hi = np.quantile(grp['Mean'], [ALPHA_CI/2, 1-ALPHA_CI/2])
                summary_dpr.append({'Holding Period': hp_val, 'Role': role_val, 'Label': label_val,
                                    'Mean': grp['Mean'].mean(), 'Lo': float(lo), 'Hi': float(hi)})
            boot_records_dpr[:] = summary_dpr

        # ------------------------------------------------------------------
        # SECOND BOOTSTRAP PASS – evaluate all replicates against a *fixed*
        # pooled equilibrium (one per HP).  This reproduces the older design
        # but keeps it in a separate figure so we can compare directly with
        # the replicate-specific equilibrium of Figure 9.
        # ------------------------------------------------------------------

        boot_records_fixed: list[dict] = []

        # ---------------------------------------------------------------
        # Helper to sample profiles with full 2×2 support (used by both
        # _one_boot_fixed and _one_boot_full).  Placed *before* the worker
        # functions so it is available when they run.
        # ---------------------------------------------------------------

        MAX_RETRY = 1000  # profile-level retries to obtain full support

        def _draw_profiles_full_support(hp_val:int, rng:np.random.Generator):
            """Bootstrap profiles with replacement until all 4 (role,strategy) pairs appear."""
            seed_map_local = _seed_profiles_cached(hp_val)
            master_profiles = list(itertools.chain.from_iterable(seed_map_local.values()))
            if not master_profiles:
                return None
            n_total = len(master_profiles)
            for _ in range(MAX_RETRY):
                profs = [master_profiles[idx] for idx in rng.integers(0, n_total, size=n_total)]
                sup=set()
                for rec in profs:
                    try:
                        if isinstance(rec,(list,tuple)) and len(rec)>=3:
                            role=rec[1]; strat=rec[2]
                        elif isinstance(rec,dict):
                            role=rec.get('role') or rec.get('Role')
                            strat=rec.get('strategy') or rec.get('Strategy')
                        else:
                            continue
                        if role and strat:
                            sup.add((role,strat))
                    except Exception:
                        continue
                if all(t in sup for t in CANONICAL_ORDER):
                    return profs
            # If still not found, return full master list as fallback (guaranteed support)
            return master_profiles

        def _one_boot_fixed(task):
            hp_val,_idx = task
            seed_map_local = _seed_profiles_cached(hp_val)

            rng_local = np.random.default_rng()
            seed_keys_local = list(seed_map_local.keys())
            chosen = rng_local.integers(0,len(seed_keys_local),size=len(seed_keys_local))
            sample_profiles=[]
            for cid in chosen:
                sample_profiles.extend(seed_map_local[seed_keys_local[cid]])
            #sample_profiles = _draw_profiles_full_support(hp_val, rng_local)
            try:
                game = Game.from_payoff_data(sample_profiles, normalize_payoffs=False)
            except Exception:
                return []

            # ----- pooled equilibrium (pre-computed) --------------------
            pooled_mix_list = pooled_eq_map.get(hp_val)
            if pooled_mix_list is None:
                # As a safeguard, compute on-the-fly (should rarely happen)
                pooled_mix_list = find_pooled_equilibrium(hp_val).tolist()

            # align pooled_mix to strategies present in this resampled game
            present_probs=[]
            for role,strats in zip(game.role_names, game.strategy_names_per_role):
                for strat in strats:
                    idx_can=None
                    for j,(r,s) in enumerate(CANONICAL_ORDER):
                        if r==role and s==strat:
                            idx_can=j; break
                    prob = pooled_mix_list[idx_can] if idx_can is not None and idx_can < len(pooled_mix_list) else 0.0
                    present_probs.append(float(prob))

            mix = torch.tensor(present_probs, dtype=torch.float32)
            # renormalise per role
            start=0
            for strats in game.strategy_names_per_role:
                seg = mix[start:start+len(strats)]
                sm = seg.sum()
                if sm>1e-12:
                    mix[start:start+len(strats)] = seg/sm
                else:
                    mix[start:start+len(strats)] = 1.0/len(strats)
                start += len(strats)

            dev = game.deviation_payoffs(mix)
            rec_local=[]
            idx_loc=0
            for role,strats in zip(game.role_names, game.strategy_names_per_role):
                seg = slice(idx_loc, idx_loc+len(strats))
                Vr=float((mix[seg]*dev[seg]).sum())
                rec_local.append({'Holding Period':hp_val,'Role':role,'Label':'V','Mean':Vr})
                for strat in strats:
                    rec_local.append({'Holding Period':hp_val,'Role':role,'Label':strat,'Mean':float(dev[idx_loc])})
                    idx_loc+=1
            return rec_local

        if not FIGURE_9_ONLY:
            with Pool(processes=cpu_count()) as pool:
                for rec_list in pool.imap_unordered(_one_boot_fixed, tasks, chunksize=50):
                    boot_records_fixed.extend(rec_list)

        # ---- summarise CI for fixed-EQ bootstrap -----------------------
        if not FIGURE_9_ONLY and boot_records_fixed:
            _df_fixed = pd.DataFrame(boot_records_fixed)
            summary_fixed=[]
            for (hp_val,role_val,label_val),grp in _df_fixed.groupby(['Holding Period','Role','Label']):
                lo,hi = np.quantile(grp['Mean'],[ALPHA_CI/2,1-ALPHA_CI/2])
                summary_fixed.append({'Holding Period':hp_val,'Role':role_val,'Label':label_val,
                                      'Mean':grp['Mean'].mean(),'Lo':float(lo),'Hi':float(hi)})
            boot_records_fixed = summary_fixed



        bootA, bootB = [], []  # records for Figures 11 and 12
        '''
        def _one_boot_full(task):
            hp_val, _idx = task
            rng_local = np.random.default_rng()
            sample_profiles = _draw_profiles_full_support(hp_val, rng_local)
            #if not sample_profiles:
            #    return [], []

            try:
                game = Game.from_payoff_data(sample_profiles, normalize_payoffs=False)
            except Exception:
                return [], []

            # ---------- replicate-specific equilibrium (Option A) ----------
            mix_A = None
            try:
                if all(len(lst)==2 for lst in game.strategy_names_per_role):
                    eqs = game.find_nash_equilibrium_2x2()
                    if eqs:
                        mix_A = eqs[0][0]
            except Exception:
                mix_A=None
            if mix_A is None:
                from marketsim.egta.solvers.equilibria import replicator_dynamics
                try:
                    mix_A = replicator_dynamics(game, torch.ones(game.num_strategies)/game.num_strategies,
                                                iters=1000, use_multiple_starts=False,
                                                converge_threshold=STOP_REG)
                except Exception:
                    return [], []

            # ---------- fixed pooled equilibrium (Option B) ---------------
            pooled_mix_list = pooled_eq_map[hp_val]
            present_probs=[]
            for role,strats in zip(game.role_names, game.strategy_names_per_role):
                for strat in strats:
                    idx_can=None
                    for j,(r,s) in enumerate(CANONICAL_ORDER):
                        if r==role and s==strat:
                            idx_can=j; break
                    p = pooled_mix_list[idx_can] if idx_can is not None else 0.0
                    present_probs.append(float(p))
            mix_B = torch.tensor(present_probs, dtype=torch.float32)
            # renormalise per role
            st=0
            for strats in game.strategy_names_per_role:
                seg=mix_B[st:st+len(strats)]
                s=seg.sum()
                if s>1e-12:
                    mix_B[st:st+len(strats)] = seg/s
                else:
                    mix_B[st:st+len(strats)] = 1.0/len(strats)
                st+=len(strats)

            # compute deviation payoffs for both mixes in one pass each
            dev_A = game.deviation_payoffs(mix_A)
            dev_B = game.deviation_payoffs(mix_B)

            recA, recB = [], []
            idx=0
            for role,strats in zip(game.role_names, game.strategy_names_per_role):
                seg = slice(idx, idx+len(strats))
                VrA=float((mix_A[seg]*dev_A[seg]).sum())
                VrB=float((mix_B[seg]*dev_B[seg]).sum())
                recA.append({'Holding Period':hp_val,'Role':role,'Label':'V','Mean':VrA})
                recB.append({'Holding Period':hp_val,'Role':role,'Label':'V','Mean':VrB})
                for strat in strats:
                    recA.append({'Holding Period':hp_val,'Role':role,'Label':strat,'Mean':float(dev_A[idx])})
                    recB.append({'Holding Period':hp_val,'Role':role,'Label':strat,'Mean':float(dev_B[idx])})
                    idx+=1
            return recA, recB

        with Pool(processes=cpu_count()) as pool:
            for pair in pool.imap_unordered(_one_boot_full, tasks, chunksize=50):
                recA, recB = pair
                bootA.extend(recA)
                bootB.extend(recB)

        # summarise CIs
        def _summarise_boot(lst):
            df=pd.DataFrame(lst)
            out=[]
            for (hp_v, role_v, label_v), grp in df.groupby(['Holding Period','Role','Label']):
                lo,hi = np.quantile(grp['Mean'], [ALPHA_CI/2, 1-ALPHA_CI/2])
                out.append({'Holding Period':hp_v,'Role':role_v,'Label':label_v,
                            'Mean':grp['Mean'].mean(),'Lo':float(lo),'Hi':float(hi)})
            return out

        bootA = _summarise_boot(bootA) if bootA else []
        bootB = _summarise_boot(bootB) if bootB else []

# ---------- Figure 1: per-strategy payoff vs equilibrium mixture ----------
"""
# ---------- Figure 1: per-strategy payoff vs equilibrium mixture ----------
"""

# HP ≤ 400 for clarity
for ax in axes1:
    ax.set_xlim(0, 320)

plt.tight_layout()
out1 = Path(__file__).with_name("payoffs_vs_eq_by_hp_normalize_normalize_normalize.png")
fig1.savefig(out1, dpi=300)
print("Saved per-strategy equilibrium payoff plot to", out1.relative_to(Path.cwd()))

# ---------------- Figure 2: per-strategy payoff vs average mixture -------------
if avg_strat_records:
    df_avg_strat = pd.DataFrame(avg_strat_records)
    fig2, axes2 = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    for ax, role in zip(axes2, df_avg_strat["Role"].unique()):
        sub = df_avg_strat[df_avg_strat["Role"] == role]
        for strat, g in sub.groupby("Strategy"):
            g_sorted = g.sort_values("Holding Period")
            ax.plot(g_sorted["Holding Period"], g_sorted["Mean"], marker="o", label=strat)
            ax.fill_between(
                g_sorted["Holding Period"],
                g_sorted["Mean"] - g_sorted["Std"],
                g_sorted["Mean"] + g_sorted["Std"],
                alpha=0.2,
            )
        ax.set_title(f"{role} – Payoff vs Average Mixture")
        ax.set_xlabel("Holding Period")
        ax.set_ylabel("Expected Payoff versus ¯x (mean ±1σ)")
        ax.legend()
    plt.tight_layout()
    out2 = Path(__file__).with_name("payoffs_vs_avg_mix_by_hp_normalize.png")
    fig2.savefig(out2, dpi=300)
    print("Saved per-strategy average-mixture payoff plot to", out2.relative_to(Path.cwd()))

# -------------- Figure 3: aggregated equilibria (unchanged) --------------
if agg_records:
    agg_df = pd.DataFrame(agg_records)
    fig3, axes3 = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    for ax, role in zip(axes3, agg_df["Role"].unique()):
        sub = agg_df[agg_df["Role"] == role]
        for strat, g in sub.groupby("Strategy"):
            g_sorted = g.sort_values("Holding Period")
            ax.plot(g_sorted["Holding Period"], g_sorted["Mean"], marker="o", label=strat)
            # Std is zero here (single game evaluation per HP). Keep for API consistency.
        ax.set_title(f"{role} – Payoff vs Aggregated-Game EQ")
        ax.set_xlabel("Holding Period")
        ax.set_ylabel("Expected Payoff versus x̂ (aggregated EQ)")
        ax.legend()
    plt.tight_layout()
    out3 = Path(__file__).with_name("payoffs_vs_agg_eq_by_hp_normalize.png")
    fig3.savefig(out3, dpi=300)
    print("Saved aggregated-game equilibrium payoff plot to", out3.relative_to(Path.cwd()))

# ---------- Figure 4: role-average equilibrium payoff ----------
df_role = pd.DataFrame(role_records)

if not df_role.empty:
    roles = df_role["Role"].unique()
    fig4, axes4 = plt.subplots(1, len(roles), figsize=(7 * len(roles), 6), sharey=True)
    if len(roles) == 1:
        axes4 = [axes4]
    for ax, role in zip(axes4, roles):
        sub = df_role[df_role["Role"] == role].sort_values("Holding Period")
        ax.plot(sub["Holding Period"], sub["Mean"], marker="o")
        ax.fill_between(
            sub["Holding Period"],
            sub["Mean"] - sub["Std"],
            sub["Mean"] + sub["Std"],
            alpha=0.2,
        )
        ax.set_title(f"{role} – Equilibrium Payoff (role-avg)")
        ax.set_xlabel("Holding Period")
        ax.set_ylabel("Expected Payoff V_r(x) (mean ±1σ)")
        ax.set_xlim(0, 320)
    plt.tight_layout()
    out4 = Path(__file__).with_name("role_eq_payoff_by_hp_normalize.png")
    fig4.savefig(out4, dpi=300)
    print("Saved role-average equilibrium payoff plot to", out4.relative_to(Path.cwd()))

# ---------------- Figure 5: role-average payoff vs average mixture ----------
if avg_role_records:
    df_role_avg = pd.DataFrame(avg_role_records)
    roles_r = df_role_avg["Role"].unique()
    fig5, axes5 = plt.subplots(1, len(roles_r), figsize=(7 * len(roles_r), 6), sharey=True)
    if len(roles_r) == 1:
        axes5 = [axes5]
    for ax, role in zip(axes5, roles_r):
        sub = df_role_avg[df_role_avg["Role"] == role].sort_values("Holding Period")
        ax.plot(sub["Holding Period"], sub["Mean"], marker="o")
        ax.fill_between(
            sub["Holding Period"],
            sub["Mean"] - sub["Std"],
            sub["Mean"] + sub["Std"],
            alpha=0.2,
        )
        ax.set_title(f"{role} – Payoff vs Average Mixture (role-avg)")
        ax.set_xlabel("Holding Period")
        ax.set_ylabel("Expected Payoff V_r(¯x) (mean ±1σ)")
        ax.set_xlim(0, 320)
    plt.tight_layout()
    out5 = Path(__file__).with_name("role_avg_mix_payoff_by_hp_normalize.png")
    fig5.savefig(out5, dpi=300)
    print("Saved role-average average-mixture payoff plot to", out5.relative_to(Path.cwd()))

# ---------------- Figure 6: ALL curves on same axes (equilibrium mixture) --------------
if not df_role.empty:
    fig6, axes6 = plt.subplots(1, 2, figsize=(15,6), sharey=True)
    roles_c = df_strat["Role"].unique()
    for ax, role in zip(axes6, roles_c):
        sub_s = df_strat[df_strat["Role"]==role]
        for strat, g in sub_s.groupby("Strategy"):
            g_sorted = g.sort_values("Holding Period")
            ax.plot(g_sorted["Holding Period"], g_sorted["Mean"], marker="o", label=strat)
            ax.fill_between(g_sorted["Holding Period"],
                             g_sorted["Mean"]-g_sorted["Std"],
                             g_sorted["Mean"]+g_sorted["Std"], alpha=0.15)
        # role-average overlay
        sub_r = df_role[df_role["Role"]==role].sort_values("Holding Period")
        ax.plot(sub_r["Holding Period"], sub_r["Mean"], color="black", ls="-", lw=2, marker="s", label="V_r(x)")
        ax.fill_between(sub_r["Holding Period"],
                        sub_r["Mean"]-sub_r["Std"],
                        sub_r["Mean"]+sub_r["Std"], color="grey", alpha=0.2)
        ax.set_title(f"{role} – Deviation vs V_r(x) (eq mix)")
        ax.set_xlabel("Holding Period"); ax.set_ylabel("Payoff")
        ax.set_xlim(0,320)
        ax.legend()
        # ---- Add vertical dashed lines at support-switch points ----
        # Determine switch points based on EPS criterion
        EPS = 2000  # payoff tolerance for support
        # Strategy names
        strat_melo = f"{role}_0_100_shade0_0" if role == "MOBI" else "ZI_0_100_shade250_500"
        strat_cda  = f"{role}_100_0_shade250_500" if role == "MOBI" else "ZI_100_0_shade250_500"

        df_role_sub = df_role[df_role["Role"] == role]
        hp_vals_sorted = sorted(df_role_sub["Holding Period"].unique())
        current_state = None  # 'M', 'C', 'mixed'
        switch_hps = []
        for hp_val in hp_vals_sorted:
            Vr = df_role_sub[df_role_sub["Holding Period"] == hp_val]["Mean"].values[0]
            um = df_strat[(df_strat["Role"]==role)&(df_strat["Strategy"]==strat_melo)&(df_strat["Holding Period"]==hp_val)]["Mean"]
            uc = df_strat[(df_strat["Role"]==role)&(df_strat["Strategy"]==strat_cda)&(df_strat["Holding Period"]==hp_val)]["Mean"]
            sup = []
            if len(um) and abs(um.values[0]-Vr)<=EPS: sup.append('M')
            if len(uc) and abs(uc.values[0]-Vr)<=EPS: sup.append('C')
            state = 'mixed' if len(sup)==2 else ('M' if 'M' in sup else ('C' if 'C' in sup else 'none'))
            if current_state is None:
                current_state = state
            elif state != current_state:
                switch_hps.append(hp_val)
                current_state = state
        # plot vertical lines
        for hp_sw in switch_hps:
            ax.axvline(hp_sw, color='grey', ls='--', alpha=0.6)
        # ---- end vertical lines ----
    plt.tight_layout()
    out6 = Path(__file__).with_name("compare_eq_mix_all_lines_by_hp_normalize.png")
    fig6.savefig(out6, dpi=300)
    print("Saved comparison plot (eq mix) to", out6.relative_to(Path.cwd()))

# ---------------- Figure 7: ALL curves on same axes (average mixture) --------------
if avg_strat_records and avg_role_records:
    df_avg_strat_full = pd.DataFrame(avg_strat_records)
    df_role_avg_full = pd.DataFrame(avg_role_records)
    fig7, axes7 = plt.subplots(1, 2, figsize=(15,6), sharey=True)
    for ax, role in zip(axes7, df_avg_strat_full["Role"].unique()):
        sub_s = df_avg_strat_full[df_avg_strat_full["Role"]==role]
        for strat, g in sub_s.groupby("Strategy"):
            g_sorted = g.sort_values("Holding Period")
            ax.plot(g_sorted["Holding Period"], g_sorted["Mean"], marker="o", label=strat)
            ax.fill_between(g_sorted["Holding Period"],
                             g_sorted["Mean"]-g_sorted["Std"],
                             g_sorted["Mean"]+g_sorted["Std"], alpha=0.15)
        sub_r = df_role_avg_full[df_role_avg_full["Role"]==role].sort_values("Holding Period")
        ax.plot(sub_r["Holding Period"], sub_r["Mean"], color="black", ls="-", lw=2, marker="s", label="V_r(¯x)")
        ax.fill_between(sub_r["Holding Period"],
                        sub_r["Mean"]-sub_r["Std"],
                        sub_r["Mean"]+sub_r["Std"], color="grey", alpha=0.2)
        ax.set_title(f"{role} – Deviation vs V_r(¯x) (avg mix)")
        ax.set_xlabel("Holding Period"); ax.set_ylabel("Payoff")
        ax.set_xlim(0,320)
        ax.legend()
    plt.tight_layout()
    out7 = Path(__file__).with_name("compare_avg_mix_all_lines_by_hp_normalize.png")
    fig7.savefig(out7, dpi=300)
    print("Saved comparison plot (avg mix) to", out7.relative_to(Path.cwd()))

# ---------------- Figure 8: comparison plot for aggregated game (pooled) --------------
if agg_records:
    df_agg = pd.DataFrame([r for r in agg_records if r["Strategy"]!="_ROLE_AVG_"])
    df_agg_role = pd.DataFrame([r for r in agg_records if r["Strategy"]=="_ROLE_AVG_"])
    fig8, axes8 = plt.subplots(1, 2, figsize=(15,6), sharey=True)
    for ax, role in zip(axes8, df_agg["Role"].unique()):
        sub_s = df_agg[df_agg["Role"]==role]
        for strat, g in sub_s.groupby("Strategy"):
            g_sorted = g.sort_values("Holding Period")
            ax.plot(g_sorted["Holding Period"], g_sorted["Mean"], marker="o", label=strat)
        # add role avg
        sub_r = df_agg_role[df_agg_role["Role"]==role].sort_values("Holding Period")
        if not sub_r.empty:
            ax.plot(sub_r["Holding Period"], sub_r["Mean"], color='black', lw=2, marker='s', label='V_r(x̂)')
        ax.set_title(f"{role} – Deviation vs V_r(x̂) (pooled EQ)")
        ax.set_xlabel("Holding Period"); ax.set_ylabel("Payoff")
        ax.set_xlim(0,320)
        ax.legend()
    plt.tight_layout()
    out8 = Path(__file__).with_name("compare_pooled_eq_all_lines_by_hp_normalize.png")
    fig8.savefig(out8, dpi=300)
    print("Saved comparison plot (pooled eq) to", out8.relative_to(Path.cwd()))
'''
# ---------------- Figure 9: bootstrap comparison pooled --------------------
if DO_BOOT and boot_records:
    dfb_full = pd.DataFrame(boot_records)
    # Print average mixture shares table
    share_df = dfb_full[dfb_full['Label'].str.startswith('share')]
    if not share_df.empty:
        table = share_df.pivot_table(index='Holding Period', columns=['Role','Label'], values='Mean', aggfunc='mean').round(2)
        print("\nBootstrap equilibrium mixture (average share per HP):\n")
        print(table.to_string())

    # Drop share rows for plotting payoffs
    dfb = dfb_full[~dfb_full['Label'].str.startswith('share')]

    # ---- replicate-level implied mix summary -------------------
    # (obsolete p_i summary removed)

    # ---------- implied mix that reproduces the black V_r(x) point ----------
    strat_map = {
        'MOBI': {'MELO': 'MOBI_0_100_shade0_0',
                 'CDA' : 'MOBI_100_0_shade250_500'},
        'ZI'  : {'MELO': 'ZI_0_100_shade250_500',
                 'CDA' : 'ZI_100_0_shade250_500'},
    }

    mix_rows = []
    for role, mapping in strat_map.items():
        role_df = dfb[dfb['Role'] == role]
        for hp_val, grp in role_df.groupby('Holding Period'):
            # skip if any of the three numbers is missing
            if not ((grp['Label'] == mapping['MELO']).any() and
                    (grp['Label'] == mapping['CDA' ]).any() and
                    (grp['Label'] == 'V'        ).any()):
                continue

            u_melo = grp.loc[grp['Label'] == mapping['MELO'], 'Mean'].iloc[0]
            u_cda  = grp.loc[grp['Label'] == mapping['CDA' ], 'Mean'].iloc[0]
            v_r    = grp.loc[grp['Label'] == 'V'           , 'Mean'].iloc[0]

            denom  = u_cda - u_melo
            p      = np.nan if np.isclose(denom, 0.0) else (v_r - u_melo) / denom
            mix_rows.append({'Holding Period': hp_val,
                             'Role'         : role,
                             'share_CDA'    : round(float(p)      , 2) if np.isfinite(p) else np.nan,
                             'share_MELO'   : round(float(1 - p)  , 2) if np.isfinite(p) else np.nan})

    if mix_rows:
        mix_df = (pd.DataFrame(mix_rows)
                    .pivot(index='Holding Period', columns='Role')
                    .sort_index())
        print("\nImplied equilibrium mix (matches mean V_r payoff):\n")
        print(mix_df.to_string())

    fig9, axes9 = plt.subplots(1, 2, figsize=(15,6), sharey=False)
    for ax, role in zip(axes9, dfb['Role'].unique()):
        sub = dfb[dfb['Role']==role]
        # Ensure HP axis sorted and integer
        hp_sorted = sorted(sub['Holding Period'].unique())
        ax.set_xticks(hp_sorted)
        ax.set_xticklabels([str(int(h)) for h in hp_sorted], rotation=45, ha='right')

        # Manual label mapping
        label_map = {
            CANONICAL_ORDER[1][1] if role=='MOBI' else CANONICAL_ORDER[3][1]: 'Dev: CDA',
            CANONICAL_ORDER[0][1] if role=='MOBI' else CANONICAL_ORDER[2][1]: 'Dev: MELO',
            'V': 'Equilibrium Value',
        }

        for label, g in sub.groupby('Label'):
            g_sorted = g.sort_values('Holding Period')
            x = g_sorted['Holding Period'].to_numpy()
            y = g_sorted['Mean'].to_numpy()
            lo = g_sorted['Lo'].to_numpy()
            hi = g_sorted['Hi'].to_numpy()
            # optional smoothing of means
            if USE_SMOOTHING and label in {'V', CANONICAL_ORDER[0][1], CANONICAL_ORDER[1][1], CANONICAL_ORDER[2][1], CANONICAL_ORDER[3][1]}:
                y_plot = gaussian_filter1d(y, sigma=.25)
            else:
                y_plot = y

            # optional smoothing of CI bounds
            if SMOOTH_CI and not np.isnan(lo).all():
                lo_plot = gaussian_filter1d(lo, sigma=.25)
                hi_plot = gaussian_filter1d(hi, sigma=.25)
            else:
                lo_plot, hi_plot = lo, hi

            if label == 'V':
                ax.plot(x, y_plot, color='black', lw=2, marker='s', label=label_map.get(label,'Equilibrium Value'))
                ax.fill_between(x, lo_plot, hi_plot, color='grey', alpha=0.25)
            else:
                ax.plot(x, y_plot, marker='o', label=label_map.get(label,label))
                ax.fill_between(x, lo_plot, hi_plot, alpha=0.15)

        # Removed per-HP mixture share annotations (no global pooled mix under this bootstrap definition)
 
        ax.set_title(f"{role} – Deviations vs V_r(x) (bootstrap {N_BOOT})")
        ax.set_xlabel("Holding Period")
        ax.set_ylabel("Payoff")
        ax.set_xlim(0, 320)

        # Place legend outside right of plot
        ax.legend(bbox_to_anchor=(1.05,1), loc='upper left')
 
    plt.tight_layout()
    out9 = Path(__file__).with_name("compare_pooled_bootstrap_all_lines_by_hp_no_smooth.png")
    fig9.savefig(out9, dpi=300)
    print("Saved bootstrap comparison plot to", out9.relative_to(Path.cwd())) 

# ---------------- Figure 9DPR: bootstrap comparison with DPR (n=4 per role) --------------
if DO_BOOT and boot_records_dpr:
    dfb_dpr_full = pd.DataFrame(boot_records_dpr)
    dfb_dpr = dfb_dpr_full[~dfb_dpr_full['Label'].str.startswith('share')]
    fig_dpr, axes_dpr = plt.subplots(1, 2, figsize=(15,6), sharey=False)
    for ax, role in zip(axes_dpr, dfb_dpr['Role'].unique()):
        sub = dfb_dpr[dfb_dpr['Role']==role]
        hp_sorted = sorted(sub['Holding Period'].unique())
        ax.set_xticks(hp_sorted)
        ax.set_xticklabels([str(int(h)) for h in hp_sorted], rotation=45, ha='right')
        label_map = {
            CANONICAL_ORDER[1][1] if role=='MOBI' else CANONICAL_ORDER[3][1]: 'Dev: CDA (DPR)',
            CANONICAL_ORDER[0][1] if role=='MOBI' else CANONICAL_ORDER[2][1]: 'Dev: MELO (DPR)',
            'V': 'Equilibrium Value (DPR)',
        }
        for label, g in sub.groupby('Label'):
            g_sorted = g.sort_values('Holding Period')
            x = g_sorted['Holding Period'].to_numpy()
            y = g_sorted['Mean'].to_numpy()
            lo = g_sorted.get('Lo', pd.Series(np.nan, index=g_sorted.index)).to_numpy()
            hi = g_sorted.get('Hi', pd.Series(np.nan, index=g_sorted.index)).to_numpy()
            y_plot = gaussian_filter1d(y, sigma=.25) if USE_SMOOTHING and label in {'V', CANONICAL_ORDER[0][1], CANONICAL_ORDER[1][1], CANONICAL_ORDER[2][1], CANONICAL_ORDER[3][1]} else y
            if SMOOTH_CI and not np.isnan(lo).all():
                lo_plot = gaussian_filter1d(lo, sigma=.25)
                hi_plot = gaussian_filter1d(hi, sigma=.25)
            else:
                lo_plot, hi_plot = lo, hi
            if label == 'V':
                ax.plot(x, y_plot, color='black', lw=2, marker='s', label=label_map.get(label, label))
                ax.fill_between(x, lo_plot, hi_plot, color='grey', alpha=0.25)
            else:
                ax.plot(x, y_plot, marker='o', label=label_map.get(label, label))
                ax.fill_between(x, lo_plot, hi_plot, alpha=0.15)
        ax.set_title(f"{role} – Deviations vs V_r(x) (DPR n=4, bootstrap {N_BOOT})")
        ax.set_xlabel("Holding Period")
        ax.set_ylabel("Payoff")
        ax.set_xlim(0, 320)
        ax.legend(bbox_to_anchor=(1.05,1), loc='upper left')
    plt.tight_layout()
    out_dpr = Path(__file__).with_name("compare_pooled_bootstrap_dpr_n4_by_hp.png")
    fig_dpr.savefig(out_dpr, dpi=300)
    print("Saved DPR bootstrap comparison plot to", out_dpr.relative_to(Path.cwd()))
