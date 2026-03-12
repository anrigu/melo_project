import argparse
import json
import pathlib
import re
from collections import defaultdict
from typing import List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kw: x

# ── CONFIG ─────────────────────────────────────────────────────────
BOOT = 2000
CONF_L, CONF_H = 2.5, 97.5  # 95% CI

# ------------------------------------------------------------------
# Utility helpers
# ------------------------------------------------------------------

def bootstrap_ci(samples: List[float], do_boot: bool = True):
    """Compute mean and 95% CI via bootstrap or direct mean if disabled."""
    if not samples:
        return np.nan, np.nan, np.nan
    arr = np.asarray(samples)
    if not do_boot:
        m = float(np.mean(arr))
        return m, np.nan, np.nan
    means = [np.mean(np.random.choice(arr, size=len(arr), replace=True)) for _ in range(BOOT)]
    lo, hi = np.percentile(means, [CONF_L, CONF_H])
    return np.mean(arr), lo, hi


def aggregate_payoffs(
    profiles: List,
    role: str,
    strat_pairs: Dict[str, List[str]],
    role_size: int,
    *,
    do_boot: bool
) -> pd.DataFrame:
    """
    Build a DataFrame with columns k, mean_M, lo_M, hi_M, mean_C, lo_C, hi_C
    for k = number of agents playing strategy M-ELO.
    """
    if role not in strat_pairs:
        return pd.DataFrame()
    strat_M, strat_C = strat_pairs[role]
    buckets = defaultdict(lambda: {'M': [], 'C': []})

    for prof in profiles:
        k = sum(1 for _, r, s, _ in prof if r == role and s == strat_M)
        for _, r, s, payoff in prof:
            if r != role:
                continue
            if s == strat_M:
                buckets[k]['M'].append(payoff)
            elif s == strat_C:
                buckets[k]['C'].append(payoff)

    rows = []
    for k in range(role_size + 1):
        m_M, lo_M, hi_M = bootstrap_ci(buckets[k]['M'], do_boot)
        m_C, lo_C, hi_C = bootstrap_ci(buckets[k]['C'], do_boot)
        rows.append([k, m_M, lo_M, hi_M, m_C, lo_C, hi_C])

    df = pd.DataFrame(
        rows,
        columns=['k', 'mean_M', 'lo_M', 'hi_M', 'mean_C', 'lo_C', 'hi_C']
    )
    return df.set_index('k').interpolate(method='linear', limit_area='inside').reset_index()


def find_welfare_opt_k(df: pd.DataFrame, role_size: int) -> int:
    """
    Returns the k (0..role_size) that maximizes total welfare:
      welfare(k) = k*mean_M(k) + (role_size-k)*mean_C(k)
    Treats any NaN mean as -inf welfare.
    """
    ks = df['k'].to_numpy()
    mean_M = dict(zip(ks, df['mean_M']))
    mean_C = dict(zip(ks, df['mean_C']))
    welfare = {}
    for k in ks:
        mM = mean_M.get(k, np.nan)
        mC = mean_C.get(k, np.nan)
        if np.isnan(mM) or np.isnan(mC):
            welfare[k] = -np.inf
        else:
            welfare[k] = k * mM + (role_size - k) * mC
    # select k maximizing welfare, tie-breaking on smallest k
    best_k = max(welfare.keys(), key=lambda k: (welfare[k], -k))
    return int(best_k)


def plot_role(
    role: str,
    df: pd.DataFrame,
    title_extra: str = '',
    *,
    arrows_only: bool = False
) -> plt.Figure:
    """
    Plot bar+CI for each strategy and overlay best-response arrows.
    Use arrows_only=True to omit bars.
    """
    ks = df['k'].to_numpy()
    mean_M = df['mean_M'].to_numpy()
    mean_C = df['mean_C'].to_numpy()
    lo_M, hi_M = df['lo_M'].to_numpy(), df['hi_M'].to_numpy()
    lo_C, hi_C = df['lo_C'].to_numpy(), df['hi_C'].to_numpy()

    mask_M = ~np.isnan(mean_M)
    mask_C = ~np.isnan(mean_C)

    fig, ax = plt.subplots(figsize=(10, 5))
    w = 0.35

    if not arrows_only:
        ax.bar(
            ks[mask_C] - w/2,
            mean_C[mask_C],
            width=w,
            color='#2196F3',
            yerr=[(mean_C - lo_C)[mask_C], (hi_C - mean_C)[mask_C]],
            label='Play CDA'
        )
        ax.bar(
            ks[mask_M] + w/2,
            mean_M[mask_M],
            width=w,
            color='#8BC34A',
            yerr=[(mean_M - lo_M)[mask_M], (hi_M - mean_M)[mask_M]],
            label='Play M-ELO'
        )

    ax.set_xlabel(f'# {role}s choosing M-ELO')
    ax.set_ylabel('Average payoff' if not arrows_only else '')
    ax.set_title(
        f"{role} – deviation arrows {title_extra}" if arrows_only
        else f"{role} – average payoffs {title_extra}"
    )
    ax.grid(axis='y', linestyle='--', alpha=0.4)

    x_cda = {k: k - w/2 for k in ks}
    x_melo = {k: k + w/2 for k in ks}

    mean_M_d = dict(zip(ks, mean_M))
    mean_C_d = dict(zip(ks, mean_C))
    hi_M_d = {k: hi_M[i] if not np.isnan(hi_M[i]) else mean_M[i] for i, k in enumerate(ks)}
    hi_C_d = {k: hi_C[i] if not np.isnan(hi_C[i]) else mean_C[i] for i, k in enumerate(ks)}

    for k in ks:
        gain_plus = mean_M_d.get(k+1, -np.inf) - mean_C_d.get(k, -np.inf)
        gain_minus = mean_C_d.get(k-1, -np.inf) - mean_M_d.get(k, -np.inf)
        if gain_plus > gain_minus and gain_plus > 0:
            tail = (x_cda[k], hi_C_d[k])
            head = (x_melo[k+1], hi_M_d[k+1])
        elif gain_minus > 0:
            tail = (x_melo[k], hi_M_d[k])
            head = (x_cda[k-1], hi_C_d[k-1])
        else:
            continue
        ax.annotate(
            '',
            xy=head,
            xytext=tail,
            arrowprops=dict(arrowstyle='-|>', lw=1.5, color='black'),
            zorder=3
        )

    if arrows_only:
        ys = list(hi_C_d.values()) + list(hi_M_d.values())
        ymin, ymax = min(ys), max(ys)
        margin = (ymax - ymin) * 0.1 if ymax > ymin else ymax * 0.1
        ax.set_ylim(ymin - margin, ymax + margin)

    plt.xticks(ks, rotation=45)
    plt.tight_layout()
    if not arrows_only:
        ax.legend()
    return fig


def plot_optimum_curve(opts: Dict[int, int], role: str, out_dir: pathlib.Path):
    """
    Plot the social-welfare optimum k vs. holding period, with shaded regimes:
      Regime 1: t_holding ≤ 220 (all-M-ELO optimum & eq)
      Regime 2: 220 < t_holding ≤ 250 (all-M-ELO eq only)
      Regime 3: 250 < t_holding ≤ 280 (mixed eq)
      Regime 4: t_holding > 280 (no pure M-ELO eq)
    """
    hps = sorted(opts)
    ks = [opts[hp] for hp in hps]
    fig, ax = plt.subplots(figsize=(10, 5))
    # shade regimes with legend labels
    ax.axvspan(min(hps), 220, color='green', alpha=0.1, label='Regime 1: all-M-ELO optimum')
    ax.axvspan(220, 250, color='yellow', alpha=0.1, label='Regime 2: all-M-ELO eq only')
    ax.axvspan(250, 280, color='orange', alpha=0.1, label='Regime 3: mixed eq')
    ax.axvspan(280, max(hps), color='red', alpha=0.1, label='Regime 4: no M-ELO eq')
    # plot optimum curve
    ax.plot(hps, ks, marker='o', color='black', label='Social-welfare optimum k')
    ax.set_xlabel('Holding Period')
    ax.set_ylabel('Number of M-ELO agents (k)')
    ax.set_title(f'Holding Period Dependence of {role} Trader Social Optimum')
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.legend(loc='upper right')
    fig.savefig(out_dir / f'optimum_{role.lower()}.png')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description='Generate bar+arrow payoff plots and welfare-optimum curves by holding period.'
    )
    parser.add_argument(
        'root_dirs', nargs='+', type=pathlib.Path,
        help='Root dirs containing holding_period_<k>/raw_payoff_data.json'
    )
    parser.add_argument(
        '--out', type=pathlib.Path, default=pathlib.Path('hp_plots'),
        help='Output directory'
    )
    parser.add_argument(
        '--no-bootstrap', action='store_true',
        help='Skip bootstrap resampling (no CI)'
    )
    parser.add_argument(
        '--arrows-only', action='store_true',
        help='Draw only deviation arrows'
    )
    parser.add_argument(
        '--plot-opt', action='store_true', dest='plot_opt',
        help='Also compute & plot social-welfare optimum curve'
    )
    parser.add_argument(
        '--plot-eq', action='store_true', dest='plot_opt',
        help='Alias for --plot-opt'
    )
    args = parser.parse_args()

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    profiles_by_hp = defaultdict(list)
    role_max = defaultdict(int)
    for rd in args.root_dirs:
        if not rd.exists():
            print(f'Warning: {rd} does not exist – skipping')
            continue
        for pf in rd.rglob('raw_payoff_data.json'):
            m = re.search(r'holding_period_(\d+)', str(pf))
            if not m:
                continue
            hp = int(m.group(1))
            try:
                profs = json.loads(pf.read_text())
            except json.JSONDecodeError:
                continue
            profiles_by_hp[hp].extend(profs)
            for prof in profs:
                cnt = defaultdict(int)
                for _, r, _, _ in prof:
                    cnt[r] += 1
                for r, c in cnt.items():
                    role_max[r] = max(role_max[r], c)

    strat_pairs: Dict[str, List[str]] = {}
    for profs in profiles_by_hp.values():
        for prof in profs:
            by_role = defaultdict(set)
            for _, r, s, _ in prof:
                by_role[r].add(s)
            for r, ss in by_role.items():
                if len(ss) == 2 and r not in strat_pairs:
                    strat_pairs[r] = sorted(ss)
    strat_pairs.setdefault('MOBI', ['MOBI_0_100_shade0_0', 'MOBI_100_0_shade250_500'])
    strat_pairs.setdefault('ZI', ['ZI_0_100_shade250_500', 'ZI_100_0_shade250_500'])

    opts_by_role: Dict[str, Dict[int, int]] = defaultdict(dict)

    for hp in sorted(profiles_by_hp):
        profs = profiles_by_hp[hp]
        for role in strat_pairs:
            maxp = role_max.get(role, 0)
            if maxp == 0:
                continue
            df = aggregate_payoffs(
                profs, role, strat_pairs, maxp,
                do_boot=not args.no_bootstrap
            )
            if df.empty:
                continue

            fig = plot_role(
                role, df,
                title_extra=f'(HP={hp})',
                arrows_only=args.arrows_only
            )
            fig.savefig(out_dir / f'hp{hp}_{role.lower()}.png')
            plt.close(fig)
            print(f'Saved HP {hp} {role} plot → {out_dir}')

            if args.plot_opt:
                k_opt = find_welfare_opt_k(df, maxp)
                opts_by_role[role][hp] = k_opt

    if args.plot_opt:
        for role, opts in opts_by_role.items():
            plot_optimum_curve(opts, role, out_dir)
            print(f'Saved optimum curve for {role} → {out_dir}')

if __name__ == '__main__':
    main()
