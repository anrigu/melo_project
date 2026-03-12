import json, pathlib, os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from collections import defaultdict

# ── CONFIG ─────────────────────────────────────────────────────────
BOOT      = 2000
CONF_L, CONF_H = 2.5, 97.5          # 95 % CI
path = "/Users/gabesmithline/Desktop/holding_period_0/comprehensive_rsg_results_20250619_115208/raw_payoff_data.json"      # <- change if needed
assert os.path.isfile(path)
profiles = json.loads(pathlib.Path(path).read_text())


# ── BOOTSTRAP HELPERS ────────────────────────────────────────────
def strat_names(role):
    """Return (MELO_strat, CDA_strat) names for the given role."""
    if role == "ZI":
        # Dataset encodes ZI M-ELO choice with liquidity 0–100 but shade 250–500
        return ("ZI_0_100_shade250_500", "ZI_100_0_shade250_500")
    else:  # MOBI
        return ("MOBI_0_100_shade0_0", "MOBI_100_0_shade250_500")


def bootstrap_ci(samples):
    if not samples:
        return np.nan, np.nan, np.nan
    arr = np.asarray(samples)
    means = [
        np.mean(np.random.choice(arr, size=len(arr), replace=True))
        for _ in range(BOOT)
    ]
    lo, hi = np.percentile(means, [CONF_L, CONF_H])
    return np.mean(arr), lo, hi


# ── AGGREGATE PAYOFFS BY k (#M-ELO players) ─────────────────────
def aggregate_with_interpolation(role):
    melo_strat, cda_strat = strat_names(role)
    buckets = defaultdict(lambda: dict(M=[], C=[]))

    for prof in profiles:
        k = sum(1 for _, r, s, _ in prof if r == role and s == melo_strat)

        # skip profiles missing one of the two actions
        if not (
            any(r == role and s == melo_strat for _, r, s, _ in prof)
            and any(r == role and s == cda_strat  for _, r, s, _ in prof)
        ):
            continue

        for _, r, s, payoff in prof:
            if r != role:
                continue
            (buckets[k]["M"] if s == melo_strat else buckets[k]["C"]).append(
                payoff
            )

    if not buckets:
        return pd.DataFrame()

    rows, max_k = [], max(buckets)
    for k in range(max_k + 1):
        mean_M, lo_M, hi_M = bootstrap_ci(buckets[k]["M"])
        mean_C, lo_C, hi_C = bootstrap_ci(buckets[k]["C"])
        rows.append([k, mean_M, lo_M, hi_M, mean_C, lo_C, hi_C])

    df = pd.DataFrame(
        rows,
        columns=["k", "mean_M", "lo_M", "hi_M", "mean_C", "lo_C", "hi_C"],
    )

    # ensure every k has data (linear interpolation + edge fill)
    return (
        df.set_index("k")
          .interpolate(method="linear", limit_direction="both")
          .reset_index()
    )


# ── PLOTTER WITH BEST-RESPONSE ARROWS ────────────────────────────
def plot_role(role, df):
    ks         = df["k"].to_numpy()
    mean_M     = df["mean_M"].to_numpy()
    mean_C     = df["mean_C"].to_numpy()
    lo_M, hi_M = df["lo_M"].to_numpy(), df["hi_M"].to_numpy()
    lo_C, hi_C = df["lo_C"].to_numpy(), df["hi_C"].to_numpy()

    fig, ax = plt.subplots(figsize=(12, 6))
    w = 0.35                                       # bar width

    # bars:  CDA left,  M-ELO right
    ax.bar(
        ks - w / 2,
        mean_C,
        width=w,
        color="#2196F3",
        yerr=[mean_C - lo_C, hi_C - mean_C],
        error_kw=dict(ecolor="red", capsize=3, lw=1),
        label="Play CDA",
    )
    ax.bar(
        ks + w / 2,
        mean_M,
        width=w,
        color="#8BC34A",
        yerr=[mean_M - lo_M, hi_M - mean_M],
        error_kw=dict(ecolor="red", capsize=3, lw=1),
        label="Play M-ELO",
    )

    # —— deviation arrows (one per profile, no "floating") ———————
    # bar centres
    x_cda  = {k: k - w / 2 for k in ks}
    x_melo = {k: k + w / 2 for k in ks}

    # maps for fast lookup
    mean_M_d, mean_C_d = dict(zip(ks, mean_M)), dict(zip(ks, mean_C))
    hi_M_d,   hi_C_d   = dict(zip(ks, hi_M )), dict(zip(ks, hi_C ))

    ks_set = set(ks)
    for k in ks:

        # gains from possible unilateral moves
        gain_plus  = -np.inf    # CDA → M-ELO (k → k+1)
        gain_minus = -np.inf    # M-ELO → CDA (k → k−1)

        # CDA trader deviating (needs k+1)
        if (
            (k + 1) in ks_set
            and not np.isnan(mean_C_d[k])
            and not np.isnan(mean_M_d[k + 1])
        ):
            gain_plus = mean_M_d[k + 1] - mean_C_d[k]

        # M-ELO trader deviating (needs k−1)
        if (
            (k - 1) in ks_set
            and not np.isnan(mean_M_d[k])
            and not np.isnan(mean_C_d[k - 1])
        ):
            gain_minus = mean_C_d[k - 1] - mean_M_d[k]

        # choose the best positive gain
        best_gain, dest = max(
            (gain_plus,  k + 1),
            (gain_minus, k - 1),
            key=lambda t: t[0],
        )
        if best_gain <= 0:            # no profitable deviation
            continue

        if dest == k + 1:             # CDA → M-ELO
            tail_x, tail_y = x_cda[k],  hi_C_d[k]
            head_x, head_y = x_melo[dest], hi_M_d[dest]
        else:                         # M-ELO → CDA
            tail_x, tail_y = x_melo[k], hi_M_d[k]
            head_x, head_y = x_cda[dest], hi_C_d[dest]

        # draw arrow from losing-bar top to winning-bar top
        ax.annotate(
            "",
            xy=(head_x, head_y),
            xytext=(tail_x, tail_y),
            arrowprops=dict(
                arrowstyle="-|>",
                lw=1,
                color="black",
                shrinkA=1,
                shrinkB=1,
            ),
            zorder=3,
        )
    # ————————————————————————————————————————————————

    ax.set_xlabel(f"# {role}s choosing M-ELO")
    ax.set_ylabel("Average payoff")
    ax.set_title(
        f"{role} – average payoffs (95 % CI)\n"
        "(arrows = most profitable one-agent deviation)"
    )
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend()
    plt.xticks(ks, rotation=45)
    plt.tight_layout()
    return fig


# ── RUN ───────────────────────────────────────────────────────────
df_mobi = aggregate_with_interpolation("MOBI")
df_zi   = aggregate_with_interpolation("ZI")

plot_role("MOBI", df_mobi)
plot_role("ZI",   df_zi)
plt.show()