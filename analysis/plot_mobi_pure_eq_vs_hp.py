import argparse
import json
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt


def melo_prob(mix: List[float], strat_counts: List[int], mobi_role_idx: int, melo_strat_idx: int) -> float:
    """Return probability mass on M-ELO inside MOBI role."""
    start = sum(strat_counts[:mobi_role_idx])
    return mix[start + melo_strat_idx]


def main():
    parser = argparse.ArgumentParser(description="Plot #M-ELO users in pure equilibria vs holding period")
    parser.add_argument("json_file", type=Path, help="Path to rsg_mobi_only.json (solutions)")
    parser.add_argument("--out", type=Path, default=Path("hp_pure_eq.png"), help="Base name for output image(s)")
    parser.add_argument("--mobi-role", default="MOBI", help="Role name for MOBI traders")
    parser.add_argument("--melo-key", default="_0_100", help="Substring that identifies the M-ELO strategy name")
    parser.add_argument("--role-size", type=int, default=28, help="Number of MOBI players")
    parser.add_argument("--best-line", action="store_true",
                        help="Plot a single line that takes the *minimum-regret* equilibrium per HP (shows tipping)")
    parser.add_argument("--replicator-avg", action="store_true",
                        help="Plot expected #M-ELO MOBIs averaging over all replicator equilibria that clear the regret cutoff")
    parser.add_argument("--avg-all", action="store_true",
                        help="Plot expected #M-ELO MOBIs averaging over *all* equilibria (any type) that meet the regret cutoff")
    parser.add_argument("--max-regret", type=float, default=1e-3,
                        help="Discard equilibria whose regret exceeds this value (default 1e-3)")
    parser.add_argument("--max-hp", type=int, default=None,
                        help="Ignore HP values greater than this (e.g., 400 to focus on low range)")
    parser.add_argument("--exclude-hp", type=int, nargs="*", default=[],
                        help="Holding-period values to ignore entirely (e.g. 120)")
    parser.add_argument("--avg-min-melo", type=float, default=0.0,
                        help="When using --replicator-avg, ignore replicator equilibria whose M-ELO share is below this threshold (e.g. 0.01 to drop CDA-pure mixes)")
    parser.add_argument("--replicator-only", action="store_true",
                        help="Ignore grid_xx equilibria and use replicator results only")
    args = parser.parse_args()

    data = json.loads(args.json_file.read_text())

    hp_vals_best, counts_best = [], []
    hp_vals_melo, counts_melo = [], []
    hp_vals_cda, counts_cda = [], []
    hp_vals_rep, counts_rep   = [], []  # replicator-average
    hp_vals_all, counts_all   = [], []  # average over all eq types

    for hp_str, entry in data.items():
        hp = int(hp_str)
        if (args.max_hp is not None and hp > args.max_hp) or hp in args.exclude_hp:
            continue  # skip HP beyond cutoff
        role_names = entry["role_names"]
        strat_names = entry["strategy_names_per_role"]
        try:
            mobi_idx = role_names.index(args.mobi_role)
        except ValueError:
            continue

        # locate M-ELO strategy index within MOBI role
        melo_idx = None
        for idx, name in enumerate(strat_names[mobi_idx]):
            if args.melo_key.lower() in name.lower():
                melo_idx = idx
                break
        # fallback: if substring matches multiple or none, assume first strategy is M-ELO
        if melo_idx is None and strat_names[mobi_idx]:
            melo_idx = 0

        if melo_idx is None:
            continue

        # choose equilibrium with smallest regret that is (almost) pure
        best_overall = None  # smallest regret, any type
        best_reg_all = float("inf")
        best_melo = None
        best_cda  = None
        best_reg_melo = float("inf")
        best_reg_cda  = float("inf")

        # collect equilibria for optional averaging later
        rep_probs: List[float] = []
        all_probs: List[float] = []

        for eq in entry["equilibria"]:
            if args.replicator_only and eq["type"].startswith("grid"):
                continue  # skip grid candidate entirely

            mix = eq["mixture"]
            # flat mixture length = sum role sizes; infer role sizes from strat_names
            flat_sizes = [len(lst) for lst in strat_names]
            start = sum(flat_sizes[:mobi_idx])
            mobi_slice = mix[start : start + flat_sizes[mobi_idx]]
            prob_melo = mobi_slice[melo_idx]
            reg = eq["regret"]

            if reg > args.max_regret:
                continue  # skip high-regret candidate

            if reg < best_reg_all:
                best_reg_all = reg
                best_overall = (prob_melo, reg)

            # pure M-ELO
            if prob_melo >= 0.99 and reg < best_reg_melo:
                best_reg_melo = reg
                best_melo = mix

            # pure CDA (effectively zero M-ELO)
            if prob_melo <= 0.01 and reg < best_reg_cda:
                best_reg_cda = reg
                best_cda = mix

            # stash replicator candidates for average line
            if eq["type"] == "replicator" and prob_melo >= args.avg_min_melo:
                rep_probs.append(prob_melo)

            # stash all candidates (respect max_regret)
            all_probs.append(prob_melo)

        if best_melo is not None:
            hp_vals_melo.append(hp)
            counts_melo.append(args.role_size)  # full M-ELO usage

        if best_cda is not None:
            hp_vals_cda.append(hp)
            counts_cda.append(0)  # zero M-ELO in CDA-pure

        # if best-overall happened to be a CDA pure point we have it already
        if best_overall is not None and args.best_line:
            prob_melo, _ = best_overall
            hp_vals_best.append(int(hp_str))
            counts_best.append(int(round(prob_melo * args.role_size)))

        # --- replicator average line ---
        if args.replicator_avg and rep_probs:
            avg_prob = sum(rep_probs) / len(rep_probs)
            hp_vals_rep.append(hp)
            counts_rep.append(avg_prob * args.role_size)

        # --- all-equilibria average line ---
        if args.avg_all and all_probs:
            avg_prob_all = sum(all_probs) / len(all_probs)
            hp_vals_all.append(hp)
            counts_all.append(avg_prob_all * args.role_size)

    if args.avg_all and hp_vals_all:
        hp_vals_all, counts_all = zip(*sorted(zip(hp_vals_all, counts_all)))
        plt.figure(figsize=(8,4))
        plt.plot(hp_vals_all, counts_all, "o-")
        plt.xlabel("Holding Period")
        plt.ylabel("Expected # MOBIs using M-ELO (avg of all ε-eq)")
        plt.title("Expected MOBI M-ELO Usage vs Holding Period (all ε-eq)")
        plt.grid(alpha=0.3)
        out_all = args.out.with_name(args.out.stem + "_avg_all" + args.out.suffix)
        out_all.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_all, dpi=150)
        print("Saved average-all plot to", out_all)
        return

    if args.replicator_avg and hp_vals_rep:
        hp_vals_rep, counts_rep = zip(*sorted(zip(hp_vals_rep, counts_rep)))
        plt.figure(figsize=(8, 4))
        plt.plot(hp_vals_rep, counts_rep, "o-")
        plt.xlabel("Holding Period")
        plt.ylabel("Expected # MOBIs using M-ELO (replicator avg)")
        plt.title("Expected MOBI M-ELO Usage vs Holding Period")
        plt.grid(alpha=0.3)
        out_rep = args.out.with_name(args.out.stem + "_replicator" + args.out.suffix)
        out_rep.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_rep, dpi=150)
        print("Saved replicator-average plot to", out_rep)
        return  # done – don't produce other modes simultaneously

    if hp_vals_melo:
        hp_vals_melo, counts_melo = zip(*sorted(zip(hp_vals_melo, counts_melo)))
        plt.figure(figsize=(8,4))
        plt.plot(hp_vals_melo, counts_melo, "-o")
        plt.xlabel("Holding Period")
        plt.ylabel("# MOBIs using M-ELO (pure M-ELO eq)")
        plt.title("Pure M-ELO Equilibria vs Holding Period")
        plt.grid(alpha=0.3)
        out_melo = args.out.with_name(args.out.stem + "_melo" + args.out.suffix)
        out_melo.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_melo)
        print("Saved M-ELO plot to", out_melo)

    if hp_vals_cda:
        hp_vals_cda, counts_cda = zip(*sorted(zip(hp_vals_cda, counts_cda)))
        plt.figure(figsize=(8,4))
        plt.plot(hp_vals_cda, counts_cda, "-o", color="orange")
        plt.xlabel("Holding Period")
        plt.ylabel("# MOBIs using M-ELO (pure CDA eq)")
        plt.title("Pure CDA Equilibria vs Holding Period")
        plt.grid(alpha=0.3)
        out_cda = args.out.with_name(args.out.stem + "_cda" + args.out.suffix)
        out_cda.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_cda)
        print("Saved CDA plot to", out_cda)

    # --- draw ---
    if args.best_line and hp_vals_best:
        plt.figure(figsize=(6,4))
        # sort by HP to avoid the spaghetti look
        hp_vals_best, counts_best = zip(*sorted(zip(hp_vals_best, counts_best)))
        plt.plot(hp_vals_best, counts_best, "o-", color="tab:green")
        plt.xlabel("Holding period")
        plt.ylabel("# MOBIs playing M-ELO in min-regret eq")
        plt.tight_layout()
        plt.savefig(args.out, dpi=150)
    else:
        plt.figure(figsize=(6,4))
        plt.plot(*zip(*sorted(zip(hp_vals_melo, counts_melo))), "o-", label="pure M-ELO")
        plt.plot(*zip(*sorted(zip(hp_vals_cda, counts_cda))), "o-", label="pure CDA")
        plt.xlabel("Holding period")
        plt.ylabel("# MOBIs playing M-ELO in pure eq")
        plt.legend()
        base = args.out.stem
        plt.tight_layout()
        plt.savefig(args.out.parent / f"{base}_melo.png", dpi=150)
        plt.clf()
        plt.figure(figsize=(6,4))
        plt.plot(hp_vals_cda, counts_cda, "o-", color="tab:orange")
        plt.xlabel("Holding period")
        plt.ylabel("# MOBIs playing M-ELO in pure CDA eq")
        plt.tight_layout()
        plt.savefig(args.out.parent / f"{base}_cda.png", dpi=150)
    print("Saved plots to", args.out.parent)


if __name__ == "__main__":
    main() 