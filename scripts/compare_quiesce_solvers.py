import argparse
import importlib
import os, sys, asyncio
import numpy as np
import torch
from typing import List, Tuple, Set
import json

# ---------------------------------------------------------------------------
# Ensure project root and *optional* vendored gameanalysis are on PYTHONPATH
# ---------------------------------------------------------------------------

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Fallback: if the repo contains marketsim/egta/gameanalysis-old/gameanalysis
# add that directory so the historical solver imports without system-wide
# installation.
GA_PARENT = os.path.join(ROOT_DIR, "marketsim", "egta", "gameanalysis-old")
# import expects   gameanalysis/ __init__.py  inside a sys.path entry
if os.path.isdir(os.path.join(GA_PARENT, "gameanalysis")) and GA_PARENT not in sys.path:
    sys.path.insert(0, GA_PARENT)

# -------------------   imports now safe   -------------------

from marketsim.egta.solvers.equilibria import quiesce_sync
from marketsim.egta.core.game import Game

# Old solver (vendored snapshot)
from marketsim.egta.quiesce_old.egta import innerloop as qold_inner

# ---------------------------------------------------
# Benchmark game generators (adapted from unit tests)
# ---------------------------------------------------

def create_rps_game():
    payoff = torch.tensor([
        [0., -1., 1.],
        [1., 0., -1.],
        [-1., 1., 0.],
    ])
    class RPS:
        def __init__(self):
            self.strategy_names = ["R", "P", "S"]
            self.num_strategies = 3
            self.num_actions = 3
            self.is_role_symmetric = True
            self.role_names = ["Player"]
            self.strategy_names_per_role = [self.strategy_names]
            self.num_players_per_role = torch.tensor([2])
            self.num_players = 2
            self.game = self
            self.device = torch.device("cpu")
        def deviation_payoffs(self, mix):
            return payoff @ mix
        def regret(self, mix):
            dev = self.deviation_payoffs(mix)
            return (torch.max(dev) - mix @ dev).item()
    return Game(RPS())


def create_brinkman_game():
    A = torch.tensor([
        [3., 0., 0.],
        [0., 2., 0.],
        [0., 0., 1.],
    ])
    class Brinkman:
        def __init__(self):
            self.strategy_names = ["A", "B", "C"]
            self.num_strategies = 3
            self.num_actions = 3
            self.is_role_symmetric = True
            self.role_names = ["Player"]
            self.strategy_names_per_role = [self.strategy_names]
            self.num_players_per_role = torch.tensor([2])
            self.num_players = 2
            self.game = self
            self.device = torch.device("cpu")
        def deviation_payoffs(self, mix):
            return A @ mix
        def regret(self, mix):
            dev = self.deviation_payoffs(mix)
            return (torch.max(dev) - mix @ dev).item()
    return Game(Brinkman())


def create_coordination_game():
    # 4×4 coordination: diag=1, off diag=0
    payoff = torch.eye(4)
    class Coord:
        def __init__(self):
            self.strategy_names = [f"S{i}" for i in range(4)]
            self.num_strategies = 4
            self.num_actions = 4
            self.is_role_symmetric = True
            self.role_names = ["Player"]
            self.strategy_names_per_role = [self.strategy_names]
            self.num_players_per_role = torch.tensor([4])
            self.num_players = 4
            self.game = self
            self.device = torch.device("cpu")
        def deviation_payoffs(self, mix):
            return payoff @ mix
        def regret(self, mix):
            dev = self.deviation_payoffs(mix)
            return (torch.max(dev) - mix @ dev).item()
    return Game(Coord())

BENCHMARKS = {
    "rps": create_rps_game,
    "brinkman": create_brinkman_game,
    "coord4": create_coordination_game,
}

# ---------------------------------------------------
# Utility functions
# ---------------------------------------------------

def canonical_round(mix: np.ndarray, tol: float = 1e-3) -> Tuple[Tuple[float, ...]]:
    return tuple(np.round(mix / tol) * tol)


def mix_set(eqs) -> Set[Tuple[float, ...]]:
    return {canonical_round(m) for m in eqs}


def run_new_solver(game: Game):
    eqs = quiesce_sync(
        game,
        full_game=game,
        num_iters=100,
        num_random_starts=20,
        regret_threshold=1e-3,
        dist_threshold=1e-3,
        restricted_game_size=5,
        solver="replicator",
        solver_iters=2000,
        verbose=False,
    )
    return [m.cpu().numpy() for m, _ in eqs]


def run_old_solver(game: Game):
    """Execute the historical quiesce implementation (inner_loop)."""
    agame = game.game  # unwrap to RsGame-like object expected by old code

    coro_func = qold_inner.inner_loop  # async def

    loop = asyncio.new_event_loop()
    try:
        out = loop.run_until_complete(
            coro_func(
                agame,
                regret_thresh=1e-3,
                dist_thresh=1e-3,
                restricted_game_size=5,
            )
        )
    except AttributeError as exc:
        print("⚠  Old solver could not run on this mock game:", exc)
        out = []
    finally:
        loop.close()

    return [row for row in out]
 
def main():
    parser = argparse.ArgumentParser(description="Compare new vs old quiesce solvers")
    parser.add_argument("game", help="benchmark key (rps, brinkman, coord4) or path to game JSON")
    args = parser.parse_args()

    if args.game in BENCHMARKS:
        game = BENCHMARKS[args.game]()
    else:
        # Assume JSON file path
        path = args.game
        if not os.path.isfile(path):
            parser.error(f"{path} is neither a benchmark key nor a file")

        # ensure gameanalysis is importable (vendored path already in sys.path)
        from gameanalysis import paygame, rsgame
        from marketsim.egta.quiesce_old.egta import schedgame

        with open(path, "r") as fh:
            jgame = json.load(fh)

        rs_game = paygame.game_json(jgame)

        # Adapter to satisfy marketsim.Game expectations
        import torch

        class _RsAdapter:
            def __init__(self, core):
                self._core = core
                self.strategy_names = [s for sub in core.strat_names for s in sub]
                self.num_actions = core.num_strats
                self.device = torch.device("cpu")
            def __getattr__(self, item):
                return getattr(self._core, item)
            def get_strategy_name(self, idx):
                role_idx = self._core.role_indices[idx]
                local = idx - self._core.role_starts[role_idx]
                return f"{self._core.role_names[role_idx]}:{self._core.strat_names[role_idx][local]}"

        game = Game(_RsAdapter(rs_game))

        # Build the game via the original "scheduler game" stack
        base  = rsgame.empty_names(game.role_names, game.num_players_per_role, game.strategy_names)
        full  = paygame.game_replace(base, game.deviation_payoffs(torch.eye(game.num_actions)), game.deviation_payoffs(torch.eye(game.num_actions)))
        old_g = schedgame.SchedGame(full)   # adds the helper methods

    print("Running new solver…", flush=True)
    new_eq = run_new_solver(game)
    print(f"  found {len(new_eq)} eq")

    print("Running old solver…", flush=True)
    old_eq = run_old_solver(game)
    print(f"  found {len(old_eq)} eq")

    new_set = mix_set(new_eq)
    old_set = mix_set(old_eq)

    missing = old_set - new_set
    extra   = new_set - old_set

    print("\nL1-rounded comparison (1e-3):")
    print(f"  recall {100*(1-len(missing)/max(1,len(old_set))):.1f}%  (missing {len(missing)})")
    print(f"  precision {100*(1-len(extra)/max(1,len(new_set))):.1f}% (extra {len(extra)})")
    if missing:
        print("  missing mixtures:")
        for m in sorted(missing):
            print("   ", m)
    if extra:
        print("  new-only mixtures:")
        for m in sorted(extra):
            print("   ", m)

if __name__ == "__main__":
    main() 