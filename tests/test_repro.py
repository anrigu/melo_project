"""Deterministic reproducibility tests.

Ensures that with fixed global seeds the stochastic parts of the solver
produce identical outputs across runs.
"""

import pathlib, sys, random
import numpy as np
import torch
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from marketsim.egta.core.game import Game
from marketsim.egta.solvers.equilibria import quiesce_sync


# ---------------------------------------------------------------------------
# Minimal RPS game (deterministic, fast)
# ---------------------------------------------------------------------------

def _create_rps_game():
    payoff = torch.tensor([[0, -1, 1], [1, 0, -1], [-1, 1, 0]], dtype=torch.float32)

    class RPS:
        def __init__(self):
            self.strategy_names = ["R", "P", "S"]
            self.num_strategies = 3
            self.is_role_symmetric = True
            self.role_names = ["Player"]
            self.strategy_names_per_role = [self.strategy_names]
            self.num_players_per_role = torch.tensor([2])
            self.game = self
            self.device = torch.device("cpu")
            self.num_players = 2  # for symmetric wrapper
            self.num_actions = 3

        def deviation_payoffs(self, mix):
            return payoff @ mix

        def regret(self, mix):
            pay = self.deviation_payoffs(mix)
            return (torch.max(pay) - (mix @ pay)).item()

    return RPS()


# ---------------------------------------------------------------------------
# Helper to run QUIESCE under a fixed seed
# ---------------------------------------------------------------------------

def _run_quiesce(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    game = Game(_create_rps_game())
    eqs = quiesce_sync(
        game,
        full_game=game,
        num_iters=4,
        num_random_starts=2,
        regret_threshold=1e-3,
        restricted_game_size=3,
        solver="replicator",
        solver_iters=400,
        verbose=False,
    )
    # Return only mixtures for comparison
    return [mix.clone() for mix, _ in eqs]


def _compare_equilibria(list1, list2, tol=1e-6):
    assert len(list1) == len(list2)
    for m1, m2 in zip(list1, list2):
        assert torch.allclose(m1, m2, atol=tol)


def test_quiesce_deterministic_reproducibility():
    seed = 12345
    eq_first = _run_quiesce(seed)
    eq_second = _run_quiesce(seed)
    _compare_equilibria(eq_first, eq_second) 