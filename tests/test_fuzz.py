"""Additional fuzz/property and stress tests for EGTA components.

These expand coverage beyond deterministic unit tests, checking that
core invariants hold under randomly-generated games and larger sizes.
"""

# ---------------------------------------------------------------------------
# Boilerplate – ensure import path includes project root
# ---------------------------------------------------------------------------
import pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Std / 3rd-party imports
# ---------------------------------------------------------------------------
import pytest
import torch
import numpy as np
from hypothesis import given, settings, strategies as st

from marketsim.egta.core.game import Game
from marketsim.game.role_symmetric_game import RoleSymmetricGame
from marketsim.egta.solvers.equilibria import quiesce_sync, replicator_dynamics
from marketsim.egta.schedulers.dpr import DPRScheduler


# ---------------------------------------------------------------------------
# Helpers to create random games
# ---------------------------------------------------------------------------

def _random_role_symmetric_game(max_roles=3, max_strats_per_role=4, max_players=3, seed=None):
    rng = np.random.default_rng(seed)
    num_roles = rng.integers(1, max_roles + 1)
    role_names = [f"R{i}" for i in range(num_roles)]

    players_per_role = rng.integers(1, max_players + 1, size=num_roles).tolist()
    strategies_per_role = []
    for r in range(num_roles):
        k = rng.integers(2, max_strats_per_role + 1)
        strategies_per_role.append([f"S{r}_{j}" for j in range(k)])

    num_strats = sum(len(s) for s in strategies_per_role)

    # build a modest number of random configs (each config is counts vector)
    n_configs = rng.integers(10, 30)
    config_rows = []
    payoff_rows = np.full((num_strats, n_configs), np.nan, dtype=np.float32)

    base_utility = rng.standard_normal(num_strats)

    for c in range(n_configs):
        counts = []
        for n_p, strats in zip(players_per_role, strategies_per_role):
            # random multinomial counts within this role
            counts.extend(rng.multinomial(n_p, np.ones(len(strats)) / len(strats)))
        config_rows.append(counts)
        # Payoffs: base + small random noise
        payoff_rows[:, c] = base_utility + 0.1 * rng.standard_normal(num_strats)

    rsg = RoleSymmetricGame(
        role_names=role_names,
        num_players_per_role=players_per_role,
        strategy_names_per_role=strategies_per_role,
        rsg_config_table=torch.tensor(np.array(config_rows, dtype=np.float32)),
        rsg_payoff_table=torch.tensor(payoff_rows),
    )
    return Game(rsg)


# ---------------------------------------------------------------------------
# 1. Property-based: regret non-negative & equilibria within threshold
# ---------------------------------------------------------------------------

@given(seed=st.integers(0, 2**16 - 1))
@settings(max_examples=20, deadline=None)
def test_random_rsg_regret_and_quiesce(seed):
    """For random RSGs, regret is ≥0 and quiesce_sync returns low-regret mixes."""
    game = _random_role_symmetric_game(seed=seed)

    # Random mixture respecting role sizes
    mix = torch.rand(game.num_strategies)
    # normalise per role
    global_idx = 0
    for strats in game.strategy_names_per_role:
        sl = slice(global_idx, global_idx + len(strats))
        mix[sl] /= mix[sl].sum()
        global_idx += len(strats)

    # Regret should always be non-negative
    assert game.regret(mix) >= 0.0

    # Try to find equilibria quickly; we mainly test that any mixtures returned
    # indeed satisfy the regret threshold used.
    eqs = quiesce_sync(game, full_game=game, num_iters=3, regret_threshold=1e-2,
                       restricted_game_size=4, solver_iters=300, verbose=False)
    for m, _ in eqs:
        assert game.regret(m) <= 1e-2 + 1e-6


# ---------------------------------------------------------------------------
# 2. Stress: larger RSG still solvable (smoke) --------------------------------
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_large_rsg_quiesce_smoke():
    game = _random_role_symmetric_game(max_roles=4, max_strats_per_role=5, max_players=4, seed=42)
    eqs = quiesce_sync(game, full_game=game, num_iters=5, regret_threshold=2e-2,
                       restricted_game_size=6, solver_iters=600, verbose=False)
    # At least one equilibrium should be found
    assert len(eqs) >= 1



@given(seed=st.integers(0, 2**16 - 1))
@settings(max_examples=10, deadline=None)
def test_dpr_missing_deviations_fuzz(seed):
    rng = np.random.default_rng(seed)
    game = _random_role_symmetric_game(seed=seed)

    red_per_role = {}
    for r, n in zip(game.role_names, game.num_players_per_role):
        n_int = int(n)
        red_per_role[r] = max(2, n_int) if n_int > 1 else 1

    sched = DPRScheduler(
        strategies=game.strategy_names,
        num_players=game.num_players,
        role_names=game.role_names,
        num_players_per_role=[int(n) for n in game.num_players_per_role],
        strategy_names_per_role=game.strategy_names_per_role,
        reduction_size_per_role=red_per_role,
        batch_size=4,
        seed=seed,
    )

    mix = torch.rand(game.num_strategies)

    zero_idx = rng.integers(0, game.num_strategies)

    idx = 0
    for strats in game.strategy_names_per_role:
        sl = slice(idx, idx + len(strats))
        if sl.start <= zero_idx < sl.stop:
            mix[zero_idx] = 0.0
        mix[sl] /= mix[sl].sum() if mix[sl].sum() > 0 else 1.0
        idx += len(strats)

    unplayed = {i for i, p in enumerate(mix) if p < 1e-3}
    missing = sched.missing_deviations(mix.cpu().numpy(), game)

    if missing:
        found = any(any(game.strategy_names.index(strat) in unplayed for _role, strat in prof)
                    for prof in missing)
        assert found 