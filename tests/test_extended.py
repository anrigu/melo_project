import pytest
import torch
import asyncio
import numpy as np

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so standalone runs work like test_egta.py
import pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from marketsim.egta.core.game import Game
from marketsim.game.role_symmetric_game import RoleSymmetricGame
from marketsim.egta.solvers.equilibria import replicator_dynamics, quiesce_sync
from marketsim.egta.schedulers.dpr import DPRScheduler
from marketsim.egta.simulators.melo_wrapper import MeloSimulator


# ---------------------------------------------------------------------------
# Helpers -------------------------------------------------------------------
# ---------------------------------------------------------------------------

def _coordination_payoff_data():
    """Return payoff_data for a 2×2 coordination game (pure NE on A/A and B/B)."""
    # (player_id, strategy, payoff)
    prof_AA = [("p0", "A", 2.0), ("p1", "A", 2.0)]
    prof_BB = [("p0", "B", 2.0), ("p1", "B", 2.0)]
    prof_AB = [("p0", "A", 0.0), ("p1", "B", 0.0)]
    prof_BA = [("p0", "B", 0.0), ("p1", "A", 0.0)]
    return [prof_AA, prof_BB, prof_AB, prof_BA]


def _role_symmetric_payoff_data():
    """Return payoff_data for a simple matching‐pennies style 1-vs-1 game."""
    # (player_id, role, strategy, payoff)
    data = []
    # Profiles: (Row,Col)
    # HH -> Row +1, Col -1
    data.append([("p0", "Row", "H", 1.0), ("p1", "Col", "H", -1.0)])
    # HT -> Row -1, Col +1
    data.append([("p0", "Row", "H", -1.0), ("p1", "Col", "T", 1.0)])
    # TH -> Row -1, Col +1
    data.append([("p0", "Row", "T", -1.0), ("p1", "Col", "H", 1.0)])
    # TT -> Row +1, Col -1
    data.append([("p0", "Row", "T", 1.0), ("p1", "Col", "T", -1.0)])
    return data


# ---------------------------------------------------------------------------
# 1. Game.from_payoff_data – symmetric path ----------------------------------
# ---------------------------------------------------------------------------

def test_game_creation_symmetric():
    payoff_data = _coordination_payoff_data()
    g = Game.from_payoff_data(payoff_data)
    assert not g.is_role_symmetric
    assert g.num_players == 2
    assert set(g.strategy_names) == {"A", "B"}

    # Pure A/A should have zero regret (equilibrium)
    mix_pure_A = torch.tensor([1.0, 0.0])
    assert pytest.approx(g.regret(mix_pure_A), abs=1e-6) == 0.0


# ---------------------------------------------------------------------------
# 2. Game.from_payoff_data – role-symmetric path -----------------------------
# ---------------------------------------------------------------------------

def test_game_creation_role_symmetric():
    payoff_data = _role_symmetric_payoff_data()
    g = Game.from_payoff_data(payoff_data)
    assert g.is_role_symmetric
    assert g.num_strategies == 4

    mix = torch.ones(4) / 4
    r = g.regret(mix)
    assert np.isfinite(r)


# ---------------------------------------------------------------------------
# 3. RoleSymmetricGame.deviation_payoffs / regret ----------------------------
# ---------------------------------------------------------------------------

def test_rolesymmetricgame_regret_zero_on_strict_ne():
    rsg = RoleSymmetricGame(
        role_names=["M", "Z"],
        num_players_per_role=[1, 1],
        strategy_names_per_role=[["X"], ["Y"]],
        rsg_config_table=torch.tensor([[1.0, 1.0]]),
        rsg_payoff_table=torch.tensor([[0.5], [1.2]]),
    )
    mix = torch.tensor([1.0, 1.0])  
    mix = mix / mix.sum()
    assert rsg.regret(mix) == 0.0


# ---------------------------------------------------------------------------
# 4. Replicator dynamics convergence check -----------------------------------
# ---------------------------------------------------------------------------

def test_replicator_converges_on_coordination():
    payoff_data = _coordination_payoff_data()
    g = Game.from_payoff_data(payoff_data)
    init = torch.rand(2)
    init = init / init.sum()
    eq = replicator_dynamics(g, init, iters=4000)
    assert torch.allclose(eq, torch.tensor([1.0, 0.0]), atol=1e-2) or \
           torch.allclose(eq, torch.tensor([0.0, 1.0]), atol=1e-2)


# ---------------------------------------------------------------------------
# 5. quiesce_sync does not break global event loop ---------------------------
# ---------------------------------------------------------------------------

def test_quiesce_event_loop_integrity():
    payoff_data = _coordination_payoff_data()
    g = Game.from_payoff_data(payoff_data)

    default_loop = asyncio.get_event_loop()

    eqs = quiesce_sync(g, full_game=g, num_iters=5, regret_threshold=1e-3)
    assert len(eqs) >= 1

    assert not default_loop.is_closed()


# ---------------------------------------------------------------------------
# 6. DPRScheduler – batch generation smoke test ------------------------------
# ---------------------------------------------------------------------------

def test_dprscheduler_generates_profiles():
    role_names = ["R1", "R2"]
    strats_per_role = [["A", "B"], ["X", "Y"]]
    sched = DPRScheduler(
        strategies=["A", "B", "X", "Y"],
        num_players=2,
        role_names=role_names,
        num_players_per_role=[1, 1],
        strategy_names_per_role=strats_per_role,
        reduction_size_per_role={"R1": 1, "R2": 1},
        batch_size=8,
    )

    batch = sched.get_next_batch()
    assert batch and isinstance(batch[0], list)
    for prof in batch:
        roles_in_prof = {r for r, _ in prof}
        assert roles_in_prof == set(role_names)


# ---------------------------------------------------------------------------
# 7. MeloSimulator – helper accessors ----------------------------------------
# ---------------------------------------------------------------------------

def test_melosimulator_strategy_accessors():
    sim = MeloSimulator(num_strategic_mobi=2, num_strategic_zi=2, reps=1, parallel=False, force_symmetric=False)
    assert set(sim.get_role_info()[0]) == {"MOBI", "ZI"}
    assert {s for lst in sim.strategy_names_per_role for s in lst} == set(sim.get_strategies()) 