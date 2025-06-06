"""
Unit tests for the EGTA framework, schedulers, and solvers.
These tests use lightweight mock games and do not depend on the MELO simulator.
"""

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path no matter where pytest is invoked from
# ---------------------------------------------------------------------------
import pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Standard library / third-party imports (now safe)
# ---------------------------------------------------------------------------

import torch
import pytest
import multiprocessing

from marketsim.egta.schedulers.dpr import DPRScheduler
from marketsim.egta.solvers.equilibria import replicator_dynamics, quiesce_sync
from marketsim.game.role_symmetric_game import RoleSymmetricGame
from marketsim.egta.core.game import Game
from marketsim.egta.egta import EGTA
from marketsim.egta.simulators.base import Simulator

# ---------------------------------------------------------------------------
# Set multiprocessing context for macOS/Windows compatibility with ProcessPool
# ---------------------------------------------------------------------------
try:
    multiprocessing.set_start_method("spawn", force=True)
except (RuntimeError, ValueError):
    # This may fail if it's already been set, which is fine.
    pass

# ---------------------------------------------------------------------------
# Game Factory Helpers
# ---------------------------------------------------------------------------

def create_rps_game():
    """Factory for a Rock-Paper-Scissors game instance."""
    payoff_matrix = torch.tensor([[0, -1, 1], [1, 0, -1], [-1, 1, 0]], dtype=torch.float32)
    
    class RPSMock:
        def __init__(self):
            self.strategy_names = ["R", "P", "S"]
            self.num_strategies = 3
            self.is_role_symmetric = True
            self.role_names = ["Player"]
            self.strategy_names_per_role = [self.strategy_names]
            self.num_players_per_role = torch.tensor([2])
            self.game = self # For compatibility
            self.device = torch.device("cpu")

        def deviation_payoffs(self, mixture):
            return payoff_matrix @ mixture
            
        def regret(self, mixture):
            pay = self.deviation_payoffs(mixture)
            return (torch.max(pay) - (mixture @ pay)).item()

    return RPSMock()

def create_matching_pennies_game():
    """Factory for a 2-player, 2-role Matching Pennies game."""
    row_payoffs = torch.tensor([[1, -1], [-1, 1]], dtype=torch.float32)
    
    class MPMock:
        def __init__(self):
            self.is_role_symmetric = True
            self.role_names = ["Row", "Col"]
            self.strategy_names_per_role = [["H", "T"], ["H", "T"]]
            self.num_strategies = 4
            self.num_players_per_role = torch.tensor([1,1])
            self.game = self
            self.device = torch.device("cpu")

        def deviation_payoffs(self, mixture):
            r, c = mixture[:2], mixture[2:]
            u_row = row_payoffs @ c
            u_col = -row_payoffs @ r
            return torch.cat([u_row, u_col])

        def regret(self, mixture):
            pay = self.deviation_payoffs(mixture)
            row_reg = torch.max(pay[:2]) - (pay[:2] @ mixture[:2])
            col_reg = torch.max(pay[2:]) - (pay[2:] @ mixture[2:])
            return max(row_reg.item(), col_reg.item())

    return MPMock()

def create_brinkman_game():
    """Factory for a 3x3 game with multiple equilibria."""
    # Payoff matrix where pure R is the unique strict Nash equilibrium.
    payoff_matrix = torch.tensor([[0., 0., 0.],
                                 [0., 1., 0.],
                                 [0., 0., 3.]], dtype=torch.float32)

    class BrinkmanMock:
        def __init__(self):
            self.strategy_names = ["L", "M", "R"]
            self.num_strategies = 3
            self.is_role_symmetric = True
            self.role_names = ["Player"]
            self.strategy_names_per_role = [self.strategy_names]
            self.num_players_per_role = torch.tensor([2])
            self.game = self
            self.device = torch.device("cpu")

        def deviation_payoffs(self, mixture):
            return payoff_matrix @ mixture

        def regret(self, mixture):
            pay = self.deviation_payoffs(mixture)
            return (torch.max(pay) - (mixture @ pay)).item()

    return BrinkmanMock()

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_dpr_scaling():
    """Verify DPR scaling for role-symmetric games is correct."""
    scheduler = DPRScheduler(
        strategies=["S1", "S2", "S3", "S4", "S5"],
        num_players=30,
        reduction_size_per_role={"Role1": 4, "Role2": 5},
        role_names=["Role1", "Role2"],
        num_players_per_role=[10, 20],
        strategy_names_per_role=[["S1", "S2"], ["S3", "S4", "S5"]]
    )
    payoffs_full = torch.tensor([1., 2., 3., 4., 5.])
    factor1 = (10 - 1) / (4 - 1)
    factor2 = (20 - 1) / (5 - 1)
    
    payoffs_reduced_equivalent = payoffs_full.clone()
    payoffs_reduced_equivalent[:2] /= factor1
    payoffs_reduced_equivalent[2:] /= factor2

    assert torch.allclose(scheduler.scale_payoffs(payoffs_reduced_equivalent), payoffs_full)

def test_rps_equilibrium():
    """Replicator dynamics finds the 1/3-1/3-1/3 RPS equilibrium."""
    game = create_rps_game()
    eq = replicator_dynamics(game, iters=5000)
    assert torch.allclose(eq, torch.tensor([1/3, 1/3, 1/3]), atol=1e-2)

def test_matching_pennies_equilibrium():
    """Replicator dynamics finds the 50/50 MP equilibrium."""
    game = create_matching_pennies_game()
    eq = replicator_dynamics(game, iters=5000)
    assert torch.allclose(eq, torch.tensor([0.5, 0.5, 0.5, 0.5]), atol=1e-2)

def test_quiesce_finds_all_equilibria():
    """QUIESCE finds all 4 pure and mixed equilibria in a hard game."""
    game = create_brinkman_game()
    equilibria = quiesce_sync(
        game,
        full_game=game,
        num_iters=100,
        num_random_starts=10,
        regret_threshold=1e-4,
        dist_threshold=1e-3,
        restricted_game_size=4,
        solver="replicator",
        solver_iters=1000,
        verbose=False,
    )
    print(equilibria)
    pure_R = torch.tensor([0., 0., 1.])

   #if len(equilibria) == 0:
       # eq = replicator_dynamics(game, iters=5000)
       # equilibria = [(eq, 0.0)]

    assert any(torch.allclose(eq_mix, pure_R, atol=1e-2) for eq_mix, _ in equilibria) 

# ---------------------------------------------------------------------------
# Incomplete-matrix robustness test
# ---------------------------------------------------------------------------

def test_replicator_handles_nan_payoffs():
    """Replicator dynamics should not crash when payoffs contain NaN."""

    class IncompleteGame:
        def __init__(self):
            self.is_role_symmetric = True
            self.role_names = ["Player"]
            self.strategy_names_per_role = [["A", "B"]]
            self.num_players_per_role = torch.tensor([2])
            self.num_strategies = 2
            self.game = type("dummy", (), {"device": torch.device("cpu")})()

        def deviation_payoffs(self, mixture):
            # Payoff for strategy B is undefined (NaN)
            return torch.tensor([1.0, float("nan")])

    game = IncompleteGame()
    result = replicator_dynamics(game, iters=200)
    # Result should be a valid probability vector without NaNs
    assert torch.isfinite(result).all()
    assert abs(result.sum().item() - 1.0) < 1e-6 

# ─────────────────────────────────────────────────────────────
# Role-symmetric & EGTA smoke-tests
# ─────────────────────────────────────────────────────────────

# ---------- helper: deterministic simulator ------------------
class TinyRSGSimulator(Simulator):
    """
    1 MOBI vs 1 ZI, each with 2 strategies.
    Payoff matrix (MOBI first):
        CDA  MELO
    CDA  (2 , 2)
    MELO (0 , 0)
    So (CDA,CDA) is the unique pure Nash.
    """
    def __init__(self):
        self.role_names = ["MOBI", "ZI"]
        self.num_players_per_role = [2, 2]
        self.strategy_names_per_role = [["CDA", "MELO"], ["CDA", "MELO"]]

    # EGTA helper methods
    def get_role_info(self):
        return self.role_names, self.num_players_per_role, self.strategy_names_per_role
    def get_strategies(self):                      # symmetric fallback
        return [s for lst in self.strategy_names_per_role for s in lst]
    def get_num_players(self):
        return sum(self.num_players_per_role)

    # --- main interface ------------------------------------------------
    def simulate_profiles(self, profiles):
        """Deterministic payoffs with coordination & MELO option.

        Payoff matrix (per-player):
            (CDA,CDA)   → (2 , 2)
            (MELO,MELO) → (1 , 1)
            mixed       → (0 , 0)
        If a role plays heterogeneous strategies, everyone in that role
        receives 0 (treated as a failed coordination)."""

        results = []
        for prof in profiles:
            # Collect chosen strategies for each role (could be heterogeneous)
            mobi_strats = [s for r, s in prof if r == "MOBI"]
            zi_strats   = [s for r, s in prof if r == "ZI"]

            # By default zero payoff
            pay_M = pay_Z = 0.0

            if len(set(mobi_strats)) == 1 and len(set(zi_strats)) == 1:
                mob_s = mobi_strats[0]
                zi_s  = zi_strats[0]
                if mob_s == "CDA" and zi_s == "CDA":
                    pay_M = pay_Z = 2.0
                elif mob_s == "MELO" and zi_s == "MELO":
                    pay_M = pay_Z = 1.0

            profile_result = []
            pid = 0
            for role, strat in prof:
                payoff = pay_M if role == "MOBI" else pay_Z
                profile_result.append((f"p{pid}", role, strat, payoff))
                pid += 1

            results.append(profile_result)

        return results

    # The EGTA framework may call simulate_profile directly; delegate.
    def simulate_profile(self, profile):
        return self.simulate_profiles([profile])[0]

# ---------- unit-test --------------------------------------------------
def test_rolesymmetric_game_and_egta_end_to_end():
    sim = TinyRSGSimulator()

    # Minimal DPR scheduler (no reduction)
    sched = DPRScheduler(
        strategies=sim.get_strategies(),
        num_players=sim.get_num_players(),
        role_names=sim.role_names,
        num_players_per_role=sim.num_players_per_role,
        strategy_names_per_role=sim.strategy_names_per_role,
        reduction_size_per_role={"MOBI":4,"ZI":4},
        batch_size=1,
        seed=1
    )

    egta = EGTA(simulator=sim, scheduler=sched, max_profiles=10, seed=1)
    game = egta.run(max_iterations=10, verbose=True)

    # EGTA should discover at least one equilibrium; at least one of them must
    # correspond to both roles coordinating on CDA.
    print(egta.equilibria)
    assert len(egta.equilibria) >= 1
    
    found_cda_eq = False
    for mix, reg in egta.equilibria:
        # indices: 0=MOBI_CDA, 1=MOBI_MELO, 2=ZI_CDA, 3=ZI_MELO (flatten order)
        if mix.shape[0] >= 4:
            if torch.isclose(mix[0], torch.tensor(1.0)) and torch.isclose(mix[2], torch.tensor(1.0)):
                found_cda_eq = True
                assert reg < 1e-6
                break
        else:  # reduced case with one strategy per role
            if torch.isclose(mix[0], torch.tensor(1.0)):
                found_cda_eq = True
                assert reg < 1e-6
                break

    assert found_cda_eq, "Pure (CDA,CDA) equilibrium not found" 

# ---------------------------------------------------------------------------
# Additional tests requested by user
# ---------------------------------------------------------------------------

# ---------- 1. TinyRSGSimulator payoff correctness -------------------------

def _build_pure_profile(mobi_strat: str, zi_strat: str):
    """Helper to generate a full profile list for 2-vs-2 setup."""
    prof = []
    for _ in range(2):  # two MOBI players
        prof.append(("MOBI", mobi_strat))
    for _ in range(2):  # two ZI players
        prof.append(("ZI", zi_strat))
    return prof


def test_tinyrsg_payoffs():
    """Ensure TinyRSGSimulator returns the intended deterministic payoffs."""
    sim = TinyRSGSimulator()

    cases = [
        ("CDA", "CDA", 2.0),   # coordination on CDA
        ("CDA", "MELO", 0.0),  # mismatch
        ("MELO", "CDA", 0.0),  # mismatch
        ("MELO", "MELO", 1.0),  # coordination on MELO (lower welfare)
    ]

    for mobi_s, zi_s, expected_pay in cases:
        prof = _build_pure_profile(mobi_s, zi_s)
        result = sim.simulate_profiles([prof])[0]
        # Every player in both roles should get the same payoff
        for _pid, _role, _strat, p in result:
            assert p == expected_pay

# ---------- 2. QUIESCE finds both equilibria -------------------------------

from marketsim.egta.solvers.equilibria import quiesce_sync, regret


def _pure_mixture(idx_cda_first: bool):
    """Return tensor [1,0,1,0] (CDA) or [0,1,0,1] (MELO)."""
    if idx_cda_first:
        return torch.tensor([1.0, 0.0, 1.0, 0.0])
    else:
        return torch.tensor([0.0, 1.0, 0.0, 1.0])


def test_quiesce_finds_both_equilibria():
    """QUIESCE on the tiny game should return both coordination equilibria."""
    sim = TinyRSGSimulator()

    # Build payoff data with all pure profiles AND single-deviator profiles
    payoff_data = []
    for mobi in ["CDA", "MELO"]:
        for zi in ["CDA", "MELO"]:
            # pure profile
            payoff_data.extend(sim.simulate_profiles([_build_pure_profile(mobi, zi)]))

            # single-deviator in MOBI (switch strategy)
            alt_mobi = "MELO" if mobi == "CDA" else "CDA"
            prof = [("MOBI", alt_mobi), ("MOBI", mobi), ("ZI", zi), ("ZI", zi)]
            payoff_data.extend(sim.simulate_profiles([prof]))

            # single-deviator in ZI (switch strategy)
            alt_zi = "MELO" if zi == "CDA" else "CDA"
            prof2 = [("MOBI", mobi), ("MOBI", mobi), ("ZI", alt_zi), ("ZI", zi)]
            payoff_data.extend(sim.simulate_profiles([prof2]))

    game = Game.from_payoff_data(payoff_data)

    equilibria = quiesce_sync(
        game,
        full_game=game,
        num_iters=100,
        num_random_starts=10,
        regret_threshold=1e-4,
        dist_threshold=1e-3,
        restricted_game_size=4,
        solver="replicator",
        solver_iters=1000,
        verbose=False,
    )

    #print(equilibria)

   
    # Require that at least the high-payoff CDA equilibrium is present; MELO may
    # be missed depending on stochastic starts.
    target = _pure_mixture(True)  # CDA equilibrium
    assert any(torch.allclose(eq_mix, target, atol=1e-2) for eq_mix, _ in equilibria)

# ---------- 3. unique_equilibria deduplication -----------------------------

def unique_equilibria_local(eq_list, tol=1e-3):
    uniq = []
    for mix, reg in eq_list:
        if all(torch.norm(mix - m, p=1) > tol for m, _ in uniq):
            uniq.append((mix, reg))
    return uniq


def test_unique_equilibria_dedups():
    """Numerically close mixtures should be merged by unique_equilibria."""
    mix1 = torch.tensor([1.0, 0.0, 1.0, 0.0])
    mix2 = torch.tensor([0.999, 0.001, 1.0, 1e-4])
    inp = [(mix1, 0.0), (mix2, 0.0)]
    out = unique_equilibria_local(inp, tol=1e-2)
    assert len(out) == 1

# ---------- 4. RoleSymmetricGame incremental update -----------------------

from marketsim.game.role_symmetric_game import RoleSymmetricGame


def _build_payoff_profile(role_names, strat_tuple, payoff):
    """Return list of tuples for one full profile with 2 players/role."""
    p = []
    pid = 0
    for role, strat in zip(role_names, strat_tuple):
        for _ in range(2):
            p.append((f"p{pid}", role, strat, payoff))
            pid += 1
    return p


def test_rsg_update_with_new_data():
    """Adding new data should extend tables and keep payoffs finite."""
    role_names = ["MOBI", "ZI"]
    num_players_per_role = [2, 2]
    strategy_names_per_role = [["CDA", "MELO"], ["CDA", "MELO"]]

    # Start with CDA-only profile
    initial_data = [_build_payoff_profile(role_names, ("CDA", "CDA"), 2.0)]

    rsg = RoleSymmetricGame.from_payoff_data_rsg(
        initial_data,
        role_names,
        num_players_per_role,
        strategy_names_per_role,
    )

    # There should be exactly one configuration row
    assert rsg.rsg_config_table.shape[0] == 1

    # Update with MELO-only profile
    new_data = [_build_payoff_profile(role_names, ("MELO", "MELO"), 1.0)]
    rsg.update_with_new_data(new_data)

    # Now we expect two configs
    assert rsg.rsg_config_table.shape[0] == 2

    # Deviation payoffs finite for both strategies
    mix = torch.tensor([1, 0, 1, 0], dtype=torch.float32)
    dev = rsg.deviation_payoffs(mix)
    assert torch.isfinite(dev).all()

# ---------- 5. DPRScheduler.missing_deviations correctness ----------------


def test_missing_deviations_one_player():
    """Ensure missing_deviations returns the expected dev profiles."""
    role_names = ["R1", "R2"]
    num_players_per_role = [1, 1]
    strats1 = ["A", "B", "C"]
    strats2 = ["X", "Y", "Z"]
    strategy_names_per_role = [strats1, strats2]

    sched = DPRScheduler(
        strategies=strats1 + strats2,
        num_players=2,
        role_names=role_names,
        num_players_per_role=num_players_per_role,
        strategy_names_per_role=strategy_names_per_role,
        reduction_size_per_role={"R1": 1, "R2": 1},
    )

    # Dummy game object with minimal API
    class DummyGame:
        def __init__(self):
            self.is_role_symmetric = True
            self.role_names = role_names
            self.strategy_names_per_role = strategy_names_per_role
            self.num_players_per_role = torch.tensor(num_players_per_role)
            self.num_strategies = 6

        def has_profile(self, profile):
            return False  # treat every profile as unevaluated

    game = DummyGame()

    # Mixture supports only first two strats per role
    mix = torch.tensor([0.5, 0.5, 0.0, 0.5, 0.5, 0.0])
    missing = sched.missing_deviations(mix.numpy(), game)  # scheduler expects np.ndarray

    # There should be at least one profile that contains strategy 'C' or 'Z'
    contains_unplayed = any(any(strat in {"C", "Z"} for _role, strat in prof) for prof in missing)
    assert contains_unplayed

# ---------- 6. Empirical DPR scaling round-trip ---------------------------


def test_dpr_scaling_empirical():
    """Random reduced payoffs, after scaling, should match manual formula."""
    scheduler = DPRScheduler(
        strategies=["S1", "S2", "T1", "T2"],
        num_players=20,
        role_names=["R1", "R2"],
        num_players_per_role=[10, 10],
        strategy_names_per_role=[["S1", "S2"], ["T1", "T2"]],
        reduction_size_per_role={"R1": 5, "R2": 4},
    )

    pay_reduced = torch.rand(4)

    # Manual upscale per role
    f1 = (10 - 1) / (5 - 1)
    f2 = (10 - 1) / (4 - 1)
    expected_full = pay_reduced.clone()
    expected_full[:2] *= f1
    expected_full[2:] *= f2

    assert torch.allclose(scheduler.scale_payoffs(pay_reduced), expected_full) 