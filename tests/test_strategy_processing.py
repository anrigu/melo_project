import os, sys, pathlib
# Ensure project root is on PYTHONPATH when tests are run from within the tests directory.
project_root = pathlib.Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pytest
from marketsim.egta.simulators.melo_wrapper import MeloSimulator

@pytest.fixture
def strategy_lists():
    mobi = [
        "MOBI_0_100_shade0_0",
        "MOBI_100_0_shade0_200",
        "MOBI_100_0_shade250_500",
    ]
    zi = [
        "ZI_100_0_shade100_500",
        "ZI_0_100_shade250_500",
        "ZI_100_0_shade250_500",
    ]
    return mobi, zi


def test_melo_simulator_strategy_params(strategy_lists):
    """Ensure each supplied strategy appears in strategy_params with correct CDA/MELO split."""
    mobi, zi = strategy_lists

    sim = MeloSimulator(
        num_strategic_mobi=len(mobi),
        num_strategic_zi=len(zi),
        sim_time=1,               # tiny for fast initialise
        lam=1e-2,
        lam_r=1e-2,
        lam_melo=1e-3,
        mean=1000,
        r=0.0,
        shock_var=1.0,
        q_max=1,
        pv_var=1.0,
        shade=[10, 30],
        holding_period=0,
        num_background_zi=0,
        num_background_hbl=0,
        reps=1,
        mobi_strategies=mobi,
        zi_strategies=zi,
        parallel=False,
    )

    # (1) All strategies are registered
    assert set(sim.get_strategies()) == set(mobi + zi)

    # (2) Allocation proportions parsed correctly from names
    for strat in mobi + zi:
        params = sim.strategy_params.get(strat)
        assert params is not None, f"Missing params for {strat}"

        # Check allocation proportions
        if "0_100" in strat:            # 100% MELO, 0% CDA
            assert params["melo_proportion"] == 1.0
            assert params["cda_proportion"] == 0.0
        elif "100_0" in strat:          # 100% CDA, 0% MELO
            assert params["cda_proportion"] == 1.0
            assert params["melo_proportion"] == 0.0
        else:
            pytest.fail(f"Unrecognised proportion pattern in {strat}")

        # Check that shade pair matches what is encoded in the name
        try:
            shade_suffix = strat.split("_shade", 1)[1]
            low, high = map(int, shade_suffix.split("_", 1))
            assert params["shade"] == [low, high], f"Shade mismatch for {strat}"
        except Exception:
            pytest.fail(f"Cannot parse shade from {strat}")

# ---------------------------------------------------------------------------
# Additional behaviour: holding-period propagation & profile simulation
# ---------------------------------------------------------------------------

def test_simulator_holding_period_and_profile(strategy_lists):
    """Simulator should store the holding_period and accept a full role-symmetric profile."""
    mobi, zi = strategy_lists

    holding = 7  # unique value to check propagation

    sim = MeloSimulator(
        num_strategic_mobi=len(mobi),
        num_strategic_zi=len(zi),
        sim_time=1,
        lam=1e-2,
        lam_r=1e-2,
        lam_melo=1e-3,
        mean=1000,
        r=0.0,
        shock_var=1.0,
        q_max=1,
        pv_var=1.0,
        shade=[10, 30],
        holding_period=holding,
        num_background_zi=0,
        num_background_hbl=0,
        reps=1,
        mobi_strategies=mobi,
        zi_strategies=zi,
        parallel=False,
    )

    # Verify the parameter travelled to the inner simulator.
    assert sim.holding_period == holding

    # Build a profile matching the required number of strategic players
    profile = [("MOBI", s) for s in mobi] + [("ZI", s) for s in zi]

    obs = sim.simulate_profile(profile)

    # Observation should have one payoff per strategic player
    assert len(obs.payoffs) == len(profile)
    # Profile key stored in observation should equal the sorted version we expect
    expected_key = tuple(sorted(profile))
    assert obs.profile_key == expected_key 