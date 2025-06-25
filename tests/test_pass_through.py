import pytest, pathlib, sys, random, os

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from marketsim.egta.simulators.melo_wrapper import MeloSimulator


def _simple_strategy_lists():
    mobi = ["MOBI_0_100_shade0_0", "MOBI_100_0_shade250_500"]
    zi   = ["ZI_0_100_shade250_500", "ZI_100_0_shade250_500"]
    return mobi, zi


def test_full_parameter_pipeline(monkeypatch):
    """End-to-end check that key parameters propagate from wrapper → core sim."""

    captured_kwargs = {}

    # Patch the pool initialiser to record the *base_kwargs* that are broadcast
    monkeypatch.setattr(
        "marketsim.egta.simulators.melo_wrapper._init_worker",
        lambda bk: captured_kwargs.update(bk),
    )

    # Patch the actual worker body so no heavy simulation runs
    monkeypatch.setattr(
        "marketsim.egta.simulators.melo_wrapper._run_melo_single_rep",
        lambda seed: {0: 0.0, 1: 0.0},
    )

    hp          = 7          # distinctive holding period
    lam_mobi    = 1e-4
    lam_zi      = 9e-4
    mobi, zi    = _simple_strategy_lists()

    sim = MeloSimulator(
        num_strategic_mobi=len(mobi),
        num_strategic_zi=len(zi),
        sim_time=10,
        lam=1e-2,
        lam_r=1e-2,
        lam_melo_mobi=lam_mobi,
        lam_melo_zi=lam_zi,
        holding_period=hp,
        mobi_strategies=mobi,
        zi_strategies=zi,
        reps=1,
        parallel=False,
    )

    # Trigger one simulation so captured_kwargs is filled
    profile = [("MOBI", mobi[0]), ("ZI", zi[0])]
    sim.simulate_profile(profile)

    # --- Assertions ---------------------------------------------------
    assert captured_kwargs, "No kwargs captured – monkeypatch failed?"

    # 1) Arrival rates forwarded correctly
    assert captured_kwargs["lam_melo_mobi"] == pytest.approx(lam_mobi)
    assert captured_kwargs["lam_melo_zi"] == pytest.approx(lam_zi)

    # 2) Holding period transmitted
    assert captured_kwargs["holding_period"] == hp

    # 3) Strategy parameters intact (shade + proportions)
    params = captured_kwargs["strategy_params"]
    # Pick one representative strategy from each role
    for s in [mobi[1], zi[1]]:
        p = params[s]
        # Check shade matches suffix
        low, high = map(int, s.split("_shade",1)[1].split("_",1))
        assert p["shade"] == [low, high]
        # Check allocation proportions
        if "0_100" in s:
            assert p["melo_proportion"] == 1.0 and p["cda_proportion"] == 0.0
        elif "100_0" in s:
            assert p["cda_proportion"] == 1.0 and p["melo_proportion"] == 0.0
        else:
            pytest.fail(f"Unexpected allocation pattern in {s}") 