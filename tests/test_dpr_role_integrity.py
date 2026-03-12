import numpy as np
import torch
import pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from marketsim.egta.schedulers.dpr import DPRScheduler
from marketsim.game.role_symmetric_game import RoleSymmetricGame
from marketsim.egta.core.game import Game


def build_dummy_game():
    # Two roles, each with two strategies
    role_names = ["R1", "R2"]
    num_players_per_role = [3, 2]
    strategy_names_per_role = [["A1", "A2"], ["B1", "B2"]]

    # Empty payoff/config tables are fine for this structural test
    rsg = RoleSymmetricGame(
        role_names=role_names,
        num_players_per_role=num_players_per_role,
        strategy_names_per_role=strategy_names_per_role,
        device="cpu",
    )
    return Game(rsg)


def test_dpr_missing_deviation_role_consistency():
    game = build_dummy_game()

    # Flatten strategy list in same order as scheduler expects
    all_strats = [s for role_strats in game.strategy_names_per_role for s in role_strats]

    sched = DPRScheduler(
        strategies=all_strats,
        num_players=int(sum(game.num_players_per_role)),
        role_names=game.role_names,
        num_players_per_role=[int(x) for x in game.num_players_per_role],
        strategy_names_per_role=game.strategy_names_per_role,
        reduction_size_per_role={"R1": 3, "R2": 2},
        batch_size=2,
        seed=1,
    )

    # Mixture that plays only the first strategy of each role
    mixture = np.array([1.0, 0.0, 1.0, 0.0], dtype=float)

    dev_profiles = sched.missing_deviations(mixture, game)

    # Build a quick lookup: role -> set(strategies for that role)
    role_to_strats = {r: set(strats) for r, strats in zip(game.role_names, game.strategy_names_per_role)}

    for prof in dev_profiles:
        for role, strat in prof:
            assert strat in role_to_strats[role], (
                f"Strategy {strat} assigned to wrong role {role} in profile {prof}") 