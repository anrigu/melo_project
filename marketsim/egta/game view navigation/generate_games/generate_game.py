"""
Adapted from gameanalysis gamegen.py
"""

import numpy as np
import numpy.random as r
from scipy.special import binom
from aggfn import *
import pandas as pd
#from param_game_fam import ParameterizedGameFamily


def get_random_mask(prob, num_strats, num_funcs, vals=[]):
    """Returns a random mask with at least one true in every row and col"""
    if vals == []:
        vals = np.random.random((num_strats, num_funcs))
    mask = vals < prob
    mask[vals.argmin(0), np.arange(num_funcs)] = True
    mask[np.arange(num_strats), vals.argmin(1)] = True
    return mask

def get_action_weights(prob, num_strats, num_funcs, vals=[], weights=[]):
    if weights == []:
        weights = np.random.normal(0, 1, (num_strats, num_funcs))
    return get_random_mask(prob, num_strats, num_funcs, vals) * weights

def get_function_inputs(prob, num_funcs, num_strats, vals=[]):
    if vals == []:
        vals = np.random.random((num_strats, num_funcs))
    mask = vals < prob
    inds = np.arange(num_funcs)
    mask[vals.argmin(0), inds] = True
    mask[vals.argmax(0), inds] = False
    func_bool = np.where(mask)
    #function_neighborhoods = [[] for strat in range(num_funcs)] # FIXME
    function_neighborhoods = np.zeros((num_funcs, num_strats))
    for i in range(len(func_bool[0])):
        #function_neighborhoods[func_bool[1][i]].append(func_bool[0][i]) FIXME
        function_neighborhoods[func_bool[1][i], func_bool[0][i]]=1
    return function_neighborhoods

def get_function_outputs(num_players, num_funcs, degree, period, game_type):
    if isinstance(degree, int):
        degree = (0,) * (degree - 1) + (1,)
    max_degree = len(degree)

    # This setup makes it so that the beat frequencies approach period
    periods = ((np.arange(1, num_funcs + 1) +
                np.random.random(num_funcs) / 2 - 1 / 4) *
               period / num_funcs)
    sine_offset = np.random.random((num_funcs, 1))

    zeros = (np.random.random((num_funcs, max_degree)) * 1.5 - 0.25) * num_players
    terms = np.arange(num_players + 1)[:, None] - zeros[:, None]
    choices = np.random.choice(
        max_degree, (num_funcs, num_players + 1), True, degree)
    terms[choices[..., None] < np.arange(max_degree)] = 1
    poly = terms.prod(2) / num_players ** choices

    # The prevents too many small polynomials from making functions
    # effectively constant
    scale = poly.max() - poly.min()
    offset = poly.min() + 1

    if game_type == "add":
        return np.sin(
               (np.linspace(0, 1, num_players + 1) * periods[:, None] + sine_offset) * 2 * np.pi) + \
               (poly - offset) / (1 if np.isclose(scale, 0) else scale)

    elif game_type == "mult":
        return np.sin(
            (np.linspace(0, 1, num_players + 1) * periods[:, None] + sine_offset) * 2 * np.pi) * \
            (poly - offset) / (1 if np.isclose(scale, 0) else scale)

    elif game_type == "sine":
        return np.sin(
            (np.linspace(0, 1, num_players + 1) * periods[:, None] + sine_offset) * 2 * np.pi)

    elif game_type == "poly":
        return (poly - offset) / (1 if np.isclose(scale, 0) else scale)


def generate_sin_game(num_players, num_strats, num_funcs, game_type="add", input_prob=0.2,
                      weight_prob=0.2, period=4, degree=4):
    actions_per_role = np.array([num_strats])
    players_per_role = np.array([num_players])
    action_weights = get_action_weights(weight_prob, num_strats, num_funcs)
    action_neighborhoods = np.tile(np.arange(num_funcs), (num_strats,1))
    function_neighborhoods = get_function_inputs(input_prob, num_funcs, num_strats)
    function_outputs = get_function_outputs(num_players, num_funcs, degree, period, game_type)

    return BipartiteActionGraphGame(actions_per_role, players_per_role, function_outputs,
                                    action_weights, action_neighborhoods, function_neighborhoods)

"""
def generate_sin_game_family(mm_players, mm_strats, mm_funcs, mm_input_prob=[0.2, 0.2, 1],
                             mm_weight_prob=[0.2, 0.2, 1], period=[4], degree=[4], game_type="add"):
    max_players, max_strats, max_funcs = mm_players[-1], mm_strats[-1], mm_funcs[-1]

    action_neighborhoods = np.tile(np.arange(max_funcs), (max_strats, 1))
    function_outputs = get_function_outputs(max_players, max_funcs, degree[-1], period[-1], game_type)

    instances_info = []
    game_obj_arr = []
    for weight_prob in np.linspace(mm_weight_prob[0], mm_weight_prob[1], mm_weight_prob[2]):
        vals = np.random.random((max_strats, max_funcs))
        weights = np.random.normal(0, 1, (max_strats, max_funcs))
        action_weights = get_action_weights(weight_prob, max_strats, max_funcs, vals=vals, weights=weights)
        for input_prob in np.linspace(mm_input_prob[0], mm_input_prob[1], mm_input_prob[2]):
            vals = np.random.random((max_strats, max_funcs))
            function_neighborhoods = get_function_inputs(input_prob, max_funcs, max_strats, vals=vals)
            for num_players in range(max_players, mm_players[0]-1, -1):
                for num_strats in range(max_strats, mm_strats[0]-1,-1):
                    for num_funcs in range(max_funcs, mm_funcs[0]-1, -1):
                        instances_info.append([num_players, num_strats, num_funcs, game_type, input_prob, weight_prob, period[-1], degree[-1]])
                        instance_obj = BipartiteActionGraphGame(np.array([num_strats]), np.array([num_players]), function_outputs[:num_funcs,:],
                                                action_weights[:num_strats,:num_funcs], action_neighborhoods[:num_strats,:num_funcs], function_neighborhoods[:num_funcs,:num_strats])
                        game_obj_arr.append(instance_obj)
    columns=["num_players", "num_strats", "num_funcs", "game_type", "input_prob", "weight_prob", "period", "degree"]
    instances_df = pd.DataFrame(instances_info, columns=columns)
    return ParameterizedGameFamily(instances_df, game_obj_arr)
"""


def prisoners_dilemma():
    action_neighborhoods = np.array([[0], [1]])
    function_neighborhoods = np.array([[0], [1]])
    actions_per_role = np.array([2])
    players_per_role = np.array([2])
    function_outputs = np.array([[0., 0., 2.],[0., 3., 1.]])
    action_weights = np.array([[1], [1]])
    aggfna = BipartiteActionGraphGame(actions_per_role, players_per_role, function_outputs,
                                action_weights, action_neighborhoods, function_neighborhoods)
    return aggfna

def chicken():
    action_neighborhoods = np.array([[0], [1]])
    function_neighborhoods = np.array([[0], [1]])
    actions_per_role = np.array([2])
    players_per_role = np.array([2])
    function_outputs = np.array([[0, 1, -99], [0, -1, 0]])
    action_weights = np.array([[1], [1]])
    aggfna = BipartiteActionGraphGame(actions_per_role, players_per_role, function_outputs,
                                    action_weights, action_neighborhoods, function_neighborhoods)
    return aggfna


def rock_paper_scissors():
    action_neighborhoods = np.array([[0, 1, 5], [1, 2, 3], [3, 4, 5]])
    function_neighborhoods = np.array([[0], [0, 1], [1], [1, 2], [2], [0, 2]], dtype=object)
    actions_per_role = np.array([3])
    players_per_role = np.array([2])
    function_outputs = np.array([[0, 0, 0], [1, 0, 1], [0, 0, 0], [1, 0, 1], [0, 0, 0], [1, 0, 1]])
    action_weights = np.array([[1, -1, 1], [1, 1, -1], [1, 1, -1]])

    aggfna = BipartiteActionGraphGame(actions_per_role, players_per_role, function_outputs,
                                action_weights, action_neighborhoods, function_neighborhoods)
    return aggfna







def main():
    # num_players = 100
    # num_strats = 3
    # num_funcs = 10
    # game = generate_sin_game(num_players, num_strats, num_funcs)
    # profiles = np.array([[1/3, 1/3,1/3],[1/4, 1/2, 1/4], [5/8, 1/8, 1/4], [1, 0, 0]])
    # # dev_pays = np.empty((0,num_strats), float)
    # # for prof in profiles:
    # #     dev_pays = np.vstack((dev_pays, game.deviation_payoffs(prof)))
    # #dpays = game.many_deviation_payoffs(profiles)
    # # np.set_printoptions(precision=16)
    # # print(dev_pays)
    # # print(dpays)
    # # print(dev_pays - dpays)
    # # print(np.array_equal(dpays, dev_pays))
    # num_mixtures = 5000
    #
    # rng = np.random.default_rng()
    # mixtures = rng.dirichlet(alpha=np.full(num_strats, 1), size=num_mixtures)
    # player_counts = np.random.randint(low=50, high=num_players, size=num_mixtures)
    # configs = np.array([rng.multinomial(player_counts[i], mixtures[i]) for i in range(num_mixtures)])
    #
    # # configs = np.array([[24, 41, 35], [81, 12, 7], [58, 23, 19], [11, 9, 80], [34, 8, 58]])
    # payoffs = np.empty((0, num_strats), float)
    # for config in configs:
    #     payoffs = np.vstack((payoffs, game.get_payoffs(config)))
    # many_payoffs = game.get_many_payoffs(configs)
    # #np.set_printoptions(precision=16)
    # # print(payoffs)
    # # print(many_payoffs)
    # #print(many_payoffs - payoffs)
    # print(np.array_equal(many_payoffs, payoffs))


    """
    Test making a random game and getting payoffs
    --random old and new are the same, so old payoffs must be correct
    --seems to suggest maybe there's something wrong with the definition of the rps game
    """
    """
    num_players = 50
    num_strats = 5
    num_funcs = 5
    game = generate_sin_game(num_players, num_strats, num_funcs)
    #print(game.__repr__())
    profiles = [[50, 0, 0, 0, 0], [0, 25, 0, 25, 0], [0, 0, 0, 30, 20]]
    for p in profiles:
        print('Profile: {0}'.format(p))
        print('Payoff new: {0}'.format(game.get_payoffs2(p)))
        print('Payoff old: {0}'.format(game.get_payoffs(p)))
        print()
    """

    """
    Test how to access payoffs for profiles
    --maybe there's something wrong with the rps game definition....
    """
    """
    game = rock_paper_scissors()
    #temp = game.__repr__()
    #print(temp)

    profiles = [[2, 0, 0], [1, 1, 0], [1, 0, 1], [0, 2, 0], [0, 1, 1], [0, 0, 2]]
    for p in profiles:
        print(p)
        print('Old method: {0}'.format(game.get_payoffs(p)))
        print('New method: {0}'.format(game.get_payoffs2(p)))
        print()
    """


if __name__ == "__main__":
    main()
