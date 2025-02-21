import itertools
import numpy as np
from collections import defaultdict
import pandas as pd
from game import Game
import open_spiel as sp
from cvxopt import solvers, matrix
import gambit

def process_game_data(raw_data):
    """
    convert raw payoff data into a structured game matrix.

    Parameters:
        raw_data : list of lists
            Each sublist represents a strategy profile containing agent-level data.

    Returns:
        Game object with correctly computed expected payoffs.
    """
    strategy_names = set()
    
    for profile in raw_data:
        for _, strategy, _ in profile:
            strategy_names.add(strategy)
    
    strategy_names = sorted(list(strategy_names))  #ensure consistent ordering

    profiles = []
    payoffs = []

    for profile in raw_data:
        #track strategy counts and payoffs per profile
        strat_count = {strat: 0 for strat in strategy_names}
        strat_payoffs = {strat: [] for strat in strategy_names}

        for _, strategy, payoff in profile:
            strat_count[strategy] += 1
            strat_payoffs[strategy].append(payoff)

        profiles.append([strat_count[strat] for strat in strategy_names])
        expected_payoffs = [np.mean(strat_payoffs[strat]) if strat_payoffs[strat] else 0 for strat in strategy_names]
        payoffs.append(expected_payoffs)

    return Game(strategy_names, profiles, payoffs)
  

#example payoff matrix for 6 agents and 2 strategies
"""
Melo  Lit   
0  6   | 0 50
1  5   | 50 10
2  4   | 30 20
3  3   | 50 50
4  2   | 40 70
5  1   | 10 100
6  0   | 100 0 
"""

#example raw data input for each strategy profile for 6 agents and 2 strategies
raw_data_single = [[ #2, 4
    ['agent_1', 'MELO + ZI STRAT', 50],
    ['agent_2', 'LIT_ORDERBOOK + ZI STRAT', 30],
    ['agent_3', 'MELO + ZI STRAT', 40],
    ['agent_4', 'LIT_ORDERBOOK + ZI STRAT', 20],
    ['agent_5', 'MELO + ZI STRAT', 60],
    ['agent_6', 'MELO + ZI STRAT', 45]
],
[ #3, 3
    ['agent_1', 'MELO + ZI STRAT', 50],
    ['agent_2', 'LIT_ORDERBOOK + ZI STRAT', 30],
    ['agent_3', 'MELO + ZI STRAT', 40],
    ['agent_4', 'LIT_ORDERBOOK + ZI STRAT', 20],
    ['agent_5', 'LIT_ORDERBOOK + ZI STRAT', 60],
    ['agent_6', 'MELO + ZI STRAT', 45]
],
[ #4, 2
    ['agent_1', 'MELO + ZI STRAT', 50],
    ['agent_2', 'LIT_ORDERBOOK + ZI STRAT', 30],
    ['agent_3', 'MELO + ZI STRAT', 40],
    ['agent_4', 'LIT_ORDERBOOK + ZI STRAT', 20],
    ['agent_5', 'LIT_ORDERBOOK + ZI STRAT', 60],
    ['agent_6', 'LIT_ORDERBOOK + ZI STRAT', 45]
],
[ #5, 1
    ['agent_1', 'MELO + ZI STRAT', 50],
    ['agent_2', 'LIT_ORDERBOOK + ZI STRAT', 30],
    ['agent_3', 'LIT_ORDERBOOK + ZI STRAT', 40],
    ['agent_4', 'LIT_ORDERBOOK + ZI STRAT', 20],
    ['agent_5', 'LIT_ORDERBOOK + ZI STRAT', 60],
    ['agent_6', 'LIT_ORDERBOOK + ZI STRAT', 45]
],
[ #1, 5
    ['agent_1', 'MELO + ZI STRAT', 50],
    ['agent_2', 'LIT_ORDERBOOK + ZI STRAT', 30],
    ['agent_3', 'MELO + ZI STRAT', 40],
    ['agent_4', 'MELO + ZI STRAT', 20],
    ['agent_5', 'MELO + ZI STRAT', 60],
    ['agent_6', 'MELO + ZI STRAT', 45]
], 
[ #0, 6
    ['agent_1', 'MELO + ZI STRAT', 50],
    ['agent_2', 'MELO + ZI STRAT', 30],
    ['agent_3', 'MELO + ZI STRAT', 40],
    ['agent_4', 'MELO + ZI STRAT', 20],
    ['agent_5', 'MELO + ZI STRAT', 60],
    ['agent_6', 'MELO + ZI STRAT', 45]
],
[ #6, 0
    ['agent_1', 'LIT_ORDERBOOK + ZI STRAT', 50],
    ['agent_2', 'LIT_ORDERBOOK + ZI STRAT', 30],
    ['agent_3', 'LIT_ORDERBOOK + ZI STRAT', 40],
    ['agent_4', 'LIT_ORDERBOOK + ZI STRAT', 20],
    ['agent_5', 'LIT_ORDERBOOK + ZI STRAT', 60],
    ['agent_6', 'LIT_ORDERBOOK + ZI STRAT', 45]
]]


game_single = process_game_data(raw_data_single)
for row in game_single.get_payoff_matrix():
    print(row)



