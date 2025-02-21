import itertools
import numpy as np
from collections import defaultdict
import pandas as pd
from game import Game
import open_spiel as sp
from cvxopt import solvers, matrix
import gambit

def process_game_data(raw_data): #generates game object for symmetric games
    """
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
    
    strategy_names = sorted(list(strategy_names))

    profile_dict = defaultdict(lambda: {"count": 0, "payoffs": defaultdict(list)})

    for profile in raw_data:
        strat_count = tuple(sorted([(strategy, 
                                     sum(1 for _, s, _ in profile if s == strategy)) 
                                     for strategy in strategy_names]))
        
        profile_dict[strat_count]["count"] += 1
        for _, strategy, payoff in profile:
            profile_dict[strat_count]["payoffs"][strategy].append(payoff)

    #check for repeats in profile_dict
    print(list(profile_dict.keys())[0])
    for strat_count, data in profile_dict.items():
        if data["count"] > 1:
            print(f"Repeat profile: {strat_count}")

    profiles = []
    payoffs = []
    for strat_count, data in profile_dict.items(): #here we get expected payoffs for each strategy profile
        profiles.append([count for _, count in strat_count])  

        expected_payoffs = [np.mean(data["payoffs"][strat]) if data["payoffs"][strat] else 0 
                            for strat, _ in strat_count]
        
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
[ #2, 4
    ['agent_1', 'MELO + ZI STRAT', 45],
    ['agent_2', 'LIT_ORDERBOOK + ZI STRAT', 20],
    ['agent_3', 'MELO + ZI STRAT', 13],
    ['agent_4', 'LIT_ORDERBOOK + ZI STRAT', 2],
    ['agent_5', 'MELO + ZI STRAT', 90],
    ['agent_6', 'MELO + ZI STRAT', 10],
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
]]


game_single = process_game_data(raw_data_single)
for row in game_single.get_payoff_matrix():
    print(row)

new_raw_data = [
[ #5, 1
    ['agent_1', 'MELO + ZI STRAT', 11],
    ['agent_2', 'LIT_ORDERBOOK + ZI STRAT', 11],
    ['agent_3', 'LIT_ORDERBOOK + ZI STRAT', 11],
    ['agent_4', 'LIT_ORDERBOOK + ZI STRAT', 11],
    ['agent_5', 'LIT_ORDERBOOK + ZI STRAT', 11],
    ['agent_6', 'LIT_ORDERBOOK + ZI STRAT', 3]
],
[ #3, 3
    ['agent_1', 'MELO + ZI STRAT', 9],
    ['agent_2', 'LIT_ORDERBOOK + ZI STRAT', 9],
    ['agent_3', 'MELO + ZI STRAT', 9],
    ['agent_4', 'LIT_ORDERBOOK + ZI STRAT', 9],
    ['agent_5', 'LIT_ORDERBOOK + ZI STRAT', 9],
    ['agent_6', 'MELO + ZI STRAT', 9]
]


]
#update payoffs for game_single
new_payoffs = [[100, 100], [100, 100]]
new_profiles = [[2, 4], [2, 4]]

prior_payoff_matrix, new_payoff_matrix = game_single.update_payoffs(new_raw_data)  
print("Prior payoff matrix")
for row in prior_payoff_matrix:
    print(row)
print("New payoff matrix")
for row in new_payoff_matrix:
    print(row)

    

