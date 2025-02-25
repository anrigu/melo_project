import itertools
import numpy as np
from collections import defaultdict
import pandas as pd
from egta.game import Game
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
    num_players = 0
    for profile in raw_data:
        num_players = len(profile)
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


    return Game(strategy_names, profiles, payoffs, num_players)
  

