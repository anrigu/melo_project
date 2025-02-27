import itertools
import numpy as np
from collections import defaultdict
import pandas as pd
from egta.game import AbstractGame


import torch
import numpy as np
from collections import defaultdict
from egta.symmetric_game import SymmetricGame
from egta.utils.log_multimodal import logmultinomial



def process_game_data(raw_data): 
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
    print(strategy_names)

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
    
    #print profiles and payoffs and strategy names
    for profile, payoff, strategy in zip(profiles, payoffs, strategy_names):
        print(profile, payoff, strategy)

    return AbstractGame(strategy_names, profiles, payoffs, num_players)



def create_symmetric_game_from_data(raw_data, device="cpu"):
    """
    Create a SymmetricGame from raw EGTA simulation data.
    
    Parameters:
        raw_data : list of lists
            Each sublist represents a strategy profile containing agent-level data.
            Format: [(player_id, strategy_name, payoff), ...]
        device : str
            PyTorch device to use ("cpu" or "cuda")
    
    Returns:
        SymmetricGame object with efficiently computed expected payoffs.
    """
    # Extract strategy names and number of players
    strategy_names = set()
    num_players = 0
    for profile in raw_data:
        num_players = len(profile)
        for _, strategy, _ in profile:
            strategy_names.add(strategy)
    
    strategy_names = sorted(list(strategy_names))
    num_actions = len(strategy_names)
    
    # Create mapping from strategy names to indices
    strategy_to_index = {name: i for i, name in enumerate(strategy_names)}
    
    # Aggregate data by profile
    profile_dict = defaultdict(lambda: {"count": 0, "payoffs": defaultdict(list)})
    
    for profile in raw_data:
        # Convert profile to counts of each strategy
        strat_counts = [0] * num_actions
        for _, strategy, _ in profile:
            strat_idx = strategy_to_index[strategy]
            strat_counts[strat_idx] += 1
        
        # Use tuple for dictionary key
        strat_counts_tuple = tuple(strat_counts)
        
        # Increment profile count and collect payoffs
        profile_dict[strat_counts_tuple]["count"] += 1
        for _, strategy, payoff in profile:
            profile_dict[strat_counts_tuple]["payoffs"][strategy].append(payoff)
    
    # Check for repeat profiles (same observations multiple times)
    repeat_profiles = [profile for profile, data in profile_dict.items() if data["count"] > 1]
    if repeat_profiles:
        print(f"Found {len(repeat_profiles)} repeat profiles in data")
    
    # Create config_table and calculate payoffs
    configs = list(profile_dict.keys())
    num_configs = len(configs)
    
    # Initialize arrays
    config_table = np.zeros((num_configs, num_actions))
    raw_payoff_table = np.zeros((num_actions, num_configs))
    
    # Fill the tables
    for c, config in enumerate(configs):
        # Set the configuration counts
        config_table[c] = config
        
        # Calculate expected payoffs for each strategy
        for strat_idx, strat_name in enumerate(strategy_names):
            if config[strat_idx] > 0:  # Only if strategy was used
                payoffs = profile_dict[config]["payoffs"][strat_name]
                if payoffs:
                    raw_payoff_table[strat_idx, c] = np.mean(payoffs)
    
    # Convert to tensors for efficiency
    config_table = torch.tensor(config_table, dtype=torch.float32, device=device)
    raw_payoff_table = torch.tensor(raw_payoff_table, dtype=torch.float32, device=device)
    
    # Calculate log multinomial coefficients for each profile
    log_coefficients = torch.zeros(num_configs, device=device)
    for c, config in enumerate(configs):
        log_coefficients[c] = logmultinomial(*config)
    
    # Set scaling parameters to normalize payoffs
    min_payoff = raw_payoff_table[raw_payoff_table > 0].min().item() if torch.any(raw_payoff_table > 0) else 1e-5
    max_payoff = raw_payoff_table.max().item() if torch.any(raw_payoff_table > 0) else 1.0
    
    # Special case handling
    if min_payoff == max_payoff:
        offset = 0.0
        scale = 1.0
    else:
        # Calculate offset and scale using Bryce's method
        offset = min_payoff - (max_payoff - min_payoff) * 1e-5 / (1.0 - 1e-5)
        scale = (max_payoff - min_payoff) / (1.0 - 1e-5)
    
    # Normalize payoffs
    normalized_payoffs = (raw_payoff_table - offset) / scale
    
    # Clamp to ensure numerical stability (avoid zeros or negatives)
    normalized_payoffs = torch.clamp(normalized_payoffs, min=1e-5, max=1.0)
    
    # Convert to log space and add log coefficients
    weighted_log_payoffs = torch.log(normalized_payoffs)
    
    # Create the SymmetricGame instance
    game = SymmetricGame(
        num_players=num_players,
        num_actions=num_actions,
        config_table=config_table,
        payoff_table=weighted_log_payoffs,
        offset=offset,
        scale=scale,
        strategy_names=strategy_names,
        device=device
    )
    
    return game
