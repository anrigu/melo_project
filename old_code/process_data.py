import itertools
import numpy as np
from collections import defaultdict
import pandas as pd
from game import AbstractGame


import torch
import numpy as np
from collections import defaultdict
from symmetric_game import SymmetricGame
from utils.log_multimodal import logmultinomial






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
        print(profile)
        for _, strategy, _ in profile:
            strategy_names.add(strategy)
    
    strategy_names = sorted(list(strategy_names))
    num_actions = len(strategy_names)
    
    # Create mapping from strategy names to indices
    strategy_to_index = {name: i for i, name in enumerate(strategy_names)}
    
    # Aggregate data by profile
    profile_dict = defaultdict(lambda: {"count": 0, "payoffs": [[] for _ in range(num_actions)]})

    for profile in raw_data:
        # Convert profile to counts of each strategy
        strat_counts = [0] * num_actions
        for _, strategy, _ in profile:
            strat_idx = strategy_to_index[strategy]
            strat_counts[strat_idx] += 1
        
        # Use tuple for dictionary key
        strat_counts_tuple = tuple(strat_counts)
        
        # Increment profile count and collect payoffs by strategy index
        profile_dict[strat_counts_tuple]["count"] += 1
        for _, strategy, payoff in profile:
            strat_idx = strategy_to_index[strategy]
            profile_dict[strat_counts_tuple]["payoffs"][strat_idx].append(float(payoff))
        
    # Check for repeat profiles
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
        for strat_idx in range(num_actions):
            if config[strat_idx] > 0:  # Only if strategy was used
                payoffs = profile_dict[config]["payoffs"][strat_idx]
                if payoffs:
                    # Make sure we get all payoffs for all players using this strategy
                    raw_payoff_table[strat_idx, c] = np.mean(payoffs)
                    
                    # Debug output to verify correct averaging
                    print(f"Profile {config}: Strategy {strategy_names[strat_idx]} has {len(payoffs)} payoffs with mean {raw_payoff_table[strat_idx, c]:.4f}")
    
    # Print raw payoff table for debugging
    print("Raw payoff table:")
    for c, config in enumerate(configs):
        payoffs_str = ", ".join([f"{strategy_names[i]}: {raw_payoff_table[i, c]:.2f}" for i in range(num_actions) if config[i] > 0])
        config_str = ", ".join([f"{strategy_names[i]}: {config[i]}" for i in range(num_actions) if config[i] > 0])
        print(f"Config {c+1}: [{config_str}] → [{payoffs_str}]")
    
    # Convert to tensors
    config_table = torch.tensor(config_table, dtype=torch.float32, device=device)
    raw_payoff_table = torch.tensor(raw_payoff_table, dtype=torch.float32, device=device)
    
    # Simple normalization for RPS-like games
    min_payoff = raw_payoff_table.min().item()  
    max_payoff = raw_payoff_table.max().item()
    
    if min_payoff == max_payoff:
        offset = min_payoff
        scale = 1.0
    else:
        offset = min_payoff
        scale = max_payoff - min_payoff
    
    # Normalize to [0, 1] range
    normalized_payoffs = (raw_payoff_table - offset) / scale
    
    # Epsilon to avoid log(0)
    epsilon = 1e-6
    normalized_payoffs = torch.clamp(normalized_payoffs, min=epsilon, max=1.0)
    
    # Convert to log space
    log_payoffs = torch.log(normalized_payoffs)
    
    # Create the SymmetricGame instance
    game = SymmetricGame(
        num_players=num_players,
        num_actions=num_actions,
        config_table=config_table,
        payoff_table=log_payoffs,
        offset=offset,
        scale=scale,
        strategy_names=strategy_names,
        device=device
    )
    
    return game