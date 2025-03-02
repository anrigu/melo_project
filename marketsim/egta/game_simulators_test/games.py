import torch
import random
import numpy

def generate_rps_25_player_data():
    """Generate symmetric Rock Paper Scissors data for 25 players"""
    
    strategies = ["ROCK", "PAPER", "SCISSORS"]
    num_players = 25
    
    # RPS payoff matrix (row beats column)
    # ROCK, PAPER, SCISSORS
    payoff_matrix = [
        [0, -1, 1],  # ROCK
        [1, 0, -1],  # PAPER
        [-1, 1, 0]   # SCISSORS
    ]
    
    # Generate a sample of possible strategy distributions
    # We'll create diverse representative profiles instead of all combinations
    profiles = []
    
    # Add uniform profiles (all same strategy)
    for strat_idx, strat in enumerate(strategies):
        profile = []
        for i in range(num_players):
            player_payoff = 0  # All players have same strategy = all ties
            profile.append([f'agent_{i+1}', strat, player_payoff])
        profiles.append(profile)
    
    distributions = [
    # Original distributions
    (15, 5, 5),    # Heavy ROCK
    (5, 15, 5),    # Heavy PAPER
    (5, 5, 15),    # Heavy SCISSORS
    (10, 10, 5),   # Equal ROCK/PAPER
    (10, 5, 10),   # Equal ROCK/SCISSORS
    (5, 10, 10),   # Equal PAPER/SCISSORS
    (8, 8, 9),     # Near equilibrium 
    (9, 8, 8),     # Near equilibrium
    (8, 9, 8),     # Near equilibrium
    (2, 2, 21),    # Very heavy SCISSORS
    
    # Extreme distributions (one dominant strategy)
    (20, 3, 2),    # Dominant ROCK
    (3, 20, 2),    # Dominant PAPER
    (2, 3, 20),    # Dominant SCISSORS
    (23, 1, 1),    # Almost all ROCK
    (1, 23, 1),    # Almost all PAPER
    (1, 1, 23),    # Almost all SCISSORS
    (25, 0, 0),    # All ROCK
    (0, 25, 0),    # All PAPER
    (0, 0, 25),    # All SCISSORS
    
    # Balanced two-strategy distributions
    (13, 12, 0),   # Only ROCK & PAPER
    (13, 0, 12),   # Only ROCK & SCISSORS
    (0, 13, 12),   # Only PAPER & SCISSORS
    
    # Various mixed distributions
    (9, 8, 8),     # Slight ROCK preference
    (8, 9, 8),     # Slight PAPER preference
    (8, 8, 9),     # Slight SCISSORS preference
    (12, 7, 6),    # Moderate ROCK preference
    (6, 12, 7),    # Moderate PAPER preference
    (7, 6, 12),    # Moderate SCISSORS preference
    
    # Near-equilibrium distributions
    (8, 8, 9),     # Close to uniform
    (9, 8, 8),     # Close to uniform
    (8, 9, 8),     # Close to uniform
    (9, 7, 9),     # Equal R-S, less P
    (7, 9, 9),     # Equal P-S, less R
    (9, 9, 7),     # Equal R-P, less S
    
    # More unusual distributions
    (15, 9, 1),    # Heavy R, some P, minimal S
    (1, 15, 9),    # Heavy P, some S, minimal R
    (9, 1, 15),    # Heavy S, some R, minimal P
    (17, 4, 4),    # Very heavy ROCK
    (4, 17, 4),    # Very heavy PAPER
    (4, 4, 17),    # Very heavy SCISSORS
    (21, 2, 2),    # Extremely heavy ROCK
    (2, 21, 2),    # Extremely heavy PAPER
    (2, 2, 21),    # Extremely heavy SCISSORS
    
    # Prime number distributions
    (11, 7, 7),    # Prime R
    (7, 11, 7),    # Prime P
    (7, 7, 11),    # Prime S
    (13, 5, 7),    # All primes
    (17, 3, 5),    # All primes
    (19, 3, 3),    # Prime R
    
    # Fibonacci-inspired
    (13, 8, 4),    # Fibonacci sequence
    (8, 5, 12),    # Reversed Fibonacci-like
    
    # Other interesting patterns
    (10, 10, 5),   # Equal R-P, half S
    (10, 5, 10),   # Equal R-S, half P
    (5, 10, 10),   # Equal P-S, half R
    (5, 5, 15),    # Equal R-P, triple S
    (5, 15, 5),    # Equal R-S, triple P
    (15, 5, 5),    # Equal P-S, triple R
    (7, 7, 11),    # Equal R-P, more S
    (7, 11, 7),    # Equal R-S, more P
    (11, 7, 7)     # Equal P-S, more R
    ]   
    
    for dist in distributions:
        strat_counts = list(dist)
        profile = []
        
        # Assign strategies to players
        player_strats = []
        for s_idx, count in enumerate(strat_counts):
            player_strats.extend([s_idx] * count)
        
        # Calculate payoffs for each player
        for p_idx in range(num_players):
            strat_idx = player_strats[p_idx]
            strat = strategies[strat_idx]
            
            # Calculate payoff based on other players
            payoff = 0
            for other_idx in range(num_players):
                if other_idx != p_idx:
                    other_strat_idx = player_strats[other_idx]
                    payoff += payoff_matrix[strat_idx][other_strat_idx]
            
            profile.append([f'agent_{p_idx+1}', strat, payoff])
        
        profiles.append(profile)
    
    # Add some random profiles
    for _ in range(10):
        strat_counts = [0, 0, 0]
        for _ in range(num_players):
            strat_counts[random.randint(0, 2)] += 1
        
        profile = []
        player_strats = []
        for s_idx, count in enumerate(strat_counts):
            player_strats.extend([s_idx] * count)
        random.shuffle(player_strats)
        
        for p_idx in range(num_players):
            strat_idx = player_strats[p_idx]
            strat = strategies[strat_idx]
            
            # Calculate payoff based on other players
            payoff = 0
            for other_idx in range(num_players):
                if other_idx != p_idx:
                    other_strat_idx = player_strats[other_idx]
                    payoff += payoff_matrix[strat_idx][other_strat_idx]
            
            profile.append([f'agent_{p_idx+1}', strat, payoff])
        
        profiles.append(profile)
    
    return profiles

def generate_blotto_data(num_players=10, num_battlefields=3, num_troops=5):
    """
    Generate data for a symmetric Colonel Blotto game.
    
    Parameters:
        num_players: Number of players in the game
        num_battlefields: Number of battlefields (typically 3-5)
        num_troops: Total troops each player can allocate
    
    Returns:
        List of game data entries
    """
    import itertools
    import random
    
    # Generate all possible strategies (allocations of troops)
    strategies = []
    for allocation in itertools.combinations_with_replacement(range(num_battlefields), num_troops):
        # Convert to counts per battlefield
        strategy = [0] * num_battlefields
        for bf in allocation:
            strategy[bf] += 1
        strategies.append(tuple(strategy))
    
    # Convert to named strategies
    strategy_names = [f"S{''.join(map(str, s))}" for s in strategies]
    
    # Generate profiles with various strategy distributions
    profiles = []
    
    # Generate a diverse set of configurations
    for _ in range(50):  # Number of profiles to generate
        profile = []
        # Randomly assign strategies to players
        player_strategies = random.choices(range(len(strategies)), k=num_players)
        
        # Calculate payoffs
        for p_idx in range(num_players):
            p_strategy = strategies[player_strategies[p_idx]]
            payoff = 0
            
            # Compare against each other player
            for other_idx in range(num_players):
                if other_idx != p_idx:
                    other_strategy = strategies[player_strategies[other_idx]]
                    # Count battlefields won
                    wins = sum(1 for bf in range(num_battlefields) 
                              if p_strategy[bf] > other_strategy[bf])
                    losses = sum(1 for bf in range(num_battlefields) 
                                if p_strategy[bf] < other_strategy[bf])
                    payoff += wins - losses
            
            profile.append([f'agent_{p_idx+1}', 
                           strategy_names[player_strategies[p_idx]], 
                           payoff])
        
        profiles.append(profile)
    
    return profiles, strategy_names

# Known equilibrium for 3 battlefields, 5 troops:
# Uniform randomization over all strategies that don't put all troops on one battlefield    

def generate_market_entry_game(num_players=25, capacity=12):
    """
    Generate data for a Market Entry Game, where players decide whether to enter
    a market with limited capacity.
    
    Parameters:
        num_players: Number of players
        capacity: Market capacity (how many can profitably enter)
    
    Returns:
        List of game data entries
    """
    import random
    import itertools
    
    strategies = ["Enter", "Stay_Out"]
    profiles = []
    
    # Generate profiles with varying numbers of entrants
    for num_entrants in range(num_players + 1):
        for _ in range(min(5, num_entrants+1)):  # Multiple samples per configuration
            profile = []
            
            # Create a profile with exactly num_entrants players entering
            entrants = random.sample(range(num_players), num_entrants)
            
            for p_idx in range(num_players):
                if p_idx in entrants:
                    # Player enters - payoff depends on how many others enter
                    # Payoff decreases as more players enter beyond capacity
                    if num_entrants <= capacity:
                        payoff = 10  # Full profit when under capacity
                    else:
                        payoff = 10 - 2 * (num_entrants - capacity)  # Declining profit
                    strategy = "Enter"
                else:
                    # Player stays out - gets a fixed outside option
                    payoff = 5  # Outside option
                    strategy = "Stay_Out"
                
                profile.append([f'agent_{p_idx+1}', strategy, payoff])
            
            profiles.append(profile)
    
    return profiles, strategies

# Known equilibrium: Each player enters with probability capacity/num_players