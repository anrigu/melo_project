"""
Test script for the improved QUIESCE implementation.

This script creates a simple game and runs the new QUIESCE algorithm to find equilibria.
"""
import os
import sys
import torch
import numpy as np
from typing import List, Tuple

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from marketsim.game.symmetric_game import SymmetricGame
from marketsim.egta.core.game import Game
from marketsim.egta.solvers.equilibria import quiesce_sync, replicator_dynamics, regret


def create_rock_paper_scissors_game():
    """
    Create a standard Rock-Paper-Scissors game.
    This game has a unique mixed strategy equilibrium where each strategy is played with equal probability.
    """
    num_players = 2
    num_actions = 3
    strategy_names = ["Rock", "Paper", "Scissors"]
    
    # Create a direct payoff matrix for Rock-Paper-Scissors
    # Payoff matrix is:
    # [0, -1, 1]
    # [1, 0, -1]
    # [-1, 1, 0]
    # Where rows are player strategy, columns are opponent strategy
    payoff_matrix = np.array([
        [0, -1, 1],
        [1, 0, -1],
        [-1, 1, 0]
    ])
    
    # Define a payoff function using the payoff matrix
    def payoff_function(config):
        # For a 2-player game, we need to determine which strategy pair is being used
        # When asked for the payoff of strategy s, we need to consider all opponent's strategies
        
        # The strategy for which we're calculating payoff
        player_strategy = np.argmax(config)
        
        # Compute the expected payoff against the opponent strategy distribution
        opponent_config = config.copy()
        opponent_config[player_strategy] -= 1  # Remove the player
        
        # If all opponents are using the same strategy
        if np.sum(opponent_config) == 1:
            opponent_strategy = np.argmax(opponent_config)
            return payoff_matrix[player_strategy, opponent_strategy]
        else:
            # This shouldn't happen in our 2-player game test
            return 0
    
    # Create the symmetric game
    sym_game = SymmetricGame.from_payoff_function(
        num_players=num_players,
        num_actions=num_actions,
        payoff_function=payoff_function,
        strategy_names=strategy_names,
        device="cpu"
    )
    
    # Create a Game wrapper
    game = Game(sym_game, metadata={"name": "Rock-Paper-Scissors"})
    
    return game


def create_prisoners_dilemma_game():
    """
    Create a Prisoner's Dilemma game.
    This game has a unique pure strategy equilibrium (Defect, Defect).
    """
    num_players = 2
    num_actions = 2
    strategy_names = ["Cooperate", "Defect"]
    
    # Create a direct payoff matrix for Prisoner's Dilemma
    # Payoff matrix is:
    # [3, 0]
    # [5, 1]
    # Where rows are player strategy, columns are opponent strategy
    # Cooperate=0, Defect=1
    payoff_matrix = np.array([
        [3, 0],  # Cooperate payoffs
        [5, 1]   # Defect payoffs
    ])
    
    # Define a payoff function using the payoff matrix
    def payoff_function(config):
        # The strategy for which we're calculating payoff
        player_strategy = np.argmax(config)
        
        # Compute the expected payoff against the opponent strategy distribution
        opponent_config = config.copy()
        opponent_config[player_strategy] -= 1  # Remove the player
        
        # If all opponents are using the same strategy
        if np.sum(opponent_config) == 1:
            opponent_strategy = np.argmax(opponent_config)
            return payoff_matrix[player_strategy, opponent_strategy]
        else:
            # This shouldn't happen in our 2-player game test
            return 0
    
    # Create the symmetric game
    sym_game = SymmetricGame.from_payoff_function(
        num_players=num_players,
        num_actions=num_actions,
        payoff_function=payoff_function,
        strategy_names=strategy_names,
        device="cpu"
    )
    
    # Create a Game wrapper
    game = Game(sym_game, metadata={"name": "Prisoner's Dilemma"})
    
    return game


def create_stag_hunt_game():
    """
    Create a Stag Hunt game.
    This game has two pure strategy equilibria (Stag, Stag) and (Hare, Hare).
    It also has a mixed strategy equilibrium.
    """
    num_players = 2
    num_actions = 2
    strategy_names = ["Stag", "Hare"]
    
    # Create a direct payoff matrix for Stag Hunt
    # Payoff matrix is:
    # [4, 0]
    # [2, 2]
    # Where rows are player strategy, columns are opponent strategy
    # Stag=0, Hare=1
    payoff_matrix = np.array([
        [4, 0],  # Stag payoffs
        [2, 2]   # Hare payoffs
    ])
    
    # Define a payoff function using the payoff matrix
    def payoff_function(config):
        # The strategy for which we're calculating payoff
        player_strategy = np.argmax(config)
        
        # Compute the expected payoff against the opponent strategy distribution
        opponent_config = config.copy()
        opponent_config[player_strategy] -= 1  # Remove the player
        
        # If all opponents are using the same strategy
        if np.sum(opponent_config) == 1:
            opponent_strategy = np.argmax(opponent_config)
            return payoff_matrix[player_strategy, opponent_strategy]
        else:
            # This shouldn't happen in our 2-player game test
            return 0
    
    # Create the symmetric game
    sym_game = SymmetricGame.from_payoff_function(
        num_players=num_players,
        num_actions=num_actions,
        payoff_function=payoff_function,
        strategy_names=strategy_names,
        device="cpu"
    )
    
    # Create a Game wrapper
    game = Game(sym_game, metadata={"name": "Stag Hunt"})
    
    return game


def main():
    print("Testing improved QUIESCE implementation with classic games")
    print("=" * 70)
    
    # Test with Rock-Paper-Scissors
    print("\nGame 1: Rock-Paper-Scissors")
    print("-" * 40)
    rps_game = create_rock_paper_scissors_game()
    
    # Print the payoff matrix
    print("Payoff Matrix:")
    payoff_matrix = np.array([
        [0, -1, 1],
        [1, 0, -1],
        [-1, 1, 0]
    ])
    for i, row in enumerate(payoff_matrix):
        print(f"{rps_game.strategy_names[i]}: {row}")
    
    # Test the uniform mixture directly (should be an equilibrium in RPS)
    print("\nTesting uniform mixture (should be an equilibrium):")
    uniform_mix = torch.ones(rps_game.num_strategies, device=rps_game.game.device) / rps_game.num_strategies
    uniform_regret = rps_game.regret(uniform_mix)
    print(f"Uniform mixture regret: {uniform_regret.item():.6f}")
    
    print("\nFinding equilibria...")
    rps_eq = modified_quiesce_sync(
        game=rps_game,
        num_iters=5,
        regret_threshold=1e-4,
        dist_threshold=1e-3,
        restricted_game_size=3,
        solver='replicator',
        solver_iters=1000,
        verbose=True
    )
    
    print("\nSummary of equilibria found:")
    for i, (eq_mix, eq_regret) in enumerate(rps_eq):
        strat_str = ", ".join([
            f"{rps_game.strategy_names[s]}: {eq_mix[s].item():.4f}" 
            for s in range(rps_game.num_strategies)
            if eq_mix[s].item() > 0.01
        ])
        print(f"Equilibrium {i+1}: regret={eq_regret:.6f}, {strat_str}")
    
    # Test with Prisoner's Dilemma
    print("\nGame 2: Prisoner's Dilemma")
    print("-" * 40)
    pd_game = create_prisoners_dilemma_game()
    print("Finding equilibria...")
    pd_eq = modified_quiesce_sync(
        game=pd_game,
        num_iters=5,
        regret_threshold=1e-4,
        dist_threshold=1e-3,
        restricted_game_size=2,
        solver='replicator',
        solver_iters=1000,
        verbose=True
    )
    
    print("\nSummary of equilibria found:")
    for i, (eq_mix, eq_regret) in enumerate(pd_eq):
        strat_str = ", ".join([
            f"{pd_game.strategy_names[s]}: {eq_mix[s].item():.4f}" 
            for s in range(pd_game.num_strategies)
            if eq_mix[s].item() > 0.01
        ])
        print(f"Equilibrium {i+1}: regret={eq_regret:.6f}, {strat_str}")
    
    # Test with Stag Hunt
    print("\nGame 3: Stag Hunt")
    print("-" * 40)
    sh_game = create_stag_hunt_game()
    print("Finding equilibria...")
    sh_eq = modified_quiesce_sync(
        game=sh_game,
        num_iters=5,
        regret_threshold=1e-4,
        dist_threshold=1e-3,
        restricted_game_size=2,
        solver='replicator',
        solver_iters=1000,
        verbose=True
    )
    
    print("\nSummary of equilibria found:")
    for i, (eq_mix, eq_regret) in enumerate(sh_eq):
        strat_str = ", ".join([
            f"{sh_game.strategy_names[s]}: {eq_mix[s].item():.4f}" 
            for s in range(sh_game.num_strategies)
            if eq_mix[s].item() > 0.01
        ])
        print(f"Equilibrium {i+1}: regret={eq_regret:.6f}, {strat_str}")


# Add this function to manually add the uniform mixed strategy to the initial candidates
def modified_quiesce_sync(
    game: Game, 
    num_iters: int = 10,
    regret_threshold: float = 1e-4,
    dist_threshold: float = 1e-3,
    restricted_game_size: int = 4,
    solver: str = 'replicator',
    solver_iters: int = 1000,
    verbose: bool = True
) -> List[Tuple[torch.Tensor, float]]:
    """
    Modified version of quiesce_sync that adds uniform mixture as an initial candidate.
    """
    from marketsim.egta.solvers.equilibria import quiesce_sync
    
    # First, create a uniform mixture
    uniform_mix = torch.ones(game.num_strategies, device=game.game.device) / game.num_strategies
    
    # Find equilibrium starting from uniform mixture
    uniform_eq = replicator_dynamics(game, uniform_mix, iters=solver_iters)
    uniform_regret = game.regret(uniform_eq)
    
    # Run standard QUIESCE
    equilibria = quiesce_sync(
        game=game,
        num_iters=num_iters,
        regret_threshold=regret_threshold,
        dist_threshold=dist_threshold,
        restricted_game_size=restricted_game_size,
        solver=solver,
        solver_iters=solver_iters,
        verbose=verbose
    )
    
    # If the uniform mixture produces an equilibrium, add it
    if uniform_regret < regret_threshold:
        is_duplicate = False
        for mix, _ in equilibria:
            if torch.max(torch.abs(mix - uniform_eq)) < dist_threshold:
                is_duplicate = True
                break
        
        if not is_duplicate:
            equilibria.append((uniform_eq, uniform_regret.item()))
            
            if verbose:
                print(f"Added equilibrium from uniform mixture with regret {uniform_regret.item():.6f}")
    
    # Sort by regret
    equilibria.sort(key=lambda x: x[1])
    
    return equilibria


if __name__ == "__main__":
    main() 