import torch
import numpy as np
from marketsim.egta.core.game import Game

def main():
    # Create a simple example to simulate payoff matrix computation
    strategies = ['MELO_100_0', 'MELO_0_100']
    device = 'cpu'

    # Create mock payoff data that would favor MELO (strategy 1)
    # Format: [(player_id, strategy, payoff), ...]
    # Four players with 2 strategies
    profiles = [
        # All playing strategy 0
        [(0, 'MELO_100_0', 100), (1, 'MELO_100_0', 100), (2, 'MELO_100_0', 100), (3, 'MELO_100_0', 100)],
        
        # 3 playing strategy 0, 1 playing strategy 1
        [(0, 'MELO_100_0', 90), (1, 'MELO_100_0', 90), (2, 'MELO_100_0', 90), (3, 'MELO_0_100', 120)],
        
        # 2 playing each strategy
        [(0, 'MELO_100_0', 80), (1, 'MELO_100_0', 80), (2, 'MELO_0_100', 110), (3, 'MELO_0_100', 110)],
        
        # 1 playing strategy 0, 3 playing strategy 1
        [(0, 'MELO_100_0', 70), (1, 'MELO_0_100', 100), (2, 'MELO_0_100', 100), (3, 'MELO_0_100', 100)],
        
        # All playing strategy 1
        [(0, 'MELO_0_100', 90), (1, 'MELO_0_100', 90), (2, 'MELO_0_100', 90), (3, 'MELO_0_100', 90)]
    ]

    # Create the game
    game = Game.from_payoff_data(profiles, strategy_names=strategies, device=device)

    # Get the payoff matrix
    payoff_matrix = game.get_payoff_matrix()
    print('\nPayoff Matrix:')
    for i in range(2):
        print(f'  {strategies[i]}: [{payoff_matrix[i, 0].item():.4f}, {payoff_matrix[i, 1].item():.4f}]')

    # Check pure equilibria
    print('\nChecking pure equilibria:')
    print(f'  Is {strategies[0]} a pure equilibrium? {payoff_matrix[0, 0] >= payoff_matrix[1, 0]}')
    print(f'    Regret: {max(0, float(payoff_matrix[1, 0] - payoff_matrix[0, 0])):.4f}')
    print(f'  Is {strategies[1]} a pure equilibrium? {payoff_matrix[1, 1] >= payoff_matrix[0, 1]}')
    print(f'    Regret: {max(0, float(payoff_matrix[0, 1] - payoff_matrix[1, 1])):.4f}')

    # Check mixed equilibrium condition
    denom = payoff_matrix[0, 0] - payoff_matrix[1, 0] - payoff_matrix[0, 1] + payoff_matrix[1, 1]
    print(f'\nMixed equilibrium denominator: {denom.item():.4f}')
    if abs(denom) > 1e-10:
        p = (payoff_matrix[1, 1] - payoff_matrix[0, 1]) / denom
        if 0 < p < 1:
            mixed = torch.tensor([p, 1-p], device=device)
            print(f'Mixed equilibrium: {strategies[0]}:{p.item():.4f}, {strategies[1]}:{1-p.item():.4f}')
        else:
            print(f'No valid mixed equilibrium (p={p.item():.4f})')
    else:
        print('No mixed equilibrium (denominator too small)')
    
    # Now let's test a reversed case that would favor CDA
    print("\n--- Reversed Payoffs (favoring CDA) ---")
    
    profiles_reversed = [
        [(0, 'MELO_100_0', 90), (1, 'MELO_100_0', 90), (2, 'MELO_100_0', 90), (3, 'MELO_100_0', 90)],
        
        # 3 playing strategy 0, 1 playing strategy 1
        [(0, 'MELO_100_0', 100), (1, 'MELO_100_0', 100), (2, 'MELO_100_0', 100), (3, 'MELO_0_100', 70)],
        
        # 2 playing each strategy
        [(0, 'MELO_100_0', 110), (1, 'MELO_100_0', 110), (2, 'MELO_0_100', 80), (3, 'MELO_0_100', 80)],
        
        [(0, 'MELO_100_0', 120), (1, 'MELO_0_100', 90), (2, 'MELO_0_100', 90), (3, 'MELO_0_100', 90)],
        
        # All playing strategy 1
        [(0, 'MELO_0_100', 100), (1, 'MELO_0_100', 100), (2, 'MELO_0_100', 100), (3, 'MELO_0_100', 100)]
    ]
    
    game_reversed = Game.from_payoff_data(profiles_reversed, strategy_names=strategies, device=device)

    # Get the payoff matrix
    payoff_matrix_rev = game_reversed.get_payoff_matrix()
    print('\nPayoff Matrix (Reversed):')
    for i in range(2):
        print(f'  {strategies[i]}: [{payoff_matrix_rev[i, 0].item():.4f}, {payoff_matrix_rev[i, 1].item():.4f}]')

    print('\nChecking pure equilibria:')
    print(f'  Is {strategies[0]} a pure equilibrium? {payoff_matrix_rev[0, 0] >= payoff_matrix_rev[1, 0]}')
    print(f'    Regret: {max(0, float(payoff_matrix_rev[1, 0] - payoff_matrix_rev[0, 0])):.4f}')
    print(f'  Is {strategies[1]} a pure equilibrium? {payoff_matrix_rev[1, 1] >= payoff_matrix_rev[0, 1]}')
    print(f'    Regret: {max(0, float(payoff_matrix_rev[0, 1] - payoff_matrix_rev[1, 1])):.4f}')

if __name__ == "__main__":
    main() 