#!/usr/bin/env python3
"""
Test script to verify that the updated quiesce function correctly handles
pure MELO and non-MELO starting points.
"""

import torch
import sys
sys.path.append(".")

from marketsim.egta.core.game import Game
from marketsim.egta.solvers.equilibria import quiesce_sync, format_mixture

def create_test_game():
    """Create a simple test game with MELO and non-MELO strategies."""
    # Create a simple 4x4 game with 2 MELO and 2 non-MELO strategies
    num_strategies = 4
    strategy_names = ["MELO_100_0", "MELO_0_100", "ZI", "AA"]
    
    # Create a simple payoff matrix
    payoff_matrix = torch.tensor([
        [0.5, 0.3, 0.7, 0.2],
        [0.7, 0.6, 0.4, 0.3],
        [0.3, 0.6, 0.5, 0.4],
        [0.8, 0.7, 0.6, 0.5]
    ], dtype=torch.float32)
    
    # Create a mock game object
    class MockGame:
        def __init__(self, payoff_matrix, strategy_names):
            self.payoff_matrix = payoff_matrix
            self.strategy_names = strategy_names
            self.num_strategies = len(strategy_names)
            self.device = payoff_matrix.device
            
        def deviation_payoffs(self, mixture):
            """Calculate deviation payoffs."""
            if not torch.is_tensor(mixture):
                mixture = torch.tensor(mixture, device=self.device)
            return self.payoff_matrix @ mixture
            
        def regret(self, mixture):
            """Calculate regret."""
            dev_payoffs = self.deviation_payoffs(mixture)
            expected_payoff = torch.sum(mixture * dev_payoffs)
            max_payoff = torch.max(dev_payoffs)
            return (max_payoff - expected_payoff).item()
            
        def best_responses(self, mixture, atol=1e-8):
            """Find best responses."""
            dev_payoffs = self.deviation_payoffs(mixture)
            max_payoff = torch.max(dev_payoffs)
            return torch.abs(dev_payoffs - max_payoff) < atol
            
        def restrict(self, restriction):
            """Create restricted game."""
            restricted_matrix = self.payoff_matrix[restriction][:, restriction]
            restricted_names = [self.strategy_names[i] for i in restriction]
            return MockGame(restricted_matrix, restricted_names)
            
        def get_payoff_matrix(self):
            """Get payoff matrix."""
            return self.payoff_matrix
    
    mock_game = MockGame(payoff_matrix, strategy_names)
    
    # Wrap in Game class
    game = Game(mock_game)
    game.strategy_names = strategy_names
    
    return game

def main():
    print("Testing updated quiesce function with MELO/non-MELO starting points...")
    
    # Create test game
    game = create_test_game()
    print(f"\nCreated test game with strategies: {game.strategy_names}")
    
    # Run quiesce with verbose output
    print("\nRunning quiesce...")
    equilibria = quiesce_sync(
        game=game,
        num_iters=10,
        num_random_starts=5,
        regret_threshold=1e-3,
        dist_threshold=0.05,
        restricted_game_size=4,
        solver="replicator",
        solver_iters=1000,
        verbose=True
    )
    
    print(f"\nFound {len(equilibria)} equilibria:")
    for i, (mixture, regret) in enumerate(equilibria):
        print(f"\nEquilibrium {i+1}:")
        print(f"  Mixture: {format_mixture(mixture, game.strategy_names)}")
        print(f"  Regret: {regret:.6f}")
        
        # Check if it's MELO-heavy or non-MELO-heavy
        melo_weight = mixture[0].item() + mixture[1].item()
        non_melo_weight = mixture[2].item() + mixture[3].item()
        
        if melo_weight > 0.8:
            print("  Type: MELO-dominant")
        elif non_melo_weight > 0.8:
            print("  Type: Non-MELO-dominant")
        else:
            print("  Type: Mixed")

if __name__ == "__main__":
    main() 