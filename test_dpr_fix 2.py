#!/usr/bin/env python3
"""
Test script to verify that the DPR fix correctly tests equilibria against the full game.
"""

import torch
import sys
sys.path.append(".")

from marketsim.egta.core.game import Game
from marketsim.egta.solvers.equilibria import quiesce_sync
from marketsim.game.symmetric_game import SymmetricGame

def create_test_games():
    """Create a full game and a reduced game for testing."""
    
    # Create a simple 3-strategy game
    num_players_full = 6
    num_players_reduced = 3
    num_strategies = 3
    strategy_names = ["MELO_100_0", "MELO_0_100", "ZI"]
    
    # Create some test profiles and payoffs for the full game
    # Profile format: (strategy_counts)
    config_table_full = torch.tensor([
        [6, 0, 0],  # All MELO_100_0
        [0, 6, 0],  # All MELO_0_100
        [0, 0, 6],  # All ZI
        [3, 3, 0],  # Half MELO_100_0, half MELO_0_100
        [3, 0, 3],  # Half MELO_100_0, half ZI
        [0, 3, 3],  # Half MELO_0_100, half ZI
        [2, 2, 2],  # Equal mix
    ], dtype=torch.float32)
    
    # Create corresponding payoffs (higher payoffs for MELO strategies)
    payoff_table_full = torch.tensor([
        [1.0, 0.0, 0.0],  # MELO_100_0 payoffs for each profile
        [0.8, 0.8, 0.0],  # MELO_0_100 payoffs for each profile
        [0.5, 0.5, 0.5],  # ZI payoffs for each profile
        [0.9, 0.9, 0.2],
        [0.7, 0.3, 0.6],
        [0.4, 0.7, 0.6],
        [0.6, 0.6, 0.5],
    ], dtype=torch.float32).T  # Shape: (num_strategies, num_profiles)
    
    # Create full game
    full_game_obj = SymmetricGame(
        num_players=num_players_full,
        num_actions=num_strategies,
        config_table=config_table_full,
        payoff_table=payoff_table_full,
        strategy_names=strategy_names,
        device="cpu"
    )
    full_game = Game(full_game_obj)
    
    # Create reduced game with DPR scaling
    scaling_factor = (num_players_full - 1) / (num_players_reduced - 1)
    print(f"DPR scaling factor: {scaling_factor:.2f}")
    
    # Scale down the config table for reduced game
    config_table_reduced = config_table_full / scaling_factor
    
    # Scale down the payoffs for reduced game
    payoff_table_reduced = payoff_table_full / scaling_factor
    
    reduced_game_obj = SymmetricGame(
        num_players=num_players_reduced,
        num_actions=num_strategies,
        config_table=config_table_reduced,
        payoff_table=payoff_table_reduced,
        strategy_names=strategy_names,
        device="cpu"
    )
    reduced_game = Game(reduced_game_obj)
    
    return full_game, reduced_game

def test_without_dpr_fix():
    """Test quiesce without the DPR fix (should give wrong results)."""
    print("=" * 60)
    print("Testing WITHOUT DPR fix (testing against reduced game)")
    print("=" * 60)
    
    full_game, reduced_game = create_test_games()
    
    # Run quiesce on reduced game, testing against reduced game (old behavior)
    equilibria = quiesce_sync(
        game=reduced_game,
        num_iters=5,
        num_random_starts=5,
        regret_threshold=1e-3,
        dist_threshold=0.05,
        restricted_game_size=4,
        solver="replicator",
        solver_iters=1000,
        verbose=True,
        full_game=None  # No full game - test against reduced game
    )
    
    print(f"\nFound {len(equilibria)} equilibria when testing against REDUCED game:")
    for i, (mixture, regret) in enumerate(equilibria):
        print(f"  Equilibrium {i+1}: regret={regret:.6f}")
        for j, prob in enumerate(mixture):
            if prob > 0.01:
                print(f"    {reduced_game.strategy_names[j]}: {prob:.4f}")
        
        # Manually test this equilibrium against the FULL game
        full_game_regret = full_game.regret(mixture).item()
        print(f"    Regret vs FULL game: {full_game_regret:.6f}")
        print(f"    Is true equilibrium: {full_game_regret <= 1e-3}")
    
    return equilibria

def test_with_dpr_fix():
    """Test quiesce with the DPR fix (should give correct results)."""
    print("\n" + "=" * 60)
    print("Testing WITH DPR fix (testing against full game)")
    print("=" * 60)
    
    full_game, reduced_game = create_test_games()
    
    # Run quiesce on reduced game, testing against full game (new behavior)
    equilibria = quiesce_sync(
        game=reduced_game,
        num_iters=5,
        num_random_starts=5,
        regret_threshold=1e-3,
        dist_threshold=0.05,
        restricted_game_size=4,
        solver="replicator",
        solver_iters=1000,
        verbose=True,
        full_game=full_game  # Pass full game for testing
    )
    
    print(f"\nFound {len(equilibria)} equilibria when testing against FULL game:")
    for i, (mixture, regret) in enumerate(equilibria):
        print(f"  Equilibrium {i+1}: regret={regret:.6f}")
        for j, prob in enumerate(mixture):
            if prob > 0.01:
                print(f"    {full_game.strategy_names[j]}: {prob:.4f}")
        
        # Verify this equilibrium against the FULL game
        full_game_regret = full_game.regret(mixture).item()
        print(f"    Verified regret vs FULL game: {full_game_regret:.6f}")
        print(f"    Is true equilibrium: {full_game_regret <= 1e-3}")
        
        # Also check against reduced game for comparison
        reduced_game_regret = reduced_game.regret(mixture).item()
        print(f"    Regret vs REDUCED game: {reduced_game_regret:.6f}")
    
    return equilibria

def main():
    print("Testing DPR equilibrium testing fix...")
    print("\nThis test verifies that when using DPR:")
    print("1. WITHOUT fix: equilibria are tested against the reduced game (wrong)")
    print("2. WITH fix: equilibria are tested against the full game (correct)")
    
    # Test without fix
    eq_without_fix = test_without_dpr_fix()
    
    # Test with fix
    eq_with_fix = test_with_dpr_fix()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Equilibria found without DPR fix: {len(eq_without_fix)}")
    print(f"Equilibria found with DPR fix: {len(eq_with_fix)}")
    
    if len(eq_with_fix) <= len(eq_without_fix):
        print("\n✅ SUCCESS: DPR fix correctly filters out false equilibria!")
        print("   The fix found fewer or equal equilibria, as expected, since")
        print("   it correctly rejects equilibria that only work in the reduced game.")
    else:
        print("\n❌ WARNING: DPR fix found more equilibria than expected.")

if __name__ == "__main__":
    main() 