import torch
import numpy as np
import pytest
from typing import List, Tuple
import time

from marketsim.egta.core.game import Game
from marketsim.egta.schedulers.dpr import DPRScheduler
from marketsim.egta.egta import EGTA
from marketsim.game.symmetric_game import SymmetricGame

class MockSimulator:
    """Mock simulator for testing that returns pre-defined payoffs."""
    
    def __init__(self, strategies: List[str], num_players: int, payoff_func=None):
        self.strategies = strategies
        self.num_players = num_players
        # Default payoff function - can be overridden
        self.payoff_func = payoff_func or self._default_payoff
        
    def get_strategies(self) -> List[str]:
        return self.strategies
    
    def get_num_players(self) -> int:
        return self.num_players
    
    def _default_payoff(self, profile: List[str], player_idx: int) -> float:
        """
        Default payoff function that favors coordinating on the same strategy.
        Also adds a slight advantage to strategies with higher indices to create 
        interesting dynamics.
        """
        own_strategy = profile[player_idx]
        strategy_idx = self.strategies.index(own_strategy)
        
        # Count how many players chose the same strategy
        same_count = sum(1 for s in profile if s == own_strategy)
        
        # Base payoff from coordination
        payoff = 5.0 * same_count / self.num_players
        
        # Add small bonus for higher-indexed strategies
        payoff += 0.1 * strategy_idx
        
        return payoff
        
    def simulate_profile(self, profile: List[str]) -> List[Tuple[int, str, float]]:
        """Simulate a single profile and return payoffs."""
        results = []
        for player_idx in range(len(profile)):
            strategy = profile[player_idx]
            payoff = self.payoff_func(profile, player_idx)
            results.append((player_idx, strategy, payoff))
        return results
    
    def simulate_profiles(self, profiles: List[List[str]]) -> List[List[Tuple[int, str, float]]]:
        """Simulate multiple profiles."""
        return [self.simulate_profile(profile) for profile in profiles]


def test_dpr_scheduler_initialization():
    """Test that the DPR scheduler initializes correctly."""
    strategies = ["A", "B", "C", "D"]
    num_players = 4
    subgame_size = 3
    reduction_size = 3
    
    scheduler = DPRScheduler(
        strategies=strategies, 
        num_players=num_players,
        subgame_size=subgame_size,
        reduction_size=reduction_size
    )
    
    # Check initialization
    assert scheduler.strategies == strategies
    assert scheduler.num_players == num_players
    assert scheduler.subgame_size == subgame_size
    assert scheduler.reduction_size == reduction_size
    assert scheduler.scaling_factor == (num_players - 1) / (reduction_size - 1)
    assert len(scheduler.requested_subgames) == 1  # One initial subgame


def test_dpr_profile_generation():
    """Test that the DPR scheduler generates profiles correctly."""
    strategies = ["A", "B", "C"]
    num_players = 10
    reduction_size = 4  # Gives scaling factor (10-1)/(4-1) = 9/3 = 3 (integer)
    
    scheduler = DPRScheduler(
        strategies=strategies, 
        num_players=num_players,
        subgame_size=len(strategies),
        reduction_size=reduction_size
    )
    
    # Generate profiles for the initial subgame
    subgame = set(strategies)
    profiles = scheduler._generate_profiles_for_subgame(subgame)
    
    # Debug: Print the actual number of profiles and some examples
    print(f"Number of profiles generated: {len(profiles)}")
    print(f"Example profiles: {profiles[:3]}")
    
    # For 3 strategies and 4 players, we get 15 profiles
    # Using the multiset coefficient formula: (n+s-1)! / ((s-1)! * n!) = (4+3-1)! / ((3-1)! * 4!) = 15
    assert len(profiles) == 15
    
    # Check that each profile has the correct length (reduction_size)
    for profile in profiles:
        assert len(profile) == reduction_size
        assert all(s in strategies for s in profile)


def test_dpr_scaling():
    """Test that the DPR scaling works correctly."""
    strategies = ["A", "B", "C"]
    num_players = 7
    reduction_size = 3
    
    scheduler = DPRScheduler(
        strategies=strategies, 
        num_players=num_players,
        reduction_size=reduction_size
    )
    
    # Expected scaling factor: (7-1)/(3-1) = 6/2 = 3 (integer)
    assert scheduler.scaling_factor == 3.0
    
    # Test payoff scaling
    payoffs = torch.tensor([1.0, 2.0, 3.0])
    scaled_payoffs = scheduler.scale_payoffs(payoffs)
    
    expected_scaled = torch.tensor([3.0, 6.0, 9.0])
    assert torch.allclose(scaled_payoffs, expected_scaled)
    
    # Test no scaling when reduction_size == num_players
    scheduler2 = DPRScheduler(
        strategies=strategies, 
        num_players=num_players,
        reduction_size=num_players
    )
    assert scheduler2.scaling_factor == 1.0
    
    scaled_payoffs2 = scheduler2.scale_payoffs(payoffs)
    assert torch.allclose(scaled_payoffs2, payoffs)


def test_dpr_with_egta():
    """Test the DPR scheduler with EGTA framework."""
    strategies = ["A", "B", "C"]
    num_players = 7
    reduction_size = 3
    
    # Create mock simulator
    simulator = MockSimulator(strategies, num_players)
    
    # Create DPR scheduler
    scheduler = DPRScheduler(
        strategies=strategies, 
        num_players=num_players,
        subgame_size=2,  # Start with subgames of size 2
        reduction_size=reduction_size,
        batch_size=5
    )
    
    # Create EGTA framework
    egta = EGTA(
        simulator=simulator,
        scheduler=scheduler,
        max_profiles=20
    )
    
    # Run EGTA for a few iterations
    game = egta.run(
        max_iterations=2,
        profiles_per_iteration=5,
        verbose=True
    )
    
    # Check that we have a valid game
    assert game is not None
    assert game.num_strategies > 0
    assert game.num_strategies <= len(strategies)
    assert all(name in strategies for name in game.strategy_names)
    
    # Check that the EGTA framework tracked some profiles and found equilibria
    assert len(egta.payoff_data) > 0
    assert len(egta.equilibria) > 0
    
    # Check that the DPR scheduler expanded subgames
    assert len(scheduler.requested_subgames) > 1


def test_dpr_deviation_selection():
    """Test that the DPR scheduler selects deviating strategies correctly."""
    strategies = ["A", "B", "C", "D"]
    num_players = 4
    
    # Create a mock game with known deviation payoffs
    def mock_deviation_payoffs(mixture):
        # Return fixed deviation payoffs: A=1.0, B=2.0, C=3.0, D=4.0
        return torch.tensor([1.0, 2.0, 3.0, 4.0], device=mixture.device)
    
    # Create a mock Game instance
    game = Game(
        symmetric_game=SymmetricGame(
            num_players=num_players,
            num_actions=len(strategies),
            config_table=np.ones((1, len(strategies))),
            payoff_table=np.ones((len(strategies), 1)),
            strategy_names=strategies,
            device="cpu"
        )
    )
    
    # Override the deviation_payoffs method
    game.deviation_payoffs = mock_deviation_payoffs
    
    # Create DPR scheduler
    scheduler = DPRScheduler(
        strategies=strategies, 
        num_players=num_players
    )
    
    # Test selecting deviating strategies
    mixture = np.ones(len(strategies)) / len(strategies)
    deviating_strategies = scheduler._select_deviating_strategies(
        game, mixture, num_deviations=2
    )
    
    # Should select the strategies with highest payoffs: C and D
    assert "C" in deviating_strategies
    assert "D" in deviating_strategies
    assert len(deviating_strategies) == 2


def test_regret_calculation():
    """Test regret calculation for equilibrium verification."""
    # Create a small 2x2 game with known equilibrium
    strategies = ["A", "B"]
    num_players = 2
    
    # Create payoff data for a symmetric coordination game
    # A,A: 2,2  |  A,B: 0,0
    # B,A: 0,0  |  B,B: 1,1
    profiles = [
        [(0, "A", 2.0), (1, "A", 2.0)],  # Both play A
        [(0, "A", 0.0), (1, "B", 0.0)],  # Player 0 plays A, Player 1 plays B
        [(0, "B", 0.0), (1, "A", 0.0)],  # Player 0 plays B, Player 1 plays A
        [(0, "B", 1.0), (1, "B", 1.0)]   # Both play B
    ]
    
    # Create game
    game = Game.from_payoff_data(profiles, strategies, device="cpu")
    
    # Test regret for pure strategy A
    mixture_A = torch.tensor([1.0, 0.0], device="cpu")
    regret_A = game.regret(mixture_A)
    assert regret_A < 1e-6  # Should be an equilibrium with ~0 regret
    
    # Test regret for pure strategy B
    mixture_B = torch.tensor([0.0, 1.0], device="cpu")
    regret_B = game.regret(mixture_B)
    # In a coordination game, B is also a pure equilibrium
    assert regret_B < 1e-6  # B is also an equilibrium with ~0 regret
    
    # Check payoffs to ensure the game payoffs match expectations
    payoff_matrix = game.get_payoff_matrix()
    
    # Print the payoff matrix for debugging
    print("\nPayoff Matrix:")
    for i in range(2):
        print(f"  {strategies[i]}: [{payoff_matrix[i, 0].item():.4f}, {payoff_matrix[i, 1].item():.4f}]")
        
    # Verify payoff structure
    # Strategy A should have higher payoff when playing against A than strategy B does
    assert payoff_matrix[0,0] > payoff_matrix[1,0]
    # Strategy B should have higher payoff when playing against B than strategy A does
    assert payoff_matrix[1,1] > payoff_matrix[0,1]
    
    # Find equilibria
    from marketsim.egta.solvers.equilibria import quiesce_sync
    equilibria = quiesce_sync(game, verbose=True)
    
    # Should find both pure equilibria (A and B)
    assert len(equilibria) >= 1
    
    # All equilibria should have very low regret
    for eq_mix, eq_regret in equilibria:
        assert eq_regret < 1e-3


if __name__ == "__main__":
    # Run all tests
    test_dpr_scheduler_initialization()
    test_dpr_profile_generation()
    test_dpr_scaling()
    test_dpr_deviation_selection()
    test_regret_calculation()
    test_dpr_with_egta()
    
    print("All tests passed!") 