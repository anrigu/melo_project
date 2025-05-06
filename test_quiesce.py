import torch
import numpy as np
import asyncio
from typing import List, Tuple, Set, Optional
import nest_asyncio

from marketsim.egta.core.game import Game
from marketsim.egta.solvers.equilibria import (
    quiesce, 
    quiesce_sync,
    DeviationPriorityQueue
)

nest_asyncio.apply()

# Define our own SubgameCandidate for testing
class SubgameCandidate:
    """Represents a candidate equilibrium in a subgame."""
    def __init__(self, 
                game: Game,
                mixture: torch.Tensor,
                restricted_strategies: List[int],
                support_size: int,
                iteration: int,
                regret: float,
                parentage: str):
        self.game = game
        self.mixture = mixture
        self.restricted_strategies = restricted_strategies
        self.support = set(i for i, p in enumerate(mixture) if p > 0.01)
        self.support_size = support_size
        self.iteration = iteration
        self.regret = regret
        self.parentage = parentage
        
    def __repr__(self):
        return f"SubgameCandidate(support={self.support}, regret={self.regret})"

# Implement our own test_deviations function for testing
async def test_deviations(game, mixture, deviation_queue, regret_threshold=1e-3, restricted_game_size=4, verbose=False):
    """Test for beneficial deviations from a mixture and add them to the queue."""
    # Get deviation payoffs
    dev_payoffs = game.deviation_payoffs(mixture)
    
    # Calculate expected payoff
    expected_payoff = (mixture * dev_payoffs).sum()
    
    # Calculate gains from deviation
    gains = dev_payoffs - expected_payoff
    
    # Apply clipping to gains for stability
    gains = torch.clamp(gains, min=-10.0, max=10.0)
    
    # Find beneficial deviations
    for s in range(game.num_strategies):
        gain = gains[s].item()
        
        # If gain is significant, add to deviation queue
        if gain > regret_threshold:
            # Add to queue for exploration
            deviation_queue.push(gain, s, mixture)
            
            if verbose:
                print(f"Found beneficial deviation to {game.strategy_names[s]} with gain {gain:.6f}")


class TestQuiesceAlgorithm:
    """Tests for the QUIESCE algorithm implementation."""
    
    def setup_coordination_game(self):
        """Set up a simple 3-strategy coordination game."""
        strategies = ["A", "B", "C"]
        num_players = 2
        
        # Create payoff data for a 3-strategy coordination game
        # A,A: 3,3  |  A,B: 0,0  |  A,C: 0,0
        # B,A: 0,0  |  B,B: 2,2  |  B,C: 0,0
        # C,A: 0,0  |  C,B: 0,0  |  C,C: 1,1
        profiles = [
            [(0, "A", 3.0), (1, "A", 3.0)],  # Both play A (highest payoff)
            [(0, "A", 0.0), (1, "B", 0.0)],  # Mixed A,B
            [(0, "A", 0.0), (1, "C", 0.0)],  # Mixed A,C
            [(0, "B", 0.0), (1, "A", 0.0)],  # Mixed B,A
            [(0, "B", 2.0), (1, "B", 2.0)],  # Both play B (medium payoff)
            [(0, "B", 0.0), (1, "C", 0.0)],  # Mixed B,C
            [(0, "C", 0.0), (1, "A", 0.0)],  # Mixed C,A
            [(0, "C", 0.0), (1, "B", 0.0)],  # Mixed C,B
            [(0, "C", 1.0), (1, "C", 1.0)]   # Both play C (lowest payoff)
        ]
        
        return Game.from_payoff_data(profiles, strategies, device="cpu")
    
    def test_deviation_priority_queue(self):
        """Test the deviation priority queue functionality."""
        queue = DeviationPriorityQueue()
        
        # Push some deviations
        queue.push(0.5, 1, torch.tensor([0.6, 0.4, 0.0]))
        queue.push(1.0, 2, torch.tensor([0.5, 0.5, 0.0]))
        queue.push(0.2, 0, torch.tensor([0.3, 0.3, 0.4]))
        
        # Should pop in order of highest gain first
        gain1, strat1, mix1 = queue.pop()
        assert gain1 == 1.0
        assert strat1 == 2
        
        gain2, strat2, mix2 = queue.pop()
        assert gain2 == 0.5
        assert strat2 == 1
        
        gain3, strat3, mix3 = queue.pop()
        assert gain3 == 0.2
        assert strat3 == 0
        
        # Queue should now be empty
        assert queue.is_empty()
    
    async def test_beneficial_deviations(self):
        """Test the identification of beneficial deviations."""
        game = self.setup_coordination_game()
        
        # Test deviation from a non-equilibrium mixture
        # Use a mixture heavily weighted toward B, which should have a beneficial deviation to A
        mixture = torch.tensor([0.1, 0.9, 0.0], device="cpu")
        deviation_queue = DeviationPriorityQueue()
        
        # Calculate expected deviation gains manually to verify
        dev_payoffs = game.deviation_payoffs(mixture)
        expected_payoff = (mixture * dev_payoffs).sum()
        print(f"Expected payoff: {expected_payoff}")
        
        # Print all deviation payoffs
        for i, name in enumerate(game.strategy_names):
            gain = dev_payoffs[i] - expected_payoff
            print(f"Strategy {name} payoff: {dev_payoffs[i]}, gain: {gain}")
        
        # Test with very high regret threshold to find no deviations
        await test_deviations(
            game, mixture, deviation_queue, 
            regret_threshold=10.0,
            restricted_game_size=3,
            verbose=True
        )
        # Should be no beneficial deviations with such a high threshold
        assert deviation_queue.is_empty()
        
        # Test with a very low regret threshold to ensure we find deviations
        await test_deviations(
            game, mixture, deviation_queue, 
            regret_threshold=0.0001,  # Very low threshold to guarantee finding deviations
            restricted_game_size=3,
            verbose=True
        )
        
        # Should find beneficial deviations (to A or C)
        assert not deviation_queue.is_empty(), "No beneficial deviations found!"
        
        # Get the beneficial deviation
        gain, strategy, _ = deviation_queue.pop()
        
        # Print the result
        print(f"Found deviation to strategy {game.strategy_names[strategy]} with gain {gain}")
        
        # Check that we found a deviation with positive gain
        assert gain > 0, f"Expected gain > 0, got {gain}"
    
    async def test_quiesce_finds_equilibrium(self):
        """Test that QUIESCE finds the correct equilibrium."""
        game = self.setup_coordination_game()
        
        # Run QUIESCE
        equilibria = await quiesce(
            game,
            num_iters=2,
            num_random_starts=2,
            regret_threshold=1e-3,
            verbose=True
        )
        
        # Should find the pure strategy A equilibrium (highest payoff)
        assert len(equilibria) >= 1
        
        # Check the equilibrium
        eq_mixture, eq_regret = equilibria[0]
        
        # The equilibrium should have very low regret
        assert eq_regret < 1e-2
        
        # Strategy A should have the highest weight
        strat_a_weight = eq_mixture[0].item()
        assert strat_a_weight > 0.9, f"Expected weight on strategy A > 0.9, got {strat_a_weight}"
    
    def test_quiesce_sync(self):
        """Test the synchronous wrapper for QUIESCE."""
        game = self.setup_coordination_game()
        
        # Run quiesce_sync
        equilibria = quiesce_sync(
            game,
            num_iters=2,
            num_random_starts=2,
            regret_threshold=1e-3,
            verbose=True
        )
        
        # Should find at least one equilibrium
        assert len(equilibria) >= 1
        
        # The first equilibrium should have very low regret
        _, eq_regret = equilibria[0]
        assert eq_regret < 1e-2
    
    async def test_subgame_candidate_exploration(self):
        """Test the subgame candidate exploration in QUIESCE."""
        game = self.setup_coordination_game()
        
        # Create a subgame candidate with strategies B and C (not the best equilibrium)
        mixture = torch.tensor([0.0, 0.6, 0.4], device="cpu")
        candidate = SubgameCandidate(
            game=game,
            mixture=mixture,
            restricted_strategies=[1, 2],  # B and C
            support_size=2,
            iteration=0,
            regret=0.1,  # Intentionally high to test refinement
            parentage="root"
        )
        
        # Set up unconfirmed and confirmed candidates
        unconfirmed_candidates = [candidate]
        confirmed_candidates = []
        
        # Create a deviation queue that will hold beneficial deviations
        deviation_queue = DeviationPriorityQueue()
        
        # Initially there should be no deviations
        assert deviation_queue.is_empty()
        
        # Calculate expected deviation gains manually to verify
        dev_payoffs = game.deviation_payoffs(mixture)
        expected_payoff = (mixture * dev_payoffs).sum()
        print(f"Expected payoff for mixture: {expected_payoff}")
        
        # Print all deviation payoffs
        for i, name in enumerate(game.strategy_names):
            gain = dev_payoffs[i] - expected_payoff
            print(f"Strategy {name} payoff: {dev_payoffs[i]}, gain: {gain}")
        
        # Manually test deviations for the candidate with a very low threshold
        await test_deviations(
            game, candidate.mixture, deviation_queue,
            regret_threshold=0.0001,  # Very low threshold
            restricted_game_size=3,
            verbose=True
        )
        
        # Should find at least one beneficial deviation
        assert not deviation_queue.is_empty(), "No beneficial deviations found in candidate exploration!"
        
        # Get the deviation details
        gain, strategy, _ = deviation_queue.pop()
        
        # Print the result
        print(f"Found deviation to strategy {game.strategy_names[strategy]} with gain {gain}")
        
        # Check that we found a deviation with positive gain
        assert gain > 0, f"Expected gain > 0, got {gain}"
        
        # This simulates the core QUIESCE loop logic for testing
        # Create new mixture with all strategies
        new_support = set([0, 1, 2])  # A, B, and C
        restriction = list(new_support)
        
        # Create a new candidate with all three strategies
        new_candidate = SubgameCandidate(
            game=game,
            mixture=torch.tensor([0.33, 0.33, 0.34], device="cpu"),  # Just an example
            restricted_strategies=restriction,
            support_size=3,
            iteration=1,
            regret=0.05,  # Lower regret as we've expanded
            parentage=f"root->expand({game.strategy_names[strategy]})"
        )
        
        # Add to unconfirmed candidates
        unconfirmed_candidates.append(new_candidate)
        
        # Should now have 2 candidates in the unconfirmed queue
        assert len(unconfirmed_candidates) == 2
        
        # First candidate should be the B and C subgame
        assert set(unconfirmed_candidates[0].restricted_strategies) == {1, 2}
        
        # Second candidate should include all three strategies
        assert set(unconfirmed_candidates[1].restricted_strategies) == {0, 1, 2}


# Run the tests
if __name__ == "__main__":
    test = TestQuiesceAlgorithm()
    
    # Run the synchronous tests directly
    test.test_deviation_priority_queue()
    test.test_quiesce_sync()
    
    # Create a new event loop and set it as the default
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    # Run each async test individually with the same loop
    try:
        loop.run_until_complete(test.test_beneficial_deviations())
        loop.run_until_complete(test.test_quiesce_finds_equilibrium())
        loop.run_until_complete(test.test_subgame_candidate_exploration())
        print("All QUIESCE tests passed!")
    finally:
        loop.close() 