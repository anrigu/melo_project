"""
Tests for the EGTA framework.
"""
import unittest
import torch
import numpy as np
from typing import List, Tuple

from marketsim.egta.core.game import Game
from marketsim.egta.solvers.equilibria import replicator_dynamics, fictitious_play, gain_descent


class TestSolvers(unittest.TestCase):
    """Test equilibrium solvers."""
    
    def setUp(self):
        """Create a simple test game."""
        # Create a simple Rock-Paper-Scissors game
        strategy_names = ["Rock", "Paper", "Scissors"]
        
        # Create payoff data for Rock-Paper-Scissors
        # Rock beats Scissors, Scissors beats Paper, Paper beats Rock
        payoff_data = []
        
        # Define payoffs for each profile
        profiles = [
            [("Rock", "Rock"), ("Rock", "Rock")],  # Both play Rock
            [("Rock", "Rock"), ("Paper", "Paper")],  # One Rock, one Paper
            [("Rock", "Rock"), ("Scissors", "Scissors")],  # One Rock, one Scissors
            [("Paper", "Paper"), ("Paper", "Paper")],  # Both play Paper
            [("Paper", "Paper"), ("Scissors", "Scissors")],  # One Paper, one Scissors
            [("Scissors", "Scissors"), ("Scissors", "Scissors")],  # Both play Scissors
        ]
        
        payoffs = [
            [0, 0],  # Tie with Rock
            [-1, 1],  # Rock loses to Paper
            [1, -1],  # Rock beats Scissors
            [0, 0],  # Tie with Paper
            [-1, 1],  # Paper loses to Scissors
            [0, 0],  # Tie with Scissors
        ]
        
        for (profile, payoff) in zip(profiles, payoffs):
            profile_data = []
            for i, (player_id, strategy) in enumerate(profile):
                profile_data.append((i, strategy, payoff[i]))
            payoff_data.append(profile_data)
        
        # Create the game
        self.game = Game.from_payoff_data(payoff_data, strategy_names)
        
        # Uniform mixture for testing
        self.uniform = torch.ones(3) / 3
    
    def test_replicator_dynamics(self):
        """Test replicator dynamics solver."""
        # Run replicator dynamics from uniform
        eq_mix = replicator_dynamics(self.game, self.uniform, iters=1000)
        
        # Check that the result is a valid mixture
        self.assertAlmostEqual(eq_mix.sum().item(), 1.0, places=5)
        
        # Check regret of result
        regret = self.game.regret(eq_mix).item()
        self.assertLess(regret, 0.01)
        
        # In RPS, the equilibrium should be uniform
        for i in range(3):
            self.assertAlmostEqual(eq_mix[i].item(), 1/3, places=2)
    
    def test_fictitious_play(self):
        """Test fictitious play solver."""
        # Run fictitious play from uniform
        eq_mix = fictitious_play(self.game, self.uniform, iters=1000)
        
        # Check that the result is a valid mixture
        self.assertAlmostEqual(eq_mix.sum().item(), 1.0, places=5)
        
        # Check regret of result
        regret = self.game.regret(eq_mix).item()
        self.assertLess(regret, 0.01)
    
    def test_gain_descent(self):
        """Test gain descent solver."""
        # Run gain descent from uniform
        eq_mix = gain_descent(self.game, self.uniform, iters=1000)
        
        # Check that the result is a valid mixture
        self.assertAlmostEqual(eq_mix.sum().item(), 1.0, places=5)
        
        # Check regret of result
        regret = self.game.regret(eq_mix).item()
        self.assertLess(regret, 0.01)


class TestPrisonersDilemma(unittest.TestCase):
    """Test with Prisoner's Dilemma game."""
    
    def setUp(self):
        """Create a Prisoner's Dilemma game."""
        strategy_names = ["Cooperate", "Defect"]
        
        # Create payoff data for Prisoner's Dilemma
        payoff_data = []
        
        # Define payoffs for each profile
        profiles = [
            [("Cooperate", "Cooperate"), ("Cooperate", "Cooperate")],  # Both Cooperate
            [("Cooperate", "Cooperate"), ("Defect", "Defect")],  # One Cooperates, one Defects
            [("Defect", "Defect"), ("Defect", "Defect")],  # Both Defect
        ]
        
        payoffs = [
            [3, 3],  # Both Cooperate: 3, 3
            [0, 5],  # Cooperate vs Defect: 0, 5
            [1, 1],  # Both Defect: 1, 1
        ]
        
        for (profile, payoff) in zip(profiles, payoffs):
            profile_data = []
            for i, (player_id, strategy) in enumerate(profile):
                profile_data.append((i, strategy, payoff[i]))
            payoff_data.append(profile_data)
        
        # Create the game
        self.game = Game.from_payoff_data(payoff_data, strategy_names)
        
        # Uniform mixture for testing
        self.uniform = torch.ones(2) / 2
    
    def test_defect_is_equilibrium(self):
        """Test that 'Defect' is the equilibrium strategy in Prisoner's Dilemma."""
        # Run replicator dynamics from uniform
        eq_mix = replicator_dynamics(self.game, self.uniform, iters=1000)
        
        # In PD, the equilibrium should be pure Defect
        self.assertGreater(eq_mix[1].item(), 0.99)  # Defect has high probability
        
        # Check regret of result
        regret = self.game.regret(eq_mix).item()
        self.assertLess(regret, 0.01)


if __name__ == '__main__':
    unittest.main() 