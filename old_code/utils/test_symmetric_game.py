import pytest
import torch
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from symmetric_game import SymmetricGame

def test_symmetric_game_initialization():
    # Simple 2-player, 2-action game
    # Strategy profiles: (2,0), (1,1), (0,2)
    config_table = np.array([
        [1, 0],  # opponent plays first strategy
        [0, 1],  # opponent plays second strategy
    ])
    
    payoff_table = np.array([
        [3.0, 0.0],  # payoffs for first strategy
        [5.0, 1.0],  # payoffs for second strategy
    ])
    
    strategy_names = ["Cooperate", "Defect"]
    
    game = SymmetricGame(
        num_players=2,
        num_actions=2,
        config_table=config_table,
        payoff_table=payoff_table,
        strategy_names=strategy_names
    )
    
    assert game.num_players == 2
    assert game.num_actions == 2
    assert len(game.strategy_names) == 2
    assert torch.all(torch.isclose(game.config_table, torch.tensor(config_table, dtype=torch.float32)))

def test_from_payoff_function():
    # Simple prisoners dilemma payoff function
    def prisoners_dilemma_payoffs(profile):
        """Returns payoffs for a 2-player prisoners dilemma
        profile[0] = number of cooperators
        profile[1] = number of defectors
        """
        payoffs = np.zeros(2)
        
        # If opponent cooperates
        if profile[0] == 1:  # One other cooperator
            payoffs[0] = 3  # Cooperate: both get 3
            payoffs[1] = 5  # Defect: I get 5, they get 0
        
        # If opponent defects
        if profile[1] == 1:  # One other defector
            payoffs[0] = 0  # Cooperate: I get 0, they get 5
            payoffs[1] = 1  # Defect: both get 1
            
        return payoffs
    
    # Need to provide strategy_names which is a required parameter
    strategy_names = ["Cooperate", "Defect"]
    
    game = SymmetricGame.from_payoff_function(
        num_players=2,
        num_actions=2,
        payoff_function=prisoners_dilemma_payoffs,
        strategy_names=strategy_names,  # Add this parameter
        device="cpu"
    )
    
    assert game.num_players == 2
    assert game.num_actions == 2
    assert game.strategy_names == strategy_names

def test_pure_payoffs():
    # Simple 2-player, 2-action game (Prisoners Dilemma)
    config_table = np.array([
        [1, 0],  # opponent plays first strategy (cooperate)
        [0, 1],  # opponent plays second strategy (defect)
    ])
    
    # Raw payoffs before any log transformations - avoid log(0) warnings
    raw_payoff_table = np.array([
        [3.0, 0.1],  # payoffs for first strategy (cooperate)
        [5.0, 1.0],  # payoffs for second strategy (defect)
    ])
    
    # Convert to log payoffs like the actual implementation would
    payoff_table = np.log(raw_payoff_table)
    
    strategy_names = ["Cooperate", "Defect"]
    
    game = SymmetricGame(
        num_players=2,
        num_actions=2,
        config_table=config_table,
        payoff_table=payoff_table,
        strategy_names=strategy_names
    )
    
    # Test payoffs when opponent cooperates
    payoffs = game.pure_payoffs([1, 0])
    assert torch.isclose(payoffs[0], torch.tensor(3.0))  # C vs C
    assert torch.isclose(payoffs[1], torch.tensor(5.0))  # D vs C
    
    # Test payoffs when opponent defects
    payoffs = game.pure_payoffs([0, 1])
    assert torch.isclose(payoffs[0], torch.tensor(0.1))  # C vs D
    assert torch.isclose(payoffs[1], torch.tensor(1.0))  # D vs D

def test_deviation_payoffs():
    # Simple 2-player, 2-action game (Prisoners Dilemma)
    config_table = np.array([
        [1, 0],  # opponent plays first strategy (cooperate)
        [0, 1],  # opponent plays second strategy (defect)
    ])
    
    # Raw payoffs before any log transformations - avoid log(0) warnings
    raw_payoff_table = np.array([
        [3.0, 0.1],  # payoffs for first strategy (cooperate)
        [5.0, 1.0],  # payoffs for second strategy (defect)
    ])
    
    # Convert to log payoffs like the actual implementation would
    payoff_table = np.log(raw_payoff_table)
    
    strategy_names = ["Cooperate", "Defect"]
    
    game = SymmetricGame(
        num_players=2,
        num_actions=2,
        config_table=config_table,
        payoff_table=payoff_table,
        strategy_names=strategy_names
    )
    
    # Test deviation payoffs for pure strategies
    # If opponent plays pure cooperate
    dev_payoffs = game.deviation_payoffs([1.0, 0.0])
    assert torch.isclose(dev_payoffs[0], torch.tensor(3.0))  # C vs C
    assert torch.isclose(dev_payoffs[1], torch.tensor(5.0))  # D vs C
    
    # If opponent plays pure defect
    dev_payoffs = game.deviation_payoffs([0.0, 1.0])
    assert torch.isclose(dev_payoffs[0], torch.tensor(0.1))  # C vs D
    assert torch.isclose(dev_payoffs[1], torch.tensor(1.0))  # D vs D
    
    # Test deviation payoffs for mixed strategy
    # If opponent plays 50% cooperate, 50% defect
    dev_payoffs = game.deviation_payoffs([0.5, 0.5])
    assert torch.isclose(dev_payoffs[0], torch.tensor(1.55))  # C vs mixed (3*0.5 + 0.1*0.5)
    assert torch.isclose(dev_payoffs[1], torch.tensor(3.0))  # D vs mixed (5*0.5 + 1*0.5)

def test_payoff_normalization():
    # Test normalization/denormalization
    raw_payoffs = np.array([1.0, 5.0, 10.0])
    
    # Use the static methods directly
    offset, scale = SymmetricGame._set_scale(min_val=1.0, max_val=10.0, lb=0.0, ub=1.0)
    normalized = SymmetricGame._normalize_payoffs(raw_payoffs, offset, scale)
    denormalized = SymmetricGame._denormalize_payoffs(normalized, offset, scale)
    
    # Check that normalized values are in [0,1]
    assert np.all(normalized >= 0.0)
    assert np.all(normalized <= 1.0)
    
    # Check that denormalization gives back original values
    assert np.allclose(denormalized, raw_payoffs)