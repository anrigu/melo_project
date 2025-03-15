"""
Game representation for EGTA framework.
This module provides a wrapper around the SymmetricGame implementation
with additional functionality for serialization and analysis.

TODO: 
- Add support for more complex game structures (e.g., games with multiple roles)
"""
import json
import torch
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Union, Optional
from marketsim.game.symmetric_game import SymmetricGame
from marketsim.custom_math.simplex_operations import logmultinomial


class Game:
    """Wrapper around SymmetricGame with additional functionality for EGTA."""
    
    def __init__(self, 
                 symmetric_game: SymmetricGame, 
                 metadata: Optional[Dict] = None):
        """
        initialize a Game wrapper.
        
        Args:
            symmetric_game: The underlying SymmetricGame instance
            metadata: Optional metadata about the game
        """
        self.game = symmetric_game
        self.metadata = metadata or {}
    
    @property
    def num_players(self) -> int:
        return self.game.num_players
    
    @property
    def num_strategies(self) -> int:
        return self.game.num_actions
    
    @property
    def strategy_names(self) -> List[str]:
        return self.game.strategy_names
    
    def deviation_payoffs(self, mixture):
        """Calculate deviation payoffs for a mixture."""
        return self.game.deviation_payoffs(mixture)
    
    def regret(self, mixture):
        """Calculate regret for a mixture."""
        return self.game.regret(mixture)
    
    def best_responses(self, mixture, atol=1e-10):
        """Find best responses to a mixture."""
        return self.game.best_responses(mixture, atol)
    
    def profile_to_mixture(self, profile):
        """Convert a strategy profile to a mixture representation."""
        if torch.is_tensor(profile):
            profile = profile.cpu().numpy()
        return profile / self.num_players
    
    @classmethod
    def from_payoff_data(cls, 
                       payoff_data: List[List[Tuple]],
                       strategy_names: Optional[List[str]] = None,
                       device: str = "cpu",
                       metadata: Optional[Dict] = None) -> 'Game':
        """
        Create a Game from raw payoff data.
        
        Args:
            payoff_data: List of profiles with payoff data
                Each profile is a list of (player_id, strategy, payoff) tuples
            strategy_names: Optional list of strategy names
                If not provided, will be inferred from the data
            device: PyTorch device to use
            metadata: Optional metadata about the game
            
        Returns:
            A Game instance
        """
        if strategy_names is None:
            strategy_names_set = set()
            for profile in payoff_data:
                for _, strategy, _ in profile:
                    strategy_names_set.add(strategy)
            strategy_names = sorted(list(strategy_names_set))
        
      
        num_players = 0
        for profile in payoff_data:
            num_players = max(num_players, len(profile))
        
        strategy_to_index = {name: i for i, name in enumerate(strategy_names)}
        
        profile_dict = defaultdict(lambda: {"count": 0, "payoffs": [[] for _ in range(len(strategy_names))]})
        
        for profile in payoff_data:
            strat_counts = [0] * len(strategy_names)
            for _, strategy, _ in profile:
                strat_idx = strategy_to_index[strategy]
                strat_counts[strat_idx] += 1
            
            strat_counts_tuple = tuple(strat_counts)
            
            profile_dict[strat_counts_tuple]["count"] += 1
            for _, strategy, payoff in profile:
                strat_idx = strategy_to_index[strategy]
                profile_dict[strat_counts_tuple]["payoffs"][strat_idx].append(float(payoff))
        
        configs = list(profile_dict.keys())
        num_configs = len(configs)
        
        config_table = np.zeros((num_configs, len(strategy_names)))
        raw_payoff_table = np.zeros((len(strategy_names), num_configs))
        
        for c, config in enumerate(configs):
            config_table[c] = config
            
            for strat_idx in range(len(strategy_names)):
                if config[strat_idx] > 0:  # Only if strategy was used
                    payoffs = profile_dict[config]["payoffs"][strat_idx]
                    if payoffs:
                        raw_payoff_table[strat_idx, c] = np.mean(payoffs)
        
        sym_game = SymmetricGame(
            num_players=num_players,
            num_actions=len(strategy_names),
            config_table=config_table,
            payoff_table=raw_payoff_table,
            strategy_names=strategy_names,
            device=device
        )
        
        return cls(sym_game, metadata)
    
    @classmethod
    def from_json(cls, json_data: Union[str, Dict], device: str = "cpu") -> 'Game':
        """
        Create a Game from JSON data.
        
        Args:
            json_data: JSON string or dictionary with game data
            device: PyTorch device to use
            
        Returns:
            A Game instance
        """
        if isinstance(json_data, str):
            data = json.loads(json_data)
        else:
            data = json_data
        
        # Extract strategy names
        strategy_names = []
        for role in data.get('roles', []):
            if role['name'] == 'all':  # Only supporting symmetric games for now
                strategy_names = role['strategies']
                break
        
        # Extract number of players
        num_players = 0
        for role in data.get('roles', []):
            if role['name'] == 'all':
                num_players = role['count']
                break
        
        # Extract payoff data
        payoff_data = []
        for profile in data.get('profiles', []):
            profile_data = []
            for group in profile.get('symmetry_groups', []):
                if group['role'] == 'all':
                    strategy = group['strategy']
                    count = group['count']
                    payoff = group['payoff']
                    
                    # Create count entries for this strategy
                    for _ in range(count):
                        profile_data.append((0, strategy, payoff))  # Player ID doesn't matter for symmetric games
            
            if profile_data:
                payoff_data.append(profile_data)
        
        # Create metadata
        metadata = {
            'id': data.get('id'),
            'name': data.get('name'),
            'configuration': data.get('configuration')
        }
        
        return cls.from_payoff_data(payoff_data, strategy_names, device, metadata)
    
    def to_json(self) -> Dict:
        """
        Convert the game to a JSON-serializable dictionary.
        
        Returns:
            A dictionary representation of the game
        """
        data = {}
        
        # Add metadata
        data['id'] = self.metadata.get('id', hash(str(self.game.config_table.detach().cpu().numpy())))
        data['name'] = self.metadata.get('name', 'Symmetric Game')
        data['simulator_fullname'] = self.metadata.get('simulator_fullname', 'EGTA Simulator')
        
        if 'configuration' in self.metadata:
            data['configuration'] = self.metadata['configuration']
        
        # Add roles
        data['roles'] = [{
            'name': 'all',
            'count': self.num_players,
            'strategies': self.strategy_names
        }]
        
        # Add profiles
        data['profiles'] = []
        
        # Extract config table and payoff table
        config_table = self.game.config_table.detach().cpu().numpy()
        
        for c in range(config_table.shape[0]):
            profile = {
                'id': c,
                'observations_count': 1,
                'symmetry_groups': []
            }
            
            for s in range(self.num_strategies):
                if config_table[c, s] > 0:
                    # Get the payoff for this strategy in this profile
                    payoff = self._get_denormalized_payoff(s, c)
                    
                    group = {
                        'id': f"{c}_{s}",
                        'role': 'all',
                        'strategy': self.strategy_names[s],
                        'count': int(config_table[c, s]),
                        'payoff': float(payoff),
                        'payoff_sd': 0
                    }
                    
                    profile['symmetry_groups'].append(group)
            
            data['profiles'].append(profile)
        
        return data
    
    def _get_denormalized_payoff(self, strategy_idx, config_idx):
        """Get the denormalized payoff for a strategy in a configuration."""
        weighted_payoff = self.game.payoff_table[strategy_idx, config_idx].item()
        profile = self.game.config_table[config_idx].detach().cpu().numpy()
        repeats = logmultinomial(*profile)
        normalized_payoff = np.exp(weighted_payoff - repeats)
        actual_payoff = normalized_payoff / self.game.scale + self.game.offset
        return actual_payoff
    
    def update_with_new_data(self, payoff_data: List[List[Tuple]]):
        """
        Update the game with new payoff data.
        
        Args:
            payoff_data: List of profiles with payoff data
                Each profile is a list of (player_id, strategy, payoff) tuples
        """
        self.game.update_with_new_data(payoff_data)
    
    def save(self, filename: str):
        """
        Save the game to a JSON file.
        
        Args:
            filename: Path to save the file
        """
        with open(filename, 'w') as f:
            json.dump(self.to_json(), f, indent=2)
    
    @classmethod
    def load(cls, filename: str, device: str = "cpu") -> 'Game':
        """
        Load a game from a JSON file.
        
        Args:
            filename: Path to the JSON file
            device: PyTorch device to use
            
        Returns:
            A Game instance
        """
        with open(filename, 'r') as f:
            data = json.load(f)
        
        return cls.from_json(data, device) 