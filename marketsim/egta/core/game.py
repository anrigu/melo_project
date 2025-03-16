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
        
        # Collect all payoffs for normalization
        all_payoffs = []
        for profile in payoff_data:
            for _, _, payoff in profile:
                if payoff is not None and not np.isnan(float(payoff)) and not np.isinf(float(payoff)):
                    all_payoffs.append(float(payoff))
        
        # Calculate mean and std for normalization
        if all_payoffs:
            payoff_mean = np.mean(all_payoffs)
            payoff_std = np.std(all_payoffs)
            if payoff_std < 1e-10:  # If standard deviation is too small
                payoff_std = 1.0
        else:
            payoff_mean = 0.0
            payoff_std = 1.0
        
        # Save normalization constants in metadata
        if metadata is None:
            metadata = {}
        metadata['payoff_mean'] = payoff_mean
        metadata['payoff_std'] = payoff_std
        
        for profile in payoff_data:
            strat_counts = [0] * len(strategy_names)
            for _, strategy, _ in profile:
                strat_idx = strategy_to_index[strategy]
                strat_counts[strat_idx] += 1
            
            strat_counts_tuple = tuple(strat_counts)
            
            profile_dict[strat_counts_tuple]["count"] += 1
            for _, strategy, payoff in profile:
                strat_idx = strategy_to_index[strategy]
                # Normalize the payoff
                norm_payoff = (float(payoff) - payoff_mean) / payoff_std
                profile_dict[strat_counts_tuple]["payoffs"][strat_idx].append(norm_payoff)
        
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
            device=device,
            offset=payoff_mean,  # Store mean as offset
            scale=payoff_std     # Store std as scale
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
                    payoff = self.get_payoff(s, c)
                    
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
    
    def get_payoff(self, strategy_idx: int, config_idx: int) -> float:
        """
        Get the payoff for a strategy in a configuration.
        
        Args:
            strategy_idx: Strategy index
            config_idx: Configuration index
            
        Returns:
            Payoff value
        """
        weighted_payoff = self.game.payoff_table[strategy_idx, config_idx].item()
        profile = self.game.config_table[config_idx].detach().cpu().numpy()
        repeats = logmultinomial(*profile)
        
        # Clip the weighted_payoff to prevent overflow
        weighted_payoff_clipped = min(weighted_payoff, 700)  # Prevent exp overflow
        
        # Use the clipped value for the exponential calculation
        normalized_payoff = np.exp(weighted_payoff_clipped - repeats)
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
    
    def get_payoff_matrix(self) -> torch.Tensor:
        """
        Get the payoff matrix for this game.
        For a 2-strategy game, returns a 2x2 matrix where:
        - payoff_matrix[i,j] = payoff for strategy i when played against strategy j
        
        Returns:
            Payoff matrix
        """
        num_strategies = self.num_strategies
        if num_strategies > 10:
            raise ValueError("Direct payoff matrix computation only supported for games with ≤ 10 strategies")
            
        payoff_matrix = torch.zeros((num_strategies, num_strategies), device=self.game.device)
        
        # Iterate through all pure strategy profiles
        for i in range(num_strategies):
            for j in range(num_strategies):
                # Create a pure strategy profile where all players play strategy j
                pure_profile = torch.zeros(num_strategies, device=self.game.device)
                pure_profile[j] = 1.0
                
                # Calculate payoff for deviating to strategy i against this profile
                try:
                    dev_payoffs = self.deviation_payoffs(pure_profile)
                    payoff_matrix[i, j] = dev_payoffs[i]
                except Exception as e:
                    print(f"Error computing payoff for [{i},{j}]: {e}")
                    # Use a fallback value
                    payoff_matrix[i, j] = 0.0
        
        return payoff_matrix
    
    def find_nash_equilibrium_2x2(self) -> List[Tuple[torch.Tensor, float]]:
        """
        Find Nash equilibria for a 2x2 game using direct computation.
        This is more stable than using quiesce for 2-strategy games.
        
        Returns:
            List of (mixture, regret) tuples for equilibria
        """
        if self.num_strategies != 2:
            raise ValueError("This method only works for 2x2 games")
            
        device = self.game.device
        equilibria = []
        
        # Get the payoff matrix
        payoff_matrix = self.get_payoff_matrix()
        
        # Check if any entries are NaN and replace with safe values
        if torch.isnan(payoff_matrix).any():
            print("Warning: NaN values in payoff matrix. Replacing with zeros.")
            payoff_matrix = torch.nan_to_num(payoff_matrix, nan=0.0)
        
        # Check pure strategy equilibria
        # Strategy 0 is a pure equilibrium if it's the best response to itself
        if payoff_matrix[0, 0] >= payoff_matrix[1, 0]:
            pure_0 = torch.tensor([1.0, 0.0], device=device)
            # Compute regret directly
            regret_0 = max(0, payoff_matrix[1, 0] - payoff_matrix[0, 0])
            equilibria.append((pure_0, regret_0))
            
        # Strategy 1 is a pure equilibrium if it's the best response to itself
        if payoff_matrix[1, 1] >= payoff_matrix[0, 1]:
            pure_1 = torch.tensor([0.0, 1.0], device=device)
            # Compute regret directly
            regret_1 = max(0, payoff_matrix[0, 1] - payoff_matrix[1, 1])
            equilibria.append((pure_1, regret_1))
            
        # Check for mixed equilibrium
        # In a 2x2 game, a mixed equilibrium exists if no pure equilibrium dominates
        # and both strategies are best responses to some mixture of the opponent
        
        # Calculate the indifference points
        # For a mixed equilibrium, each player must be indifferent between their strategies
        denom_0 = payoff_matrix[0, 0] - payoff_matrix[1, 0] - payoff_matrix[0, 1] + payoff_matrix[1, 1]
        
        # Skip mixed equilibrium calculation if denominator is close to zero (parallel payoffs)
        if abs(denom_0) > 1e-10:
            p = (payoff_matrix[1, 1] - payoff_matrix[0, 1]) / denom_0
            
            # Valid mixed equilibrium must have p between 0 and 1
            if 0 < p < 1:
                mixed = torch.tensor([p, 1-p], device=device)
                # Compute regret for mixed equilibrium (should be zero or very small)
                dev_payoffs = self.deviation_payoffs(mixed)
                exp_payoff = (mixed * dev_payoffs).sum()
                regret_mixed = max(0, torch.max(dev_payoffs - exp_payoff).item())
                
                # Add if regret is small enough
                if regret_mixed < 1e-3:
                    equilibria.append((mixed, regret_mixed))
        
        # If no equilibria found (rare), fall back to replicator dynamics
        if not equilibria:
            print("No equilibria found using direct computation. Falling back to replicator dynamics.")
            from marketsim.egta.solvers.equilibria import replicator_dynamics, regret
            mixture = torch.ones(2, device=device) / 2
            eq_mixture = replicator_dynamics(self, mixture, iters=5000)
            eq_regret = regret(self, eq_mixture)
            
            # Handle potential NaN
            if torch.is_tensor(eq_regret) and torch.isnan(eq_regret).any():
                eq_regret = torch.tensor(0.01, device=device)
            equilibria.append((eq_mixture, eq_regret))
        
        return equilibria
        
    def restrict(self, strategy_indices: List[int]) -> 'Game':
        """
        Create a restricted game that only allows the specified strategies.
        
        Args:
            strategy_indices: List of strategy indices to include
            
        Returns:
            Restricted game
        """
        if not strategy_indices:
            raise ValueError("At least one strategy must be included")
            
        if len(strategy_indices) == self.num_strategies:
            # No restriction needed
            return self
            
        # Get the relevant strategy names
        restricted_strategy_names = [self.strategy_names[i] for i in strategy_indices]
        
        # Create a selector mask for the payoff table and config table
        strategy_mask = torch.zeros(self.num_strategies, dtype=torch.bool, device=self.game.device)
        for idx in strategy_indices:
            strategy_mask[idx] = True
            
        # Extract sub-matrices from the config and payoff tables
        config_table = self.game.config_table[:, strategy_mask].detach().cpu().numpy()
        
        # Filter for configurations with non-zero counts (any strategy used)
        valid_configs = np.sum(config_table, axis=1) > 0
        config_table = config_table[valid_configs]
        
        # Extract the corresponding rows from the payoff table
        payoff_table = self.game.payoff_table[strategy_mask][:, valid_configs].detach().cpu().numpy()
        
        # Create a new symmetric game
        restricted_sym_game = SymmetricGame(
            num_players=self.num_players,
            num_actions=len(strategy_indices),
            config_table=config_table,
            payoff_table=payoff_table,
            strategy_names=restricted_strategy_names,
            device=self.game.device,
            offset=self.game.offset,
            scale=self.game.scale
        )
        
        # Create metadata for the restricted game
        metadata = {**self.metadata} if self.metadata else {}
        metadata['restricted_from'] = self.strategy_names
        metadata['restricted_to'] = restricted_strategy_names
        
        return Game(restricted_sym_game, metadata) 