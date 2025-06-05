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
from marketsim.game.role_symmetric_game import RoleSymmetricGame
from marketsim.custom_math.simplex_operations import logmultinomial


class Game:
    """
    A game representation that wraps either SymmetricGame or RoleSymmetricGame.
    Automatically detects the game type from payoff data.
    """
    
    def __init__(self, game_instance, metadata: Optional[Dict] = None):
        """
        Initialize a game wrapper.
        
        Args:
            game_instance: Either a SymmetricGame or RoleSymmetricGame instance
            metadata: Optional metadata dictionary
        """
        self.game = game_instance
        self.metadata = metadata or {}
        
        # Extract common properties
        if isinstance(game_instance, RoleSymmetricGame):
            self.is_role_symmetric = True
            self.role_names = game_instance.role_names
            self.num_players_per_role = game_instance.num_players_per_role
            self.strategy_names_per_role = game_instance.strategy_names_per_role
            self.role_indices = game_instance.role_indices
            self.role_starts  = game_instance.role_starts
            # Flatten strategy names for role symmetric games
            all_strategy_names = []
            for strategies in game_instance.strategy_names_per_role:
                all_strategy_names.extend(strategies)
            self._strategy_names = all_strategy_names
            self._num_players = game_instance.num_players_per_role.sum().item()
            self._num_strategies = len(self._strategy_names)
        else:
            self.is_role_symmetric = False
            self._strategy_names = game_instance.strategy_names
            self._num_players = game_instance.num_players
            self._num_strategies = game_instance.num_actions
            self.role_names = ["Player"]  # Single role for symmetric games
            self.num_players_per_role = [self._num_players]
            self.strategy_names_per_role = [self._strategy_names]
    
    @property
    def strategy_names(self):
        """Get strategy names."""
        return self._strategy_names
    
    @property
    def num_players(self):
        """Get number of players."""
        return self._num_players
    
    @property
    def num_strategies(self):
        """Get number of strategies."""
        return self._num_strategies
    
    
        # ------------------------------------------------------------------
    # canonical sorting helper  – works for both symmetric & role-symmetric
    def _profile_to_key(self, profile: List[Tuple[str, str]]) -> Tuple[Tuple[str, str], ...]:
        """
        Convert a list [(role,strat), …] to a canonical, hashable key.
        For 1-role symmetric games 'role' will be the same for every entry.
        """
        # ignore player-id if it was included upstream
        return tuple(sorted((r, s) for r, s in profile))

    # ------------------------------------------------------------------
    def has_profile(self, profile: List[Tuple[str, str]]) -> bool:
        """
        True  ↔  every player's payoff for this pure profile
        already exists in the empirical game tables.
        """
        key = self._profile_to_key(profile)

        # ---------- quick check via a cached set ----------
        if not hasattr(self, "_profile_key_set"):
            self._profile_key_set: set = set()
            self._rebuild_profile_key_set()          # defined below
        if key in self._profile_key_set:
            return True

        # ---------- fall-back: rebuild & test once ----------
        self._rebuild_profile_key_set()
        return key in self._profile_key_set

    def _rebuild_profile_key_set(self):
        """(re)fill _profile_key_set from  current payoff tables."""
        pk = set()
        if self.is_role_symmetric and self.game.rsg_config_table is not None:
            # build keys from rsg_config_table rows
            cfg = self.game.rsg_config_table.cpu().numpy()
            # mapping global index ↦ (role,strat)
            glob2name = []
            for role, strats in zip(self.role_names, self.strategy_names_per_role):
                glob2name.extend([(role, s) for s in strats])

            for row in cfg:
                pure_profile = []
                for g_idx, count in enumerate(row.astype(int)):
                    pure_profile.extend([glob2name[g_idx]] * count)
                pk.add(tuple(sorted(pure_profile)))

        elif not self.is_role_symmetric and self.game.config_table is not None:
            cfg = self.game.config_table.cpu().numpy()  # shape: (#configs, num_strats)
            strats = self.strategy_names
            for row in cfg:
                pure_profile = []
                for s_idx, count in enumerate(row.astype(int)):
                    pure_profile.extend([(self.role_names[0], strats[s_idx])] * count)
                pk.add(tuple(sorted(pure_profile)))

        self._profile_key_set = pk
    # ------------------------------------------------------------------

    
    def deviation_payoffs(self, mixture):
        if not torch.is_tensor(mixture):
            mixture = torch.tensor(mixture, dtype=torch.float32,
                                device=self.game.device)
        elif mixture.device != self.game.device:
            mixture = mixture.to(self.game.device)
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
                        device: str = "cpu",
                        normalize_payoffs: bool = True) -> 'Game':
        """
        Create a game from payoff data, automatically detecting if it's role symmetric.
        
        Args:
            payoff_data: List of simulation results
            device: PyTorch device
            normalize_payoffs: Whether to normalize payoffs
            
        Returns:
            Game instance
        """
        # Detect if this is role symmetric data
        if payoff_data and len(payoff_data[0]) > 0:
            first_entry = payoff_data[0][0]
            
            # Check if entries have role information (4 elements vs 3)
            if len(first_entry) == 4:
                # Role symmetric: (player_id, role_name, strategy_name, payoff)
                return cls._create_role_symmetric_game(payoff_data, device, normalize_payoffs)
            elif len(first_entry) == 3:
                # Regular symmetric: (player_id, strategy_name, payoff)
                return cls._create_symmetric_game(payoff_data, device, normalize_payoffs)
            else:
                raise ValueError(f"Invalid payoff data format. Expected 3 or 4 elements per entry, got {len(first_entry)}")
        else:
            raise ValueError("Empty payoff data provided")
    
    @classmethod
    def _create_role_symmetric_game(cls, 
                                   payoff_data: List[List[Tuple[str, str, str, float]]], 
                                   device: str = "cpu",
                                   normalize_payoffs: bool = True) -> 'Game':
        """Create a role symmetric game from payoff data."""
        # Extract role and strategy information
        all_roles = set()
        all_strategies_by_role = defaultdict(set)
        role_player_counts = defaultdict(int)
        
        for profile_data in payoff_data:
            current_profile_roles = defaultdict(int)
            for player_id, role_name, strategy_name, payoff in profile_data:
                all_roles.add(role_name)
                all_strategies_by_role[role_name].add(strategy_name)
                current_profile_roles[role_name] += 1
            
            # Update max player counts per role
            for role, count in current_profile_roles.items():
                role_player_counts[role] = max(role_player_counts[role], count)
        
        # Sort roles and strategies for consistency
        role_names = sorted(list(all_roles))
        num_players_per_role = [role_player_counts[role] for role in role_names]
        strategy_names_per_role = [sorted(list(all_strategies_by_role[role])) for role in role_names]
        
        # Create role symmetric game
        rsg = RoleSymmetricGame.from_payoff_data_rsg(
            payoff_data=payoff_data,
            role_names=role_names,
            num_players_per_role=num_players_per_role,
            strategy_names_per_role=strategy_names_per_role,
            device=device,
            normalize_payoffs=normalize_payoffs
        )
        
        metadata = {
            'game_type': 'role_symmetric',
            'num_profiles': len(payoff_data),
            'payoff_mean': rsg.offset if normalize_payoffs else 0.0,
            'payoff_std': rsg.scale if normalize_payoffs else 1.0
        }
        
        return cls(rsg, metadata)
    
    @classmethod
    def _create_symmetric_game(cls, 
                              payoff_data: List[List[Tuple[int, str, float]]], 
                              device: str = "cpu",
                              normalize_payoffs: bool = True) -> 'Game':
        """Create a symmetric game from payoff data."""
        # Extract strategy information
        all_strategies = set()
        num_players = 0
        
        for profile_data in payoff_data:
            all_strategies.update(strategy for _, strategy, _ in profile_data)
            num_players = max(num_players, len(profile_data))
        
        strategy_names = sorted(list(all_strategies))
        
        def payoff_function(profile_counts):
            """Payoff function for creating the symmetric game."""
            # This is a placeholder - the actual payoffs will be updated from data
            return np.zeros(len(strategy_names))
        
        # Create symmetric game
        sym_game = SymmetricGame.from_payoff_function(
            num_players=num_players,
            num_actions=len(strategy_names),
            payoff_function=payoff_function,
            strategy_names=strategy_names,
            device=device
        )
        
        # Update with actual data
        sym_game.update_with_new_data(payoff_data)
        
        metadata = {
            'game_type': 'symmetric',
            'num_profiles': len(payoff_data),
            'payoff_mean': sym_game.offset,
            'payoff_std': sym_game.scale
        }
        
        return cls(sym_game, metadata)
    
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
    
    def update_with_new_data(self, new_data: List[List[Tuple]]):
        """Update the game with new payoff data."""
        self.game.update_with_new_data(new_data)
        if 'num_profiles' in self.metadata:
            self.metadata['num_profiles'] += len(new_data)
    
    def save(self, filepath: str):
        """Save the game to a file."""
        # Convert tensor values to Python primitives for JSON serialization
        def convert_to_primitive(obj):
            """Convert tensors and numpy arrays to Python primitives."""
            if hasattr(obj, 'item') and hasattr(obj, 'numel'):  # PyTorch tensor
                if obj.numel() == 1:
                    return obj.item()
                else:
                    return obj.tolist()
            elif hasattr(obj, 'item'):  # numpy scalar
                return obj.item()
            elif hasattr(obj, 'tolist'):  # PyTorch tensor or numpy array
                return obj.tolist()
            elif isinstance(obj, list):
                return [convert_to_primitive(item) for item in obj]
            elif isinstance(obj, dict):
                return {key: convert_to_primitive(value) for key, value in obj.items()}
            else:
                return obj
        
        game_data = {
            'metadata': convert_to_primitive(self.metadata),
            'is_role_symmetric': self.is_role_symmetric,
            'strategy_names': convert_to_primitive(self.strategy_names),
            'num_players': convert_to_primitive(self.num_players),
            'num_strategies': convert_to_primitive(self.num_strategies)
        }
        
        if self.is_role_symmetric:
            game_data.update({
                'role_names': convert_to_primitive(self.role_names),
                'num_players_per_role': convert_to_primitive(self.num_players_per_role),
                'strategy_names_per_role': convert_to_primitive(self.strategy_names_per_role)
            })
        
        with open(filepath, 'w') as f:
            json.dump(game_data, f, indent=2)
    
    def get_payoff_matrix(self) -> torch.Tensor:
        """Get the payoff matrix (for 2-player symmetric games)."""
        if not self.is_role_symmetric and self.num_strategies == 2:
            return self.game.payoff_table.cpu()
        else:
            raise NotImplementedError("Payoff matrix only available for 2-strategy symmetric games")
    
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
        
    def restrict(self, restriction_indices: List[int]) -> 'Game':
        """Create a restricted game."""
        if self.is_role_symmetric:
            restricted_game = self.game.restrict(restriction_indices)
        else:
            # For symmetric games, create restriction mask
            restriction_mask = torch.zeros(self.num_strategies, dtype=torch.bool)
            restriction_mask[restriction_indices] = True
            restricted_game = self.game.restrict(restriction_mask)
        
        return Game(restricted_game, self.metadata)
    
    def __repr__(self):
        game_type = "RoleSymmetricGame" if self.is_role_symmetric else "SymmetricGame"
        return f"Game({game_type}, {self.num_players} players, {self.num_strategies} strategies)" 
    
    
    def get_strategy_name(self, global_idx: int) -> str:
        """
        forwarder so helpers can always call Game.get_strategy_name().
        """
        if self.is_role_symmetric:                
            return self.game.get_strategy_name(global_idx)
        else:                                 
            return self.strategy_names[global_idx]


   
    def all_strategy_names(self) -> List[str]:
        return self.strategy_names if not self.is_role_symmetric else [
            f"{r}:{s}"
            for r, role_strats in zip(self.role_names, self.strategy_names_per_role)
            for s in role_strats
        ]

    def strategies_present_in_payoff_table(self) -> set:
        """Return the set {'MOBI:MOBI_0_100', ...} that actually appear."""
        present = set()
        if self.is_role_symmetric and self.game.rsg_config_table is not None:
            rows = self.game.rsg_config_table.cpu().numpy()
            for cfg_row in rows:
                for g_idx, cnt in enumerate(cfg_row.astype(int)):
                    if cnt > 0:
                        role, strat = self.get_strategy_name(g_idx).split(':', 1) \
                                    if ':' in self.get_strategy_name(g_idx) else \
                                    (self.role_names[self.role_indices[g_idx]],
                                    self.get_strategy_name(g_idx))
                        present.add(f"{role}:{strat}")
        elif not self.is_role_symmetric and self.game.config_table is not None:
            rows = self.game.config_table.cpu().numpy()
            for cfg_row in rows:
                for idx, cnt in enumerate(cfg_row.astype(int)):
                    if cnt > 0:
                        present.add(self.strategy_names[idx])
        return present
