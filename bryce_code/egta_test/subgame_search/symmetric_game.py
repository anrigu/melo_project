import numpy as np
from collections import defaultdict

import numpy as np
import torch
from abc import ABC, abstractmethod
from itertools import combinations_with_replacement
from collections import Counter

from marketsim.egta_test.subgame_search.utils.simplex_operations import logmultinomial, simplex_normalize
from torch.nn.functional import pad
from marketsim.egta_test.subgame_search.game import AbstractGame, MINIMUM_PAYOFF, MAXIMUM_PAYOFF, F32_EPSILON, F64_EPSILON
from tabulate import tabulate  
import math
import logging


class SymmetricGame(AbstractGame):
    """
    representation of symmetric games based on Bryce's implementation.
    uses count-based profiles and efficient computation of deviation payoffs.
    based off of Bryce's implementation: https://github.com/Davidson-Game-Theory-Research/gameanalysis.jl/blob/master/SymmetricGames.jl

    """
    def __init__(self, num_players, num_actions, config_table, payoff_table, strategy_names,
                 offset=0.0, scale=1.0, epsilon=None, device="cpu"):
        """
        initialize a symmetric game.
        inputs:
            num_players: Number of players
            num_actions: Number of actions/strategies
            config_table: Matrix of strategy counts (num_configs x num_actions)
            payoff_table: Matrix of payoffs (num_actions x num_configs)
            offset: Payoff normalization offset
            scale: Payoff normalization scale
            epsilon: Small constant for numerical stability
            device: PyTorch device
        """
        self.num_players = num_players
        self.num_actions = num_actions
        self.device = device
        self.offset = offset
        self.scale = scale
        self.strategy_names = strategy_names
        
        # Set epsilon based on data type
        if epsilon is None:
            # Convert device to string if it's a torch.device object
            device_str = str(device) if isinstance(device, torch.device) else device
            self.epsilon = F32_EPSILON if device_str.startswith("cuda") else F64_EPSILON
        else:
            self.epsilon = epsilon
        
        # Convert to PyTorch tensors and move to device
        self.config_table = torch.tensor(config_table, dtype=torch.float32, device=device)
        self.payoff_table = torch.tensor(payoff_table, dtype=torch.float32, device=device)

        # Calculate number of profiles
        self.num_profiles = self.config_table.shape[0]

    @classmethod
    def from_payoff_function(cls, num_players, num_actions, payoff_function,  strategy_names, device="cpu",
                            ub=MAXIMUM_PAYOFF, lb=MINIMUM_PAYOFF):
        """
        creates a symmetric game from a payoff function
        input:
            num_players: Number of players
            num_actions: Number of actions/strategies
            payoff_function: Function that takes a tuple of actions and returns a payoff
            device: PyTorch device
            ub: Upper bound for payoffs
            lb: Lower bound for payoffs
        returns:
            SymmetricGame instance
        """
        configs = []
        for config in combinations_with_replacement(range(num_actions), num_players - 1):
            counter = Counter(config)
            profile = [counter.get(a, 0) for a in range(num_actions)]
            configs.append(profile)
        
        num_configs = len(configs)

        #initialize arrays for storage
        config_table = np.zeros((num_configs, num_actions))
        payoff_table = np.zeros((num_actions, num_configs))
        repeat_table = np.zeros(num_configs)

        #now we fill tables wih config data
        for c, config in enumerate(configs):
            config_table[c] = config
            repeat_table[c] = logmultinomial(*config)
            payoff_table[:, c] = payoff_function(config)
        
        #now normalize payoffs
        min_payoff = np.min(payoff_table)
        max_payoff = np.max(payoff_table)
        offset, scale = cls._set_scale(min_payoff, max_payoff, ub, lb) 

        #apply normalization
        normalized_payoffs = cls._normalize_payoffs(payoff_table, offset, scale)
        log_normalized_payoffs = np.log(normalized_payoffs)
        weighted_payoffs = log_normalized_payoffs + repeat_table
        
        return cls(
            num_players=num_players,
            num_actions=num_actions,
            config_table=config_table,
            payoff_table=weighted_payoffs,
            offset=offset,
            scale=scale,
            strategy_names=strategy_names,
            device=device
        )

    @staticmethod
    def _set_scale(min_payoff, max_payoff, ub, lb):
        """
        compute scale and offset for payoff normalization.
        
        input: 
            min_payoff: Minimum observed payoff
            max_payoff: Maximum observed payoff
            ub: Upper bound for normalized payoffs
            lb: Lower bound for normalized payoffs
        returns:
            Tuple (offset, scale)
        """
        # Avoid division by zero
        if max_payoff == min_payoff:
            offset = min_payoff
            scale = 1.0
        else:
            offset = min_payoff
            scale = max_payoff - min_payoff
        
        return offset, scale
    
    @staticmethod
    def _normalize_payoffs(payoffs, offset, scale):
        """
        normalize payoffs to [0, 1] range.
        
        input:
            payoffs: Raw payoffs
            offset: Offset to subtract
            scale: Scale to divide by
        returns:
            Normalized payoffs
        """
        norm_payoffs = (payoffs - offset) / scale
        # Ensure payoffs are in valid range
        norm_payoffs = np.clip(norm_payoffs, F64_EPSILON, 1.0)
        return norm_payoffs

    @staticmethod
    def _denormalize_payoffs(norm_payoffs, offset, scale):
        """
        convert normalized payoffs back to original scale.
        
        input:
            norm_payoffs: Normalized payoffs
            offset: Offset to add
            scale: Scale to multiply by
        returns:
            Denormalized payoffs
        """
        return norm_payoffs * scale + offset

    def deviation_payoffs(self, mixture):
        """
        Calculate the expected payoff for deviating to each pure strategy.
        
        Uses efficient vectorized computation with log payoffs.
        
        input:
            mixture: Strategy mixture
        returns:
            Expected payoffs for each deviation strategy
        """
        if not torch.is_tensor(mixture):
            mixture = torch.tensor(mixture, dtype=torch.float32, device=self.device)
        
        # Handle all possible tensor dimensions
        original_dim = len(mixture.shape)
        
        # For scalar (0D) tensors, reshape to 1D first
        if original_dim == 0:
            mixture = mixture.reshape(1)
            original_dim = 1
        
        # For 1D tensors, reshape to 2D for batch processing
        if original_dim == 1:
            mixture = mixture.reshape(-1, 1)
        
        # Now it's safe to get the batch size
        batch_size = mixture.shape[1]
        
        # Compute log mixture (with epsilon handling)
        log_mixture = torch.log(mixture + self.epsilon)
        
        # Calculate log multinomial coefficients
        configs = self.config_table.to(self.device)
        log_coeffs = torch.zeros((configs.shape[0], batch_size), device=self.device)
        
        for action in range(self.num_actions):
            log_coeffs = log_coeffs + configs[:, action:action+1] * log_mixture[action]
        
        # Calculate deviation payoffs
        deviation_payoffs = torch.zeros((self.num_actions, batch_size), device=self.device)
        
        # Calculate payoffs for each strategy from log-space payoffs
        for action in range(self.num_actions):
            # Mask profiles where the strategy has positive count
            has_action = self.config_table[:, action] > 0
            
            # Skip if no profiles have this action
            if not torch.any(has_action):
                continue
            
            # Get log payoffs and subtract log coefficients
            log_payoffs = self.payoff_table[action, has_action].to(self.device)
            log_probabilities = log_coeffs[has_action]
            
            # Calculate log terms for each profile
            log_terms = log_payoffs.unsqueeze(1) + log_probabilities
            
            # Sum terms in log space using the log-sum-exp trick
            max_term = torch.max(log_terms, dim=0, keepdim=True)[0]
            log_sum = max_term + torch.log(torch.sum(torch.exp(log_terms - max_term), dim=0))
            
            # Convert back from log space
            deviation_payoffs[action] = torch.exp(log_sum)
        
        # Denormalize payoffs to original scale
        deviation_payoffs = self._denormalize_payoffs(deviation_payoffs, self.offset, self.scale)
        
        # Return in appropriate shape based on original input
        if original_dim == 0:
            return deviation_payoffs.squeeze()
        elif original_dim == 1:
            return deviation_payoffs.squeeze(1)
        return deviation_payoffs

    def update_with_new_data(self, raw_data):
        """
        Update the game with new simulation data.
        
        input:
            raw_data: List of (player_id, strategy_name, payoff) tuples representing
                     agent-level data from a simulation
        """
        # First, convert raw_data into profiles and payoffs
        profile_payoffs = {}
        
        for player_id, strategy_name, payoff in raw_data:
            if isinstance(strategy_name, int):
                # If strategy_name is already an index
                strategy_idx = strategy_name
            else:
                # Convert strategy name to index
                if strategy_name not in self.strategy_names:
                    logging.warning(f"Unknown strategy: {strategy_name}, skipping")
                    continue
                strategy_idx = self.strategy_names.index(strategy_name)
            
            # Group by player_id to reconstruct profiles
            profile_key = player_id
            if profile_key not in profile_payoffs:
                profile_payoffs[profile_key] = {}
            
            profile_payoffs[profile_key][strategy_idx] = payoff
        
        # Reconstruct profiles and payoffs
        for profile_data in profile_payoffs.values():
            # Create the profile
            profile = torch.zeros(self.num_actions, device=self.device)
            for strategy_idx in profile_data.keys():
                profile[strategy_idx] += 1
            
            # Update the config_table and payoff_table
            self._add_profile(profile.tolist(), profile_data)
    
    def _add_profile(self, profile, strategy_payoffs):
        """
        Add a new profile and its payoffs to the game.
        
        input:
            profile: List representing the strategy counts
            strategy_payoffs: Dict mapping strategy indices to payoffs
        """
        # Convert to tensor for easier operations
        profile_tensor = torch.tensor(profile, dtype=torch.float32, device=self.device)
        
        # Check if this profile already exists
        for i in range(self.config_table.size(0)):
            if torch.all(self.config_table[i] == profile_tensor):
                # Profile exists, update payoffs by averaging
                for strategy_idx, payoff in strategy_payoffs.items():
                    # Only update if the strategy is used in this profile
                    if profile[strategy_idx] > 0:
                        # Average with existing payoff
                        existing_payoff = self.payoff_table[strategy_idx, i].item()
                        if existing_payoff != 0:  # If there's an existing payoff
                            new_payoff = (existing_payoff + payoff) / 2
                        else:
                            new_payoff = payoff
                        
                        # Update the payoff table
                        self.payoff_table[strategy_idx, i] = new_payoff
                return
        
        # Profile doesn't exist, add it
        # Update config_table
        new_config_table = torch.cat([
            self.config_table,
            profile_tensor.unsqueeze(0)
        ], dim=0)
        
        # Update payoff_table
        new_payoffs = torch.zeros(self.num_actions, 1, device=self.device)
        for strategy_idx, payoff in strategy_payoffs.items():
            if profile[strategy_idx] > 0:
                new_payoffs[strategy_idx, 0] = payoff
        
        new_payoff_table = torch.cat([
            self.payoff_table,
            new_payoffs
        ], dim=1)
        
        # Set new tables
        self.config_table = new_config_table
        self.payoff_table = new_payoff_table
    
    def restrict(self, strategy_indices):
        """
        Create a new game restricted to a subset of strategies.
        
        Parameters:
        -----------
        strategy_indices : torch.Tensor
            Indices of strategies to include in the restricted game
        
        Returns:
        --------
        SymmetricGame
            A new game with only the specified strategies
        """
        strategy_indices = strategy_indices.to(self.device)
        
        # Create a mask for valid profiles (those that only use strategies in strategy_indices)
        valid_profiles_mask = torch.zeros(self.config_table.size(0), dtype=torch.bool, device=self.device)
        
        for i in range(self.config_table.size(0)):
            profile = self.config_table[i]
            # Check if all players are using strategies in strategy_indices
            is_valid = True
            for j in range(self.num_actions):
                if profile[j] > 0 and j not in strategy_indices:
                    is_valid = False
                    break
            valid_profiles_mask[i] = is_valid
        
        # Get valid profiles and their payoffs
        valid_profiles = self.config_table[valid_profiles_mask]
        
        if len(valid_profiles) == 0:
            raise ValueError(f"No valid profiles found for strategies {strategy_indices.tolist()}")
        
        # Create new config table with only the selected strategies
        new_config_table = torch.zeros((valid_profiles.size(0), len(strategy_indices)), device=self.device)
        for i, profile_idx in enumerate(torch.nonzero(valid_profiles_mask).squeeze(1)):
            for j, strat_idx in enumerate(strategy_indices):
                new_config_table[i, j] = self.config_table[profile_idx, strat_idx]
        
        # Create new payoff table
        new_payoff_table = torch.zeros((len(strategy_indices), valid_profiles.size(0)), device=self.device)
        for i, profile_idx in enumerate(torch.nonzero(valid_profiles_mask).squeeze(1)):
            for j, strat_idx in enumerate(strategy_indices):
                new_payoff_table[j, i] = self.payoff_table[strat_idx, profile_idx]
        
        # Get the strategy names
        new_strategy_names = [self.strategy_names[idx.item()] for idx in strategy_indices]
        
        # Create the restricted game
        return SymmetricGame(
            num_players=self.num_players,
            num_actions=len(strategy_indices),
            config_table=new_config_table.cpu().numpy(),
            payoff_table=new_payoff_table.cpu().numpy(),
            strategy_names=new_strategy_names,
            offset=self.offset,
            scale=self.scale,
            device=self.device
        )

    def expected_payoff(self, mixture):
        """
        Calculate the expected payoff of a mixture.
        
        Parameters:
        -----------
        mixture : torch.Tensor
            Strategy mixture
            
        Returns:
        --------
        float : Expected payoff
        """
        dev_payoffs = self.deviation_payoffs(mixture)
        return torch.sum(mixture * dev_payoffs)
    
    def regret(self, mixture):
        """
        Calculate the regret of a mixture.
        
        Parameters:
        -----------
        mixture : torch.Tensor
            Strategy mixture
            
        Returns:
        --------
        float : Regret (maximum gain from deviation)
        """
        expected_payoff = self.expected_payoff(mixture)
        dev_payoffs = self.deviation_payoffs(mixture)
        return torch.max(dev_payoffs - expected_payoff)
    
    def deviation_gains(self, mixture):
        """
        Calculate the gains from deviating to each pure strategy.
        
        Parameters:
        -----------
        mixture : torch.Tensor
            Strategy mixture
            
        Returns:
        --------
        torch.Tensor : Deviation gains for each strategy
        """
        expected_payoff = self.expected_payoff(mixture)
        dev_payoffs = self.deviation_payoffs(mixture)
        return torch.max(dev_payoffs - expected_payoff, torch.tensor(0.0, device=self.device))
    
    def gain_gradients(self, mixture):
        """
        Calculate the gradients of the deviation gains.
        
        Parameters:
        -----------
        mixture : torch.Tensor
            Strategy mixture
            
        Returns:
        --------
        torch.Tensor : Gradients of deviation gains
        """
        # This is a simplified implementation - a more accurate one would compute the actual Jacobian
        eps = 1e-6
        gains = self.deviation_gains(mixture)
        gradients = torch.zeros((self.num_actions, self.num_actions), device=self.device)
        
        for i in range(self.num_actions):
            perturbed_mixture = mixture.clone()
            perturbed_mixture[i] += eps
            perturbed_mixture = perturbed_mixture / perturbed_mixture.sum()  # Renormalize
            
            perturbed_gains = self.deviation_gains(perturbed_mixture)
            gradients[i] = (perturbed_gains - gains) / eps
        
        return gradients
    
    def better_response(self, mixture, scale_factor=1.0):
        """
        Apply the better response operator to a mixture.
        
        Parameters:
        -----------
        mixture : torch.Tensor
            Strategy mixture
        scale_factor : float
            Scale factor for the better response
            
        Returns:
        --------
        torch.Tensor : Updated mixture
        """
        gains = self.deviation_gains(mixture) * scale_factor
        return (mixture + gains) / (1 + gains.sum())
    
    def __str__(self):
        """
        string representation of the game.
        
        returns:
            String representation
        """
        config_table = self.config_table.cpu().numpy()
        payoff_table = self.payoff_table.cpu().numpy()
        
        # Denormalize payoffs for display
        denorm_payoffs = np.exp(payoff_table) * self.scale + self.offset
        
        # Create a list of rows for tabulation
        rows = []
        for c in range(config_table.shape[0]):
            # Format strategy counts
            counts = [int(count) for count in config_table[c]]
            config_str = ", ".join([f"{self.strategy_names[i]}: {int(count)}" for i, count in enumerate(counts) if count > 0])
            
            # Format payoffs
            payoff_str = ", ".join([f"{self.strategy_names[i]}: {denorm_payoffs[i, c]:.2f}" for i in range(self.num_actions) if config_table[c, i] > 0])
            
            rows.append([c+1, config_str, payoff_str])
        
        # Create table
        table = tabulate(rows, headers=["Profile", "Strategy Counts", "Payoffs"], tablefmt="grid")
        
        return f"Symmetric Game with {self.num_players} players and {self.num_actions} strategies:\n{table}" 