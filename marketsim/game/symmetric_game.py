import numpy as np
from collections import defaultdict

import numpy as np
import torch
from abc import ABC, abstractmethod
from itertools import combinations_with_replacement
from collections import Counter

from marketsim.custom_math.simplex_operations import logmultinomial, simplex_normalize
from torch.nn.functional import pad
from marketsim.game.abstract_game import AbstractGame, MINIMUM_PAYOFF, MAXIMUM_PAYOFF, F32_EPSILON, F64_EPSILON
from tabulate import tabulate  
import math



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
            self.epsilon = F32_EPSILON if device.startswith("cuda") else F64_EPSILON
        else:
            self.epsilon = epsilon
        
        # Convert to PyTorch tensors and move to device
        self.config_table = torch.tensor(config_table, dtype=torch.float32, device=device)
        self.payoff_table = torch.tensor(payoff_table, dtype=torch.float32, device=device)


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
            strategy_names=strategy_names, 
            offset=offset,
            scale=scale,
            device=device
        )
    
    @staticmethod
    def _set_scale(min_val, max_val, ub=MAXIMUM_PAYOFF, lb=MINIMUM_PAYOFF):
        """
        set the scale and offset for payoff normalization
        inputs:
            min_val: Minimum payoff value
            max_val: Maximum payoff value
            ub: Upper bound
            lb: Lower bound
        outputs:
            offset: Normalization offset
            scale: Normalization scale
        """
        range_val = max_val - min_val
        if range_val == 0:
            # if all payoffs are identical, use default scaling
            scale = 1.0
            offset = min_val - lb
        else:
            scale = (ub - lb) / range_val
            offset = min_val - lb / scale
        return offset, scale
    
    @staticmethod
    def _normalize_payoffs(payoff_table, offset, scale):
        return (payoff_table - offset) * scale
    
    @staticmethod
    def _denormalize_payoffs(payoff_table, offset, scale):
        return payoff_table / scale + offset
    
    def pure_payoffs(self, profile): 
        '''
        calculates the payoffs for a pure strategy profile
        inputs:
            profile: strategy counts
        returns:
            payoffs for each strategy
        '''
        if not torch.is_tensor(profile):
            profile = torch.tensor(profile, dtype=torch.float32, device=self.device)
        
        config_match = torch.all(self.config_table == profile.reshape(1, -1), dim=1)
        
        if not torch.any(config_match):
            raise ValueError(f"Profile {profile} not found in configuration table")
        
        config_idx = torch.nonzero(config_match)[0].item()
        
        # Get payoffs and unweight them
        weighted_payoffs = self.payoff_table[:, config_idx]
        repeats = logmultinomial(*profile.detach().cpu().numpy())
        normalized_payoffs = torch.exp(weighted_payoffs - repeats)
        
        # denormalize if needed
        return self._denormalize_payoffs(normalized_payoffs, self.offset, self.scale)
    
    def deviation_payoffs(self, mixture):
        """
        calculate the expected payoff for each pure strategy against a mixture.
        inputs:
            mixture: Strategy mixture (probability distribution)
            
        Returns:
            Expected payoffs for deviating to each pure strategy
        """
        if not torch.is_tensor(mixture):
            mixture = torch.tensor(mixture, dtype=torch.float32, device=self.device)
            
        is_vector = len(mixture.shape) == 1
        if is_vector:
            mixture = mixture.reshape(-1, 1)
        
        # [num_configs, num_mixtures], log probs
        log_mixture = torch.log(mixture + self.epsilon)
        log_config_probs = self.config_table @ log_mixture  # Matrix multiplication
        
        # calculate deviation payoffs for each action
        # Shape: [num_actions, num_mixtures] (num actions is the same as num stratergies, im just stupid)
        dev_pays = torch.zeros((self.num_actions, mixture.shape[1]), device=self.device)
        #for all the actions, compute expected payoff
        for s in range(self.num_actions):
            # Expand the payoff table to match dimensions
            # payoff_table[s] shape: [num_configs]
            # log_config_probs shape: [num_configs, num_mixtures]
            expanded_payoffs = self.payoff_table[s].unsqueeze(1)  # Shape: [num_configs, 1]
            # result shape: [num_configs, num_mixtures]
            payoff_contributions = torch.exp(expanded_payoffs + log_config_probs)

            #sum over configurations
            dev_pays[s] = torch.sum(payoff_contributions, dim=0)
        
        if is_vector:
            return dev_pays.squeeze(1)
        return dev_pays
    
    def deviation_derivatives(self, mixture):
        ''' 
        compute the jacobian of deviation payoffs 
        NOTE: this is for gradient based methods
        we might need this later?
        input: 
            mixture: strategy mixture 
        returns:
            jacobian of deviation payoffs 
            (num_actions x num_actions x num_mixtures)
        '''
        if not torch.is_tensor(mixture):
            mixture = torch.tensor(mixture, dtype=torch.float32, device=self.device)
        
        is_vector = len(mixture.shape) == 1
        if is_vector:
            mixture = mixture.reshape(-1, 1)
        
        #first lets calculate the deviation payoffs 
        dev_pays = self.deviation_payoffs(mixture)
        #initialize jacobian
        num_mixtures = mixture.shape[1]
        jac = torch.zeros((self.num_actions, self.num_actions, num_mixtures), device=self.device)

        #add in that epsilon fella for numerical stability
        log_mixture = torch.log(mixture + self.epsilon)

        #calcualte the jacobian for each mixture 
        for m in range(num_mixtures):
            #recalculate probabilities for each mixture
            log_config_probs = self.config_table @ log_mixture[:, m:m+1]
            #next we compute the derivatives for each action matchup: 
            for a1 in range(self.num_actions):
                for a2 in range(self.num_actions):
                    weighted_derivative = 0
                    for c in range(self.config_table.shape[0]):
                        weight = torch.exp(self.payoff_table[a1, c] + log_config_probs[c])
                        derivative = self.config_table[c, a2] / mixture[a2, m]
                        weighted_derivative += weight * derivative
                    
                    jac[a1, a2, m] = weighted_derivative
        if is_vector:
            return jac[:, :, 0]
        return jac
    
    
    def gain_gradients(self, mixture):
        """
        computes gradients of deviation gains with respect to the mixture.
        #NOTE: for gradient based methods we may need later
        inputes:
            mixture: strategy mixture
        returns:
            gradients of deviation gains
        """
        if not torch.is_tensor(mixture):
            mixture = torch.tensor(mixture, dtype=torch.float32, device=self.device)
        
        is_vector = len(mixture.shape) == 1
        
        if is_vector:
            mixture = mixture.reshape(-1, 1)
        
        dev_pays = self.deviation_payoffs(mixture)
        mixture_expectations = torch.sum(mixture * dev_pays, dim=0)

        dev_jac = self.deviation_derivatives(mixture)

        #get utility gradients
        util_gradients = torch.zeros((self.num_actions, mixture.shape[1]), device=self.device)
        for s in range(self.num_actions):
            util_gradients[s] = torch.sum(mixture * dev_jac[:, s], dim=0)

        util_gradients += dev_pays

        #now we compute the gain jacobian matrix
        gain_jac = torch.zeros_like(dev_jac)

        for s in range(self.num_actions):
            for a in range(self.num_actions):
                gain_jac[s, a] = dev_jac[a, s] - util_gradients[s]

        #zero out entries where deviation doesn't improve payoff
        mask = dev_pays < mixture_expectations
        for a in range(self.num_actions):
            gain_jac[:, a, mask[a]] = 0
        
        #calculate gain gradients
        gain_grads = torch.sum(gain_jac, dim=1)
        
        if is_vector:
            return gain_grads.squeeze(1)
        return gain_grads
    
    #TODO: debugg
    def print_full_heuristic_payoff_table(self):
        """
        Print the full heuristic payoff table in a readable format, showing:
        - Strategy profiles (counts of each strategy)
        - Payoffs for each strategy in each profile
        """
        
        denormalized_payoffs = []
        for s in range(self.num_actions):
            strat_payoffs = []
            for c in range(self.config_table.shape[0]):
                weighted_payoff = self.payoff_table[s, c].item()
                repeats = logmultinomial(*self.config_table[c].detach().cpu().numpy())
                normalized_payoff = math.exp(weighted_payoff - repeats)
                actual_payoff = normalized_payoff / self.scale + self.offset
                strat_payoffs.append(actual_payoff)
            denormalized_payoffs.append(strat_payoffs)
        
        headers = ["Profile"]
        for name in self.strategy_names:
            headers.append(f"{name} Count")
        for name in self.strategy_names:
            headers.append(f"{name} Payoff")
        
        rows = []
        for c in range(self.config_table.shape[0]):
            row = [f"Profile {c+1}"]
            for s in range(self.num_actions):
                row.append(int(self.config_table[c, s].item()))
            for s in range(self.num_actions):
                if self.config_table[c, s] > 0:
                    row.append(f"{denormalized_payoffs[s][c]:.4f}")
                else:
                    row.append("N/A")
            rows.append(row)
        
        print(tabulate(rows, headers=headers, tablefmt="pipe"))


    #TODO: check this
    def compute_statisitcs_and_covariant(self):
        '''
        compute the mean and covariance of the payoff table
        '''
        mean_payoff = torch.mean(self.payoff_table, dim=1)
        cov_payoff = torch.cov(self.payoff_table) 
        return mean_payoff, cov_payoff
    
   

    def update_with_new_data(self, raw_data):
        """
        Update the game with new data, potentially including new strategies.
        
        inputs:
            raw_data : list of lists
                Each sublist represents a strategy profile containing agent-level data.
                Format: [(player_id, strategy_name, payoff), ...]
        
        Returns:
            None (updates the game in place)
        """
        new_strategy_names = set()
        for profile in raw_data:
            for _, strategy, _ in profile:
                new_strategy_names.add(strategy)
        
        existing_strategies = set(self.strategy_names)
        completely_new_strategies = new_strategy_names - existing_strategies
        
        merged_strategy_names = self.strategy_names.copy()
        merged_strategy_names.extend(sorted(list(completely_new_strategies)))
        num_new_actions = len(merged_strategy_names)
        
        strategy_to_index = {name: i for i, name in enumerate(merged_strategy_names)}
        old_strategy_to_index = {name: i for i, name in enumerate(self.strategy_names)}
        
        profile_dict = defaultdict(lambda: {"count": 0, "payoffs": [[] for _ in range(num_new_actions)]})

        for profile in raw_data:
            strat_counts = [0] * num_new_actions
            for _, strategy, _ in profile:
                strat_idx = strategy_to_index[strategy]
                strat_counts[strat_idx] += 1
            
            strat_counts_tuple = tuple(strat_counts)
            
            profile_dict[strat_counts_tuple]["count"] += 1
            for _, strategy, payoff in profile:
                strat_idx = strategy_to_index[strategy]
                profile_dict[strat_counts_tuple]["payoffs"][strat_idx].append(float(payoff))
        
        configs = list(profile_dict.keys())
        num_new_configs = len(configs)
        
        new_config_table = np.zeros((num_new_configs, num_new_actions))
        new_raw_payoff_table = np.zeros((num_new_actions, num_new_configs))
        
        for c, config in enumerate(configs):
            new_config_table[c] = config
            
            for strat_idx in range(num_new_actions):
                if config[strat_idx] > 0:  # Only if strategy was used
                    payoffs = profile_dict[config]["payoffs"][strat_idx]
                    if payoffs:
                        new_raw_payoff_table[strat_idx, c] = np.mean(payoffs)
        
        if completely_new_strategies:
            expanded_config_table = np.zeros((self.config_table.shape[0], num_new_actions))
            
            for old_idx, strat_name in enumerate(self.strategy_names):
                new_idx = strategy_to_index[strat_name]
                expanded_config_table[:, new_idx] = self.config_table[:, old_idx].cpu().numpy()
            
            expanded_payoff_table = np.zeros((num_new_actions, self.payoff_table.shape[1]))
            for old_idx, strat_name in enumerate(self.strategy_names):
                new_idx = strategy_to_index[strat_name]
                expanded_payoff_table[new_idx, :] = self.payoff_table[old_idx, :].cpu().numpy()
            
            merged_config_table = np.vstack([expanded_config_table, new_config_table])
            merged_payoff_table = np.hstack([expanded_payoff_table, new_raw_payoff_table])
        else:
            merged_config_table = np.vstack([self.config_table.cpu().numpy(), new_config_table])
            merged_payoff_table = np.hstack([self.payoff_table.cpu().numpy(), new_raw_payoff_table])
        
        min_payoff = np.min(new_raw_payoff_table)
        max_payoff = np.max(new_raw_payoff_table)
        
        new_normalized_payoffs = (new_raw_payoff_table - self.offset) / self.scale
        
        epsilon = self.epsilon.cpu().numpy() if torch.is_tensor(self.epsilon) else self.epsilon
        new_normalized_payoffs = np.clip(new_normalized_payoffs, epsilon, 1.0)
        
        new_log_payoffs = np.log(new_normalized_payoffs)
        
        if completely_new_strategies:
            for old_idx, strat_name in enumerate(self.strategy_names):
                new_idx = strategy_to_index[strat_name]
                merged_payoff_table[new_idx, -num_new_configs:] = new_log_payoffs[new_idx, :]
        else:
            merged_payoff_table[:, -num_new_configs:] = new_log_payoffs
        
        self.strategy_names = merged_strategy_names
        self.num_actions = len(merged_strategy_names)
        self.config_table = torch.tensor(merged_config_table, dtype=torch.float32, device=self.device)
        self.payoff_table = torch.tensor(merged_payoff_table, dtype=torch.float32, device=self.device)
        
        print(f"Game updated: {len(completely_new_strategies)} new strategies added, {num_new_configs} new profiles added")
        return None
        
    
    
    
    '''
    Statistics notes:
    variance reduction techniques:
        - one way to do this is use control variates:
            - basically exploit the correlation between a payoffs and observable features

    Spurious Equilibria:
    - Key challenge of EGTA is empirical game might not align with true underlying game
    - A profile could be in the empirical game and be an equilibrium there and not in true game
    - vice versa is also true.
    - precision: of all the profiles you find to be equilibria in the empirical game, how many truly are equilibria in the original game
    - recall: of all true equilibria in the original game, how many appear as equilibria in the empirical model.

    Sampling and approximation bounds:
    - in a uniform approximation of a game, is within some error epsilon of its true payoff, you can get this if:
    - any true equilibria of the real game, will be at most a small bounded distance from equilibrium of the empirical game
    - we can control how how error gets into the regret or equilibrium calculation bc a uniform error of epsilon in payoffs translates into bounded shift in regret. 
    - we can yield an epsilon accurate game if we collect enough samples per strategy profile:
        - Hoeffdings inequality would depend on the requirement that depends only on the maximum range of payoffs
        - Bennett's inequality can tighten these estimates if you can bound or estimate the variance of the payoffs, rather than just abolsute range

    - can wee how many samples per profile we need to make sure in the worst case all payoff estimates are accurate within epsilon with high probability 

    Measuring Uncertainty in results:
    - can use bootstrapping to build distribution of outcomes for a profile
    
    Set up Bayesian optimization or gaussian samplimg to adaptively decide which to 
    sample more

    '''
    
  














    
        


