import numpy as np
from symmetric_game import SymmetricGame
from process_data import *
from game import AbstractGame
from utils.eq_computation import *
from utils.log_multimodal import *
from utils.random_functions import *
from utils.simplex_operations import *
import torch
from itertools import combinations_with_replacement, combinations

class DPRGAME(SymmetricGame):

    '''
    deviation preserving reduction game
    based offf of Eriks DPR implementation: 
    https://github.com/egtaonline/gameanalysis/blob/master/gameanalysis/reduction/deviation_preserving.py
    '''
    def __init__(self, full_game, reduced_players, device="cpu"):
        self.full_game = full_game
        self.reduced_players = reduced_players
        self.device = device

        assert isinstance(full_game, SymmetricGame), "full game should be Symmetric"
        assert reduced_players > 1, "reduced game must have at least 2 players"
        assert (full_game.num_players - 1) % (reduced_players - 1) == 0, "ratio of (full_players-1)/(reduced_players-1) must be an integer"
        assert full_game.num_players >= reduced_players, "reduced players must be <= full players"
        reduced_configs = self._generate_configs(reduced_players, self.full_game.num_actions)
        config_table = torch.zeros((len(reduced_configs), self.full_game.num_actions), 
                                    device=self.device)
        
        for i, conf, in enumerate(reduced_configs):
            config_table[i] = torch.tensor(conf, device=self.device)

        payoff_table = self._compute_reduced_payoffs(config_table)

        #init the symmetric game data structure for this reduced game. 
        super().__init__(
            num_players=reduced_players,
            num_actions=full_game.num_actions,
            config_table=config_table,
            payoff_table=payoff_table,
            strategy_names=full_game.strategy_names,
            device=device
        )

    '''
    def _generate_configs(self, num_players, num_actions):
        
        
        
        # Generate all combinations with replacement
        combos = list(combinations_with_replacement(range(num_actions), num_players))
        
        # Convert to configuration format using sparse tensor operations
        indices = torch.tensor([[i, val] for i, combo in enumerate(combos) 
                            for val in combo], device=self.device)
        
        values = torch.ones(indices.shape[0], device=self.device)
        configs = torch.sparse_coo_tensor(
            indices.t(), values, 
            (len(combos), num_actions)
        ).to_dense()
    
    
    return configs
    '''
    def _generate_configs(self, num_players, num_actions):
        '''
        generate configurations for the reduced game that have corresponding
        valid data in the full game.
        instead of generating all theoretical profiles, this only includes
        profiles that can be mapped to existing data in the full game.
        '''
        full_game = self.full_game
        reduced_players = num_players
        scale_factor = (full_game.num_players - 1) / (reduced_players - 1)
        
        full_configs = full_game.config_table
        
        reduced_configs_set = set()
        
        for full_config in full_configs:
            for strat_idx in range(num_actions):
                if full_config[strat_idx] > 0:
                    dev_config = full_config.clone()
                    dev_config[strat_idx] -= 1
                    
                    reduced_dev_config = torch.round(dev_config / scale_factor).long()
                    
                    total_players = reduced_dev_config.sum().item()
                    if total_players != reduced_players - 1:
                        probs = dev_config.float() / max(dev_config.sum().item(), 1)
                        while total_players < reduced_players - 1:
                            idx = torch.multinomial(probs, 1).item()
                            reduced_dev_config[idx] += 1
                            total_players += 1
                        while total_players > reduced_players - 1:
                            valid_indices = (reduced_dev_config > 0).nonzero().flatten()
                            if len(valid_indices) == 0:
                                break  
                            idx = valid_indices[torch.multinomial(
                                torch.ones(len(valid_indices), device=self.device), 1
                            ).item()]
                            reduced_dev_config[idx] -= 1
                            total_players -= 1
                    
                    reduced_config = reduced_dev_config.clone()
                    reduced_config[strat_idx] += 1
                    
                    reduced_configs_set.add(tuple(reduced_config.cpu().tolist()))
        
        if not reduced_configs_set:
            return torch.zeros((0, num_actions), device=self.device)
        
        reduced_configs = torch.tensor(
            list(reduced_configs_set), 
            dtype=torch.int64, 
            device=self.device
        )
        
        return reduced_configs
        
    def _compute_reduced_payoffs(self, config_table):
            '''
            compute payoffs for reduced game use the DPR mapping

            basic steps:
            1. map to the full game deviations 
            2. compute the expected payoffs using full game
            3. transform to log-sapce for syymetric game (and exponentiate in vice versa to get true payoffs)
            '''

            full_game = self.full_game
            device = full_game.device

            #make the payoff table 
            payoff_table = torch.zeros((full_game.num_actions, len(config_table)),
                                       device=device)
            
            for config_index, config in enumerate(config_table):
                for strat_idx in range(full_game.num_actions):
                    if config[strat_idx] > 0: #strat is in that profile
                        expand_payoffs = self._expand_to_full_game(config, strat_idx)
                        payoff_table[strat_idx, config_index] = expand_payoffs

            return payoff_table
    
         

    def _find_config_index(self, config_array):
            '''
            find the index of a profile in the full game's table
            '''
            config_tensor = torch.tensor(config_array, device=self.full_game.device)
            for i, config in enumerate(self.full_game.config_table):
                if torch.all(config == config_tensor):
                    print(f"Found profile: {i}")
                    return i
            return -1 
    def _expand_to_full_game(self, reduced_config, strategy_idx):
            '''
            map reduced game profile and strat to full game payoffs
            basically this is the dpr mapping logic
            '''

            config = reduced_config.clone().cpu().numpy()

            #compute the player counts 
            full_players = self.full_game.num_players
            reduced_players = self.reduced_players
            scale_factor = (full_players - 1) / (reduced_players - 1)

            #now we do the full game config
            #remove one player from strat we're getting payoff for (as this is the focal agent)
            config[strategy_idx] -= 1

            #now we do the scalling of the remaining players to the full game size
            scaled_config = config * scale_factor
            scaled_config = np.round(scaled_config)

            while sum(scaled_config) < full_players - 1:
                #here we add back in the remaining players but proportuonally 
                probs = config / sum(config) if sum(config) > 0 else np.ones(len(config)) / len(config)
                idx = np.random.choice(range(len(config)), p=probs)
                scaled_config[idx] += 1

            #now we add focal player back to the strategy
            scaled_config[strategy_idx] += 1

            #finally look up payoff in the full game
            #convert scaled_config to the format for the full game

            full_game_config_idx = self._find_config_index(scaled_config)

            if full_game_config_idx >= 0:
                return self.full_game.payoff_table[strategy_idx, full_game_config_idx]
            else: #TODO: check if we should interpotlate from similiar pofiles 
                return self._interpolate_payoff(scaled_config, strategy_idx)
            
    def _interpolate_payoff(self, config, strategy_idx):
            '''
            interpolates the payoff from similiar profile if say the exact profile does not
            exist in the true empirical game
            '''
            similarities = []
            payoffs = []

            for i, full_profile in enumerate(self.full_game.config_table):
                #skip configurations where strategy i is not used
                if full_profile[strategy_idx] == 0:
                    continue

                config_tensor = torch.tensor(config, device=self.full_game.device)
                similarity = 1.0 / (1.0 + torch.sum(torch.abs(full_profile - config_tensor)))

                similarities.append(similarity)
                payoffs.append(self.full_game.payoff_table[strategy_idx, i])

            if not similarities:
                #fallback if no similar configurations found
                return torch.log(torch.tensor(0.5, device=self.full_game.device))
            
            #weifghted avg
            similarities_tensor = torch.tensor(similarities, device=self.full_game.device)
            return torch.sum(similarities_tensor * torch.stack(payoffs)) / torch.sum(similarities_tensor)
        
    def expand_mixture(self, reduced_mixture):
            '''
            map reduced game mixed strat to the full empirical game
            remember in DPR the mixed strategies are preserved, ie. the same mixed strategy 
            is an equilibrium in both the reduced and full games.
            '''

            return reduced_mixture
        
    def find_equilibrium(self, method="replicator_dynamics", logging=True, **kwargs):
            '''
            compute equilibirum for the reduced game using method from eq_computation

            see eq_computation.py for algorithms and inputs to use.
            '''
            eq_mixture, _, _ = find_equilibria(self, method=method, logging=logging, **kwargs)
            return eq_mixture







