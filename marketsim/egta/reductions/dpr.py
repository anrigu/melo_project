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

        def _generate_configs(self, num_players, num_actions):
            '''
            generate possible configs for reduced game
            '''
            configs = []
            for combo in combinations_with_replacement(range(num_actions), num_players):
                config = [0] * num_actions
                for id in combo:
                    config[id] += 1

                configs.append(conf)
            
            return configs
        
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
                for strat_idx in (full_game.num_actions):
                    if config[strat_idx] > 0: #strat is in that profile
                        expand_payoffs = self._expand_to_full_game(config, strat_idx)
                        payoff_table[strat_idx, config_index] = expand_payoffs

            return payoff_table
    
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

            full_game_config_idx = self._find_config_idx(scaled_config)

            if full_game_config_idx >= 0:
                return self.full_game.payoff_table[strategy_idx, full_game_config_idx]
            else: #TODO: check if we should interpotlate from similiar pofiles 
                return self._interpolate_payoff(scaled_config, strategy_idx)
            

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
        
        def find_equilibrium(self, method="replicator_dynamics", **kwargs):
            '''
            compute equilibirum for the reduced game using method from eq_computation

            see eq_computation.py for algorithms and inputs to use.
            '''
            eq_mixture, _, _ = find_equilibria(self, method=method, **kwargs)
            return eq_mixture
        






