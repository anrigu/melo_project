import torch
import numpy as np
import logging
from typing import Tuple, List, Optional, Set, Dict, Any

from utils.eq_computation import find_equilibria
from utils.simplex_operations import filter_unique
from symmetric_game import SymmetricGame
from reductions.dpr import DPRGAME

logger = logging.getLogger(__name__)

class Quiesce:
    """
    quiesce algorithm for iterative strategy exploration in symmetric games.
    
    The algorithm works by:
    1. starting with a restricted set of strategies
    2. finding equilibria in the restricted game
    3. identifying beneficial deviations outside the restricted set
    4. adding these deviations to the strategy set
    5. repeating until no beneficial deviations are found
    """
    
    def __init__(self, 
                 game, 
                 initial_strategies=None,
                 regret_thresh=1e-3,
                 max_iters=10, 
                 eq_method='replicator_dynamics',
                 eq_kwargs=None):
        """
        initialize the Quiesce algorithm.
        
        parameters:
        game : SymmetricGame or DPRGame
            the full game object
        initial_strategies : torch.Tensor or None
            initial strategy indices to start with (if None, uses first strategy)
        regret_thresh : float
            regret threshold for determining beneficial deviations
        max_iters : int
            maximum number of iterations to run
        eq_method : str
            method to use for finding equilibria
        eq_kwargs : dict or None
            additional parameters to pass to find_equilibria
        """
        self.game = game 
        self.regret_thresh = regret_thresh
        self.max_iters = max_iters
        self.eq_method = eq_method
        self.eq_kwargs = eq_kwargs or {}
        
        # Initialize strategy indices
        if initial_strategies is None:
            # Start with just the first strategy
            self.strategy_indices = torch.tensor([0], device=self.game.device)
        else:
            self.strategy_indices = torch.tensor(initial_strategies, device=self.game.device)
            
        # Initialize the restricted game
        self.update_restricted_game()
    
    def update_restricted_game(self):
        """Update the restricted game with current strategy indices"""
        self.restricted_game = self.game.restrict(self.strategy_indices)
    
    def find_beneficial_deviations(self, equilibrium) -> Set[int]:
        """
        find beneficial deviations from the given equilibrium.
        
        parameters:
        equilibrium : torch.Tensor
            Equilibrium mixture in the restricted game
            
        returns:
        Set[int] : Indices of beneficial deviation strategies
        """
        # Convert equilibrium to full game format
        full_eq = self.convert_to_full_game(equilibrium) 
        # Expected payoff in equilibrium
        eq_payoff = self.game.expected_payoff(full_eq) 
        beneficial_deviations = set()
        # check each strategy not in the restricted set
        all_strats = set(range(self.game.num_actions))
        current_strats = set(self.strategy_indices.cpu().numpy())
        outside_strats = all_strats - current_strats
        for strat_idx in outside_strats:
            # calculate deviation payoff for this strategy
            deviation_payoff = self.game.deviation_payoffs(full_eq)[strat_idx]
            # if deviation gives higher payoff by more than threshold, it's beneficial
            if deviation_payoff > eq_payoff + self.regret_thresh:
                beneficial_deviations.add(strat_idx)
                logger.info(f"Found beneficial deviation: Strategy {strat_idx} with payoff {deviation_payoff:.6f} vs equilibrium payoff {eq_payoff:.6f}")
        
        return beneficial_deviations
    
    def convert_to_full_game(self, restricted_eq) -> torch.Tensor:
        """
        convert an equilibrium from the restricted game to the full game.
        
        parameters:
        restricted_eq : torch.Tensor
            Equilibrium in the restricted game
            
        returns:
        torch.Tensor : Equivalent equilibrium in the full game
        """
        full_eq = torch.zeros(self.game.num_actions, device=self.game.device)
        for i, idx in enumerate(self.strategy_indices):
            full_eq[idx] = restricted_eq[i]
        return full_eq
    
    def run(self) -> Tuple[torch.Tensor, int, List[torch.Tensor]]:
        """
        run the Quiesce algorithm.
        returns:
        Tuple[torch.Tensor, int, List[torch.Tensor]]:
            - Final equilibrium in full game
            - Total number of profiles explored
            - payoffs
            - List of all equilibria found (in restricted games) 
        """
        logger.info(f"starting Quiesce with {len(self.strategy_indices)} strategies")
        
        all_equilibria = []
        total_profiles = 0
        
        for iter_num in range(self.max_iters):
            logger.info(f"Quiesce iteration {iter_num+1}/{self.max_iters}")
            logger.info(f"Current strategy set: {self.strategy_indices.cpu().numpy()}")
            
            best_mixture, eq_candidates, regrets = find_equilibria(
                self.restricted_game, 
                method=self.eq_method,
                **self.eq_kwargs
            )
            
            # calculate actual regret for this equilibrium
            eqa_regret = self.restricted_game.regret(best_mixture)
            logger.info(f"Found equilibrium with regret {eqa_regret.item():.6f}")
            
            # keep track of number of profiles explored
            total_profiles += self.restricted_game.num_profiles
            
            # store equilibrium
            eqa = best_mixture
            
            # if regret is too high, not a good equilibrium 
            if eqa_regret > self.regret_thresh:
                logger.warning(f"Equilibrium has high regret: {eqa_regret.item():.6f}")
                break
                
            all_equilibria.append(eqa)

            # find new beneficial deviations 
            new_strategies = self.find_beneficial_deviations(eqa)
            if not new_strategies:
                logger.info("No beneficial deviations found. Quiesce complete.")
                break
            
            # Add new strategies and update restricted game
            self.strategy_indices = torch.cat([
                self.strategy_indices, 
                torch.tensor(list(new_strategies), device=self.game.device)
            ])
            self.update_restricted_game()
        
        # Convert equilibria from restricted game to full game
        full_eqa = self.convert_to_full_game(eqa)
        
        return full_eqa, total_profiles, all_equilibria
    
def find_unique_equilibria(equilibria, threshold=1e-2):
    """
    filter a list of equilibria to remove approximate duplicates.
    
    parameters:
    equilibria : List[torch.Tensor]
        List of equilibrium mixtures
    threshold : float
        Maximum L1 distance for two equilibria to be considered duplicates 
    returns:
    List[torch.Tensor] : Filtered list of unique equilibria
    """
    if not equilibria:
        return []
    
    if len(equilibria) <= 1:
        return equilibria
        
    equilibria_tensor = torch.stack(equilibria, dim=1)
    unique_eqs = filter_unique(equilibria_tensor, max_diff=threshold)
    
    return [unique_eqs[:, i] for i in range(unique_eqs.shape[1])]









