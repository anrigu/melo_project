import numpy as np
from collections import defaultdict

import numpy as np
import torch
from abc import ABC, abstractmethod
from itertools import combinations_with_replacement
from collections import Counter
from marketsim.math.simplex_operations import logmultinomial, simplex_normalize
from torch.nn.functional import pad

# Constants matching Bryce's implementation 
#NOTE: might need to change
MAXIMUM_PAYOFF = 1e5  # all payoffs are standardized to the (MIN,MAX) range
MINIMUM_PAYOFF = 1e-5  # for numerical stability and to simplify parameter tuning
F32_EPSILON = np.finfo(np.float32).eps
F64_EPSILON = np.finfo(np.float64).eps


class AbstractGame(ABC):
    """
    abstract base class for all games.
    """
    @abstractmethod
    def deviation_payoffs(self, mixture):
        '''
        calculates expected payoff for each strategy against a mixture
        arguments:
            mix: strategy mixture (this is a probability distribution)
        returns:
            expected payoffs for deviating to each pure strategy
        '''
        pass

    def deviation_gains(self, mixture):
        """
        calculate the gain from deviating to each pure strategy
        args:
            mixture: strategy mixture (distribution over strategies)
        returns:
            expected gains for deviating to each pure strategy
        """

        if not torch.is_tensor(mixture):
            mixture = torch.tensor(mixture, dtype=torch.float32, device=self.device)


        is_vector = len(mixture.shape) == 1
        if is_vector:
            mixture = mixture.reshape(-1, 1)

        #calculate deviation payoffs
        dev_payoffs = self.deviation_payoffs(mixture)
        #compoute the mixture of expected values
        mixture_expectations = torch.sum(mixture * dev_payoffs, dim=0)

        gains = torch.clamp(dev_payoffs - mixture_expectations, min=0)
        if is_vector:
            return gains.squeeze(1)
        return gains
    
    def regret(self, mixture):
        """
        gets the regret for a mixture (ie. maximum gain from deviation)
        inputs:
            mixture: Strategy mixture
            
        returns:
            Maximum regret
        """
        gains = self.deviation_gains(mixture)
        return torch.max(gains, dim=0)[0]
    
    def best_responses(self, mixture, atol=1e-10):
        """
        find the best responses to a mixture.
        args:
            mixture: Strategy mixture
            atol: Absolute tolerance for considering strategies as best responses 
        returns:
            Boolean tensor indicating which strategies are best responses
        """
        dev_pays = self.deviation_payoffs(mixture)
        max_payoffs = torch.max(dev_pays, dim=0, keepdim=True)[0]
        return torch.isclose(dev_pays, max_payoffs, atol=atol)
    

    def better_response(self, mixture, scale_factor=1.0):
        '''
        This function implements the better response algorithm from Bryce's paper
        For use in Scarf's simplicial subdivision algorithm.
        https://arxiv.org/abs/2207.10832
        args:
            mixture: Strategy mixture
            scale_factor: Scale factor for gains
            
        returns:
            Better response mixture
        '''
        if not torch.is_tensor(mixture):
            mixture = torch.tensor(mixture, dtype=torch.float32, device=self.device)
            
        
        is_vector = len(mixture.shape) == 1
        if is_vector:
            mixture = mixture.reshape(-1, 1)  # Convert to column vector
            
        # calculate gains
        gains = torch.clamp(self.deviation_gains(mixture), min=0) * scale_factor
        
        # calculate better response
        better_resp = (mixture + gains) / (1 + torch.sum(gains, dim=0, keepdim=True))
        
        # return in the appropriate shape
        if is_vector:
            return better_resp.squeeze(1)
        return better_resp
    
    def filter_regrets(self, mixtures, threshold=1e-3, sorted=False):
        """
        filter mixtures based on regret.
        args:
            mixtures: Matrix of mixtures
            threshold: Regret threshold
            sorted: Whether to sort by regret
        returns:
            Filtered mixtures
        """
        if not torch.is_tensor(mixtures):
            mixtures = torch.tensor(mixtures, dtype=torch.float32, device=self.device)
            
        #compute regrets
        mixture_regrets = self.regret(mixtures)
        
        # filter by threshold
        #NOTE: might need to change the threshold
        below_threshold = mixture_regrets < threshold
        filtered_mixtures = mixtures[:, below_threshold]
        filtered_regrets = mixture_regrets[below_threshold]
        
        #we end by sorting by regret
        if sorted and len(filtered_regrets) > 0:
            sort_idx = torch.argsort(filtered_regrets)
            filtered_mixtures = filtered_mixtures[:, sort_idx]
            
        return filtered_mixtures
    

