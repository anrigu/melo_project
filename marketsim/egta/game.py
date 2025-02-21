import numpy as np
from collections import defaultdict

class Game:
    def __init__(self, strategy_names, profiles, payoffs):
        """
        this is the class to construct a full symmetric game, with a payoff matrix
        given a strategy profile and payoff data.

        parameters:
        role_names : list of str
            Names of each role (e.g. ['Trader'], or ['Market Maker', 'Trader']).
        num_role_players : list or array of int
            Number of players for each role.
        strat_names : list of list
            For each role, a list of strategy names (e.g. [['LIT_ORDERBOOK', 'MELO']]).
        profiles : list of list
            A list of profiles (integer counts of how many players choose each strategy).
        payoffs : list of list
            Payoff data corresponding to each profile (same shape as profiles).
        """
        self.strategy_names = strategy_names
        self.profiles = profiles
        self.payoffs = payoffs
        
        self._num_strategies = len(strategy_names)
        self._num_profiles = len(profiles)
        self._num_payoffs = len(payoffs)

        self.payoff_matrix = self.set_payoff_matrix()

    @property
    def num_strategies(self):
        return len(self.strategy_names)
    
    @property
    def num_profiles(self):
        return len(self.profiles)

    def is_empty(self):
        return self.num_profiles == 0

    def get_profiles(self):
        return self.profiles

    def get_payoffs(self):
        return self.payoffs
    
    
    def update_payoffs(self, new_raw_data):
        """
        updates the game's payoff matrix with new observed strategy profiles.
        Parameters:
            new_raw_data : list of lists
                New observed strategy profiles and their payoffs.
        Returns:
            prior_payoff_matrix : list
                The old payoff matrix before updates.
            new_payoff_matrix : list
                The updated payoff matrix.
        """
        try:
            prior_payoff_matrix = self.payoff_matrix
            new_strategy_names = set()
            for profile in new_raw_data:
                for _, strategy, _ in profile:
                    new_strategy_names.add(strategy)

            new_strategy_names = sorted(list(new_strategy_names))
            profile_dict = defaultdict(lambda: {"count": 0, "payoffs": defaultdict(list)})
            for profile in new_raw_data:
                strat_count = tuple(sorted([
                    (strategy, sum(1 for _, s, _ in profile if s == strategy)) for strategy in new_strategy_names
                ]))
                profile_dict[strat_count]["count"] += 1
                for _, strategy, payoff in profile:
                    profile_dict[strat_count]["payoffs"][strategy].append(payoff)
            new_profiles = []
            new_payoffs = []
            for strat_count, data in profile_dict.items():
                new_profiles.append([count for _, count in strat_count])  # Convert tuple back to list
                expected_payoffs = [np.mean(data["payoffs"][strat]) if data["payoffs"][strat] else 0 
                                    for strat, _ in strat_count]
                new_payoffs.append(expected_payoffs)
            updated_profiles = {tuple(p): i for i, p in enumerate(self.profiles)}
            for profile, payoff in zip(new_profiles, new_payoffs):
                profile_tuple = tuple(profile)
                
                if profile_tuple in updated_profiles: #if profile already exists, average payoffs
                    index = updated_profiles[profile_tuple]
                    self.payoffs[index] = [(self.payoffs[index][i] + payoff[i]) / 2 for i in range(len(payoff))]
                else: #if profile does not exist, add it to the game
                    self.profiles.append(profile)
                    self.payoffs.append(payoff)
            self.strategy_names = sorted(set(self.strategy_names) | set(new_strategy_names))
            self.payoff_matrix = self.set_payoff_matrix()
            return prior_payoff_matrix, self.payoff_matrix
        except Exception as e:
            print(f"Error updating payoffs with new data: {e}")
            print(f"Returning original payoff matrix")
            return self.payoff_matrix

    def set_payoff_matrix(self):
        """
        constructs the updated payoff matrix.
        Returns:
            list : The payoff matrix as a list of lists.
        """
        header = ["# " + strat for strat in self.strategy_names] + 
        ["Payoff (" + strat + ")" for strat in self.strategy_names]
        payoff_matrix = [header]

        for profile, payoff in zip(self.profiles, self.payoffs):
            payoff_matrix.append(profile + payoff)

        return payoff_matrix

    def get_payoff_matrix(self):
        return self.payoff_matrix
    
  
 
   