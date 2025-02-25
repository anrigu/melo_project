import numpy as np
from collections import defaultdict

class Game:
    def __init__(self, strategy_names, profiles, payoffs, num_players):
        """
        A class to construct a full symmetric game with payoff data.
        
        Parameters:
        strategy_names : list of str
            Names of each strategy (e.g. ['LIT_ORDERBOOK', 'MELO']).
        profiles : list of list
            A list of profiles (integer counts of how many players choose each strategy).
        payoffs : list of list
            Payoff data corresponding to each profile (same shape as profiles).
        num_players : int
            Total number of players in the game.
        """
        
        self.strategy_names = strategy_names
        self.profiles = profiles
        self.payoffs = payoffs
        self.num_players = num_players
        
        # Create a dictionary for faster profile lookup - O(1) access
        self.profile_payoff_dict = {tuple(profile): payoff 
                                   for profile, payoff in zip(profiles, payoffs)}
        
        # Create proper numeric payoff matrices
        self.numeric_payoff_matrix = self._create_numeric_payoff_matrix()
        self.payoff_table = self._create_payoff_table()

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
    
    def get_payoff_for_profile(self, profile):
        """Get payoffs for a specific profile - O(1) operation"""
        profile_tuple = tuple(profile)
        return self.profile_payoff_dict.get(profile_tuple)
    
    def _create_numeric_payoff_matrix(self):
        """
        Creates a numpy array of the payoff data only.
        Used for numerical computations.
        """
        return np.array(self.payoffs)
    
    def _create_payoff_table(self):
        """
        Creates a formatted table representation with headers for display.
        """
        header = ["# " + strat for strat in self.strategy_names] + \
                ["Payoff (" + strat + ")" for strat in self.strategy_names]
        
        table = [header]
        for profile, payoff in zip(self.profiles, self.payoffs):
            table.append(profile + payoff)
            
        return table
    
    def get_payoff_matrix(self):
        """
        Returns the formatted payoff table for display.
        """
        return self.payoff_table
    
    def get_numeric_payoff_matrix(self):
        """
        Returns the numeric payoff matrix for calculations.
        """
        return self.numeric_payoff_matrix
    
    def add_profile(self, profile, payoff):
        """
        Add a new profile and its payoffs to the game.
        
        Parameters:
        profile : list
            Strategy counts for this profile
        payoff : list
            Payoffs for each strategy in this profile
        """
        profile_tuple = tuple(profile)
        
        # Check if profile already exists
        if profile_tuple in self.profile_payoff_dict:
            return False
        
        # Add new profile and payoffs
        self.profiles.append(profile)
        self.payoffs.append(payoff)
        self.profile_payoff_dict[profile_tuple] = payoff
        
        # Update matrices
        self.numeric_payoff_matrix = self._create_numeric_payoff_matrix()
        self.payoff_table = self._create_payoff_table()
        
        return True
    
    def update_profile_payoff(self, profile, new_payoff):
        """
        Update the payoff for an existing profile.
        
        Parameters:
        profile : list
            Strategy counts for the profile to update
        new_payoff : list
            New payoffs for each strategy in this profile
        """
        profile_tuple = tuple(profile)
        
        # Check if profile exists
        if profile_tuple not in self.profile_payoff_dict:
            return False
        
        # Find the index of this profile
        idx = self.profiles.index(list(profile))
        
        # Update payoffs
        self.payoffs[idx] = new_payoff
        self.profile_payoff_dict[profile_tuple] = new_payoff
        
        # Update matrices
        self.numeric_payoff_matrix = self._create_numeric_payoff_matrix()
        self.payoff_table = self._create_payoff_table()
        
        return True
        
    def update_payoffs(self, new_raw_data):
        """
        Updates the game's payoff matrix with new observed strategy profiles.
        Parameters:
            new_raw_data : list of lists
                New observed strategy profiles and their payoffs.
        Returns:
            prior_payoff_table : list
                The old payoff matrix before updates.
            new_payoff_table : list
                The updated payoff matrix.
        """
        try:
            prior_payoff_table = self.payoff_table.copy()
            new_strategy_names = set()
            for profile in new_raw_data:
                for _, strategy, _ in profile:
                    new_strategy_names.add(strategy)

            new_strategy_names = sorted(list(new_strategy_names))
            profile_dict = defaultdict(lambda: {"count": 0, "payoffs": defaultdict(list)})
            
            # Process new data
            for profile in new_raw_data:
                strat_count = tuple(sorted([
                    (strategy, sum(1 for _, s, _ in profile if s == strategy)) 
                    for strategy in new_strategy_names
                ]))
                profile_dict[strat_count]["count"] += 1
                for _, strategy, payoff in profile:
                    profile_dict[strat_count]["payoffs"][strategy].append(payoff)
            
            # Create profiles and payoffs from new data
            new_profiles = []
            new_payoffs = []
            for strat_count, data in profile_dict.items():
                profile = [count for _, count in strat_count]
                new_profiles.append(profile)
                expected_payoffs = [
                    np.mean(data["payoffs"][strat]) if data["payoffs"][strat] else 0 
                    for strat, _ in strat_count
                ]
                new_payoffs.append(expected_payoffs)
            
            # Create index map for existing profiles
            existing_profiles = {tuple(p): i for i, p in enumerate(self.profiles)}
            
            # Update or add profiles
            for profile, payoff in zip(new_profiles, new_payoffs):
                profile_tuple = tuple(profile)
                
                if profile_tuple in existing_profiles:  # if profile exists, average payoffs
                    idx = existing_profiles[profile_tuple]
                    self.payoffs[idx] = [
                        (self.payoffs[idx][i] + payoff[i]) / 2 
                        for i in range(len(payoff))
                    ]
                    self.profile_payoff_dict[profile_tuple] = self.payoffs[idx]
                else:  # if profile does not exist, add it
                    self.profiles.append(profile)
                    self.payoffs.append(payoff)
                    self.profile_payoff_dict[profile_tuple] = payoff
            
            # Update strategy names
            self.strategy_names = sorted(set(self.strategy_names) | set(new_strategy_names))
            
            # Regenerate matrices
            self.numeric_payoff_matrix = self._create_numeric_payoff_matrix()
            self.payoff_table = self._create_payoff_table()
            
            return prior_payoff_table, self.payoff_table
            
        except Exception as e:
            print(f"Error updating payoffs with new data: {e}")
            print(f"Returning original payoff matrix")
            return self.payoff_table, self.payoff_table