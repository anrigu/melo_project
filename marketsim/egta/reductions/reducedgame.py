from egta.game import Game
import numpy as np

class ReducedGame(Game):
    """
    Extension of Game used to represent a deviation-preserving reduced game.
    Includes methods to map between the full and reduced games.
    """
    def __init__(self, strategy_names, profiles, payoffs, num_players, 
                 full_game=None, scaling_factor=None):
        """
        Initialize a reduced game.
        
        Parameters:
        strategy_names : list
            Names of strategies in the game
        profiles : list of list
            List of strategy profiles
        payoffs : list of list
            List of payoffs for each profile
        num_players : int
            Number of players in the reduced game
        full_game : Game, optional
            The original full-sized game that was reduced
        scaling_factor : float, optional
            The scaling factor used in the reduction
        """
        super().__init__(strategy_names, profiles, payoffs, num_players)
        self.full_game = full_game
        self.scaling_factor = scaling_factor
        self.is_reduced = full_game is not None
    
    def map_to_full_game(self, reduced_profile):
        """
        Maps a profile from the reduced game to the corresponding profile in the full game.
        
        Parameters:
        reduced_profile : list
            A strategy profile from the reduced game
            
        Returns:
        list : The corresponding strategy profile in the full game
        """
        if not self.is_reduced:
            raise ValueError("This is not a reduced game")
            
        N = self.full_game.num_players
        n = self.num_players
        
        # Scale the reduced profile to the full game size
        full_profile = [int(round(count * (N - 1)/(n - 1))) for count in reduced_profile]
        
        total = sum(full_profile)
        if total != N:
            diff = N - total
            
            for i in range(abs(diff)):
                if diff > 0:
                    # Need to add players
                    idx = i % len(full_profile)
                    if reduced_profile[idx] > 0:  
                        full_profile[idx] += 1
                else:
                    # Need to remove players
                    idx = i % len(full_profile)
                    if full_profile[idx] > 0: 
                        full_profile[idx] -= 1
                
        return full_profile
    
    def get_full_game_payoff(self, reduced_profile):
        """
        Gets the payoffs from the full game for a reduced game profile.
        
        Parameters:
        reduced_profile : list
            A strategy profile from the reduced game
            
        Returns:
        list : The payoffs from the full game for the corresponding profile
        """
        if not self.is_reduced:
            raise ValueError("This is not a reduced game")
            
        full_profile = self.map_to_full_game(reduced_profile)
        return self.full_game.get_payoff_for_profile(full_profile)
    
    def validate_reduction(self):
        """
        Validates that the reduced game preserves the strategic properties of the full game.
        Checks if the best responses in the reduced game match best responses in the full game.
        
        Returns:
        bool : True if the reduction is valid, False otherwise
        """
        if not self.is_reduced:
            raise ValueError("This is not a reduced game")
            
        # Check a sample of profiles to validate the reduction
        for reduced_profile in self.profiles:
            reduced_payoffs = self.get_payoff_for_profile(reduced_profile)
            full_profile = self.map_to_full_game(reduced_profile)
            full_payoffs = self.full_game.get_payoff_for_profile(full_profile)
            
            # Best response indices should match
            reduced_br = np.argmax(reduced_payoffs)
            full_br = np.argmax(full_payoffs)
            
            if reduced_br != full_br:
                return False
                
        return True
    
    def get_reduced_game_payoff(self, reduced_profile):
        """
        Gets the payoffs from the reduced game for a reduced game profile.
        """
        return self.get_payoff_for_profile(reduced_profile)
    
    def get_subgame(self, strategy_subset):
        """
        Creates a subgame using only the specified strategies.
        
        Parameters:
        strategy_subset : list
            Subset of strategy names to include
            
        Returns:
        ReducedGame : A new game with only the specified strategies
        """
        if not all(s in self.strategy_names for s in strategy_subset):
            raise ValueError("Strategy subset contains invalid strategies")
            
        # Get indices of the specified strategies
        strategy_indices = [self.strategy_names.index(s) for s in strategy_subset]
        
        # Filter profiles that only use the specified strategies
        new_profiles = []
        new_payoffs = []
        
        for profile, payoff in zip(self.profiles, self.payoffs):
            # Check if profile only uses the specified strategies
            valid_profile = True
            for i, count in enumerate(profile):
                if i not in strategy_indices and count > 0:
                    valid_profile = False
                    break
                    
            if valid_profile:
                # Create new profile and payoff with only the specified strategies
                new_profile = [profile[i] for i in strategy_indices]
                new_payoff = [payoff[i] for i in strategy_indices]
                
                new_profiles.append(new_profile)
                new_payoffs.append(new_payoff)
                
        # Create new ReducedGame with only the specified strategies
        return ReducedGame(
            strategy_subset,
            new_profiles,
            new_payoffs,
            self.num_players,
            self.full_game,
            self.scaling_factor
        )
    
    def solve_game_cfr(self, iterations=100, epsilon=1e-6):
        """
        Solves the game using CFR.
        """
        #initial uniform distribution for mss
        #TODO: hot start, start with supports proportional to times played in the true game

        strategy_probs = np.ones(self.num_strategies) / self.num_strategies

        #initalize cumulative regrets and strat sums 
        cumulative_regrets = np.zeros(self.num_strategies)
        strategy_sums = np.zeros(self.num_strategies)

        for i in range(iterations): 
            #now we compute expected payoffs for each strategy
            expected_payoffs = np.zeros(self.num_strategies)

            for i, profile in enumerate(self.profiles):
                #find probability of this strategy profile occuring 
                profile_prob = 1.0
                for j, count in enumerate(profile):
                    if count > 0:
                        profile_prob *= (strategy_probs[j] ** count)

                for k in range(self.num_strategies):
                    if profile[k] > 0:
                        expected_payoffs[k] += profile_prob * self.payoffs[i][k]

            #here i calculate regret 
            current_expected_payoff = np.dot(strategy_probs, expected_payoffs)
            regrets = expected_payoffs - current_expected_payoff
            #lol
            cumulative_regrets += regrets 
            #find new strategy with regret matching 
            pos_cumulative_regrets = np.maximum(cumulative_regrets, 0)
            regret_sum = np.sum(pos_cumulative_regrets)

            #TODO: not sure if this is correct/what do to here?
            if regret_sum > 0:
                strategy_probs = pos_cumulative_regrets / regret_sum
            else:
                strategy_probs = np.ones(self.num_strategies) / self.num_strategies

            #update strategy sums 
            strategy_sums += strategy_probs
            print(f"Iteration {i}: {strategy_probs}")
            # Check for convergence
            if i > 0 and i % 10 == 0:
                avg_strategy = strategy_sums / (i + 1)
                if np.max(np.abs(avg_strategy - strategy_probs)) < epsilon:
                    break
        
        # Calculate average strategy (approximate Nash equilibrium)
        avg_strategy = strategy_sums / iterations
        print(f"Final Strategy: {avg_strategy}")
        print(f"Regrets: {cumulative_regrets}")
        return {strat: prob for strat, prob in zip(self.strategy_names, avg_strategy)}
        

       
        






       
            
