from egta.game import Game
import numpy as np
from math import factorial, comb

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
    
    def solve_game_cfr(self, iterations=1000, epsilon=1e-6):
        """
        Solves the symmetric game using CFR.

        implemented following: https://ai.plainenglish.io/steps-to-building-a-poker-ai-part-4-regret-matching-for-rock-paper-scissors-in-python-168411edbb13
        """
        print(f"Solving game with {self.num_strategies} strategies and {self.num_players} players")

        # Initialize with uniform distribution
        strategy_probs = np.ones(self.num_strategies) / self.num_strategies
        
        # Initialize cumulative regrets and strategy sums
        cumulative_regrets = np.zeros(self.num_strategies)
        cumulative_regrets_dict = {}
        for strategy in self.strategy_names:
            cumulative_regrets_dict[strategy] = []
        strategy_probs_dict = {}
        for strategy in self.strategy_names:
            strategy_probs_dict[strategy] = []
        
    
        strategy_sums = np.zeros(self.num_strategies)

        for i in range(iterations):
            # Calculate expected payoff for each pure strategy
            expected_payoffs = np.zeros(self.num_strategies)
            
            for s in range(self.num_strategies):
                # Calculate expected payoff when playing pure strategy s
                for j, profile in enumerate(self.profiles):
                    # Skip if profile doesn't use strategy s
                    if profile[s] == 0:
                        continue
                        
                    #this creates a modified profiles of other players strategies 
                    #we remove the focal strategy s from the profile 
                    other_players_profile = profile.copy()
                    other_players_profile[s] -= 1  #remove its contribution
                    
                    #calculate probability of other players choosing this distribution
                    profile_prob = 1.0
                    n_others = self.num_players - 1
                    multinomial_coef = factorial(n_others)

                    profile_prob *= multinomial_coef

                    for strat, count in enumerate(other_players_profile):
                        if count > 0:
                            profile_prob *= (strategy_probs[strat] ** count)
                    
                    # Add weighted payoff to expected payoff for strategy s
                    expected_payoffs[s] += profile_prob * self.payoffs[j][s]
            
            # Calculate current expected value with mixed strategy
            current_expected_value = np.sum(strategy_probs * expected_payoffs)
            
            # Calculate regrets
            regrets = expected_payoffs - current_expected_value
            
            # Update cumulative regrets
            cumulative_regrets += regrets        

            

            
            # Compute new strategy probabilities using regret matching
            positive_regrets = np.maximum(0, cumulative_regrets)
            regret_sum = np.sum(positive_regrets)

            for strategy in self.strategy_names:
                cumulative_regrets_dict[strategy].append(positive_regrets[self.strategy_names.index(strategy)])
            
            if regret_sum > 0:
                strategy_probs = positive_regrets / regret_sum   
            else:
                strategy_probs = np.ones(self.num_strategies) / self.num_strategies
                
            
            # Update strategy sums for averaging normalized
            strategy_sums += strategy_probs / self.num_strategies
            
            for strategy in self.strategy_names:
                strategy_probs_dict[strategy].append(strategy_probs[self.strategy_names.index(strategy)])
            
            if i % 100 == 0:
                print(f"Iteration {i}: {strategy_probs}")
            
            # Check for convergence
            if i > 0 and i % 10 == 0:
                avg_strategy = strategy_sums / (i + 1)
                if np.max(np.abs(avg_strategy - strategy_probs)) < epsilon:
                    print(f"Converged after {i} iterations")
                    break
        
        # Calculate average strategy (approximate Nash equilibrium)
        avg_strategy = strategy_sums / iterations
        
        # Normalize to ensure probabilities sum to 1
        avg_strategy = avg_strategy / np.sum(avg_strategy)
        
        print(f"Final Strategy: {avg_strategy}")

        print(f"Regrets: {cumulative_regrets}")
        
        # Return as a dictionary
        return {strat: prob for strat, prob in zip(self.strategy_names, avg_strategy)}, cumulative_regrets_dict, strategy_probs_dict
        

       
        






       
            
