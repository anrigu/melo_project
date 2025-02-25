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