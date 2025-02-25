import numpy as np
from egta.game import Game
from egta.reductions.reducedgame import ReducedGame

class DPRGame:
    """
    Implements Deviation-Preserving Reduction for symmetric games.
    
    DPR creates a smaller game that preserves the payoff incentive structure
    of the original game, making it more computationally feasible to analyze
    while maintaining the strategic characteristics.
    """
    def __init__(self, full_game: Game, full_game_players: int, reduced_players: int):
        """
        Initialize DPR game reduction.
        
        Parameters:
        full_game : Game
            The original full-sized game to reduce
        full_game_players : int
            Number of players in the full game
        reduced_players : int
            Target number of players in the reduced game
        """
        self.full_game = full_game
        self.full_game_players = full_game_players
        self.reduced_players = reduced_players
        self.strategies = full_game.strategy_names
        self.num_strategies = len(self.strategies)
        
        # Validate the reduction
        if not self.check_reduced_players_ratio():
            raise ValueError(
                f"Invalid DPR reduction: (N-1)={(full_game_players-1)} " + 
                f"must be divisible by (n-1)={(reduced_players-1)}"
            )
        
        # Calculate the reduction scaling factor
        self.scaling_factor = (full_game_players - 1) / (reduced_players - 1)
        self.reduced_game = self.construct_reduced_game()

    
    def check_reduced_players_ratio(self):
        """
        Check if the reduction ratio is valid for DPR.
        N-1 must be divisible by n-1 where N is full game players and n is reduced game players.
        """
        N = self.full_game_players
        n = self.reduced_players

        if (N - 1) % (n - 1) != 0:
            return False
        return True
        
    def construct_reduced_game(self):
        """
        Constructs the reduced game from the full game.
        Returns a ReducedGame object with the properly computed payoffs.
        """
        N = self.full_game_players
        n = self.reduced_players
        
        reduced_profiles = []
        reduced_payoffs = []
        
        full_profiles = self.full_game.get_profiles()
        full_payoffs = self.full_game.get_payoffs()
        
        for profile, payoff in zip(full_profiles, full_payoffs):
            reduced_profile = self._reduce_profile(profile)
            
            if sum(reduced_profile) == n:
                reduced_profiles.append(reduced_profile)
                reduced_payoffs.append(payoff.copy())  
        return ReducedGame(
            self.strategies,
            reduced_profiles, 
            reduced_payoffs,
            self.reduced_players,
            self.full_game,
            self.scaling_factor 
        )
    
    def _reduce_profile(self, full_profile):
        """
        Reduces a full game profile to a reduced game profile using DPR.
        
        Parameters:
        full_profile : list
            A strategy profile from the full game
            
        Returns:
        list : The corresponding strategy profile in the reduced game
        """
        N = self.full_game_players
        n = self.reduced_players
        
        reduced_profile = [int(round(count * (n / N))) for count in full_profile]
        
        total = sum(reduced_profile)
        if total != n:
            largest_idx = reduced_profile.index(max(reduced_profile))
            reduced_profile[largest_idx] += (n - total)
            
            if reduced_profile[largest_idx] < 0:
                reduced_profile[largest_idx] = 0
                remaining = n - sum(reduced_profile)
                
                for i in range(len(reduced_profile)):
                    if i != largest_idx and reduced_profile[i] > 0:
                        adjustment = min(remaining, reduced_profile[i])
                        reduced_profile[i] += adjustment
                        remaining -= adjustment
                        if remaining == 0:
                            break
                
        return reduced_profile
    
    def map_to_full_profile(self, reduced_profile):
        """
        Maps a reduced game profile back to a full game profile.
        
        Parameters:
        reduced_profile : list
            A strategy profile from the reduced game
            
        Returns:
        list : The corresponding strategy profile in the full game
        """
        N = self.full_game_players
        n = self.reduced_players
        
        full_profile = [int(round(count * (N / n))) for count in reduced_profile]
        
        total = sum(full_profile)
        if total != N:
            diff = N - total
            
            for i in range(abs(diff)):
                if diff > 0:
                    idx = i % len(full_profile)
                    if reduced_profile[idx] > 0:  
                        full_profile[idx] += 1
                else:
                    # Need to remove players
                    idx = i % len(full_profile)
                    if full_profile[idx] > 0:  
                        full_profile[idx] -= 1
                
        return full_profile
        


        
            














    


