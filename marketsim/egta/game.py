class Game:
    def __init__(self, strategy_names, profiles, payoffs):
        """
        THis is the class to construct a full symmetric game, with a payoff matrix
        given a strategy profile and payoff data.

        Parameters:
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

        self.game_matrix = self.get_game_matrix()

    
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
    
    #NOTE: THis will compute full payoff matrix, need be we can apply say random sampling, MC methods, etc. to compute payoff matrix
    def get_payoff_matrix(self): #computes full payoff matrix
        """
        Constructs and returns the payoff matrix.
        returns:
            payoff_matrix : list of lists where each row represents a strategy profile.
        """
        header = ["# " + strat for strat in self.strategy_names] + ["Payoff (" + strat + ")" 
                                                                    for strat in self.strategy_names]
        payoff_matrix = [header]

        for profile, payoff in zip(self.profiles, self.payoffs):
            payoff_matrix.append(profile + payoff)

        return payoff_matrix
    