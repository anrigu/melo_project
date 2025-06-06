"""
Deviation Preserving Reduction (DPR) scheduler for EGTA with Role Symmetric Game support.
Based on the original implementation from quiesce-master.
"""
import itertools
import random
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional, Set, Union
import torch
from marketsim.egta.core.game import Game
from marketsim.egta.schedulers.base import Scheduler
from functools import lru_cache
import numpy as np



class DPRScheduler(Scheduler):
    """
    Deviation Preserving Reduction scheduler for EGTA.
    Now supports both symmetric and role symmetric games.
    DPR reduces the number of profiles that need to be simulated
    by only requesting profiles needed to compute approximate equilibria.
    """
    
    def __init__(self, 
                strategies: List[str], 
                num_players: int, 
                subgame_size: int = 3,
                batch_size: int = 10,
                reduction_size: Optional[int] = None,
                reduction_size_per_role: Optional[Dict[str, int]] = None,
                seed: Optional[int] = None,
                # Role symmetric parameters
                role_names: Optional[List[str]] = None,
                num_players_per_role: Optional[List[int]] = None,
                strategy_names_per_role: Optional[List[List[str]]] = None):
        """
        Initialize a DPR scheduler.
        
        Args:
            strategies: List of strategy names (for symmetric games) or all strategies (for RSG)
            num_players: Total number of players (N)
            subgame_size: Size of subgames to explore
            batch_size: Number of profiles to return per batch
            reduction_size: Number of players in the reduced game (n)
                If None, uses the full num_players (no reduction)
            seed: Random seed
            role_names: Role names for role symmetric games
            num_players_per_role: Number of players per role for RSG
            strategy_names_per_role: Strategy names per role for RSG
        """
        self.strategies = strategies
        self.num_players = num_players
        self.subgame_size = min(subgame_size, len(strategies))
        self.batch_size = batch_size
        self.rand = random.Random(seed)
        self.game = None
        
        # Role symmetric game support
        self.is_role_symmetric = role_names is not None
        self.role_names = role_names or ["Player"]
        self.num_players_per_role = num_players_per_role or [num_players]
        self.strategy_names_per_role = strategy_names_per_role or [strategies]
        
        if self.is_role_symmetric:
            # -------- role–symmetric branch --------
            # Default: no reduction => n_r = N_r for every role
            if reduction_size_per_role is None:
                reduction_size_per_role = {
                    r: self.num_players_per_role[i]
                    for i, r in enumerate(self.role_names)
                }
            # sanity-check
            for r, n_r in reduction_size_per_role.items():
                if n_r < 1:
                    raise ValueError(f"reduction_size_per_role[{r}] must be ≥ 1")

            self.reduction_size_per_role = reduction_size_per_role
            self.reduction_size          = None         #  <- disable scalar
        else:
            # -------- symmetric branch --------
            self.reduction_size          = (
                int(reduction_size) if reduction_size is not None
                else num_players                            # no reduction
            )
            self.reduction_size_per_role = {}  

        # bookkeeping
        if self.is_role_symmetric:
            self.scaling_factor_per_role = {}
            for i, r in enumerate(self.role_names):
                N_r = self.num_players_per_role[i]
                n_r = self.reduction_size_per_role[r]
                self.scaling_factor_per_role[r] = (
                    1.0 if n_r == N_r else (N_r - 1) / (n_r - 1)
                )
            self.scaling_factor = None                     # <- unused
        else:
            self.scaling_factor = (
                1.0 if self.reduction_size == self.num_players
                else (self.num_players - 1) / (self.reduction_size - 1)
            )
            self.scaling_factor_per_role = {}

        self.scheduled_profiles: Set[Tuple] = set()
        self.requested_subgames: List[Dict] = []
        
        self._initialize_with_uniform_subgame()
    
    def _n_players_for_role(self, role_name: str, role_idx: int) -> int:
        """
        Current number of players the *reduced* game keeps for a role.
        Falls back gracefully to full-game size.
        """
        if self.reduction_size_per_role:
            return self.reduction_size_per_role.get(
                role_name, self.num_players_per_role[role_idx]
            )
        return min(self.num_players_per_role[role_idx], self.reduction_size)
    
    def _initialize_with_uniform_subgame(self):
        """Initialize with a uniform subgame."""
        if self.is_role_symmetric:
            # Choose strategies per role
            initial_subgame = {}
            for role_idx, (role_name, role_strategies) in enumerate(zip(self.role_names, self.strategy_names_per_role)):
                # Choose min(subgame_size, available_strategies) for each role
                max_strats_for_role = min(self.subgame_size, len(role_strategies))
                if max_strats_for_role > 0:
                    chosen_strategies = set(self.rand.sample(role_strategies, max_strats_for_role))
                    initial_subgame[role_name] = chosen_strategies
            self.requested_subgames.append(initial_subgame)
        else:
            # Symmetric game: choose strategies uniformly at random
            initial_strategies = set(self.rand.sample(self.strategies, self.subgame_size))
            self.requested_subgames.append({"Player": initial_strategies})
    '''
    def _generate_profiles_for_subgame(self, subgame: Dict[str, Set[str]]) -> List[List[Tuple[str, str]]]:
        """
        Generate all profiles for a role symmetric subgame.
        
        Args:
            subgame: Dictionary mapping role names to sets of strategies
            
        Returns:
            List of role symmetric profiles
        """
        if self.is_role_symmetric:
            return self._generate_role_symmetric_profiles(subgame)
        else:
            # For symmetric games, convert to old format and generate profiles
            strategies = list(subgame["Player"])
            return self._generate_symmetric_profiles(strategies)
       '''
    def _generate_role_symmetric_profiles(self, subgame: Dict[str, Set[str]]) -> List[List[Tuple[str, str]]]:
        """
        Generate all full‐length profiles for a role‐symmetric subgame.
        Uses a cached helper to enumerate {k strategies, n players} partitions exactly once.
        """
        profiles: List[List[Tuple[str, str]]] = []
        role_profile_options = []

        # 1) For each role, build (role_name, [strategies], [all partitions of n_r players])
        for role_idx, role_name in enumerate(self.role_names):
            if role_name not in subgame or not subgame[role_name]:
                continue

            strategies = list(subgame[role_name])
            # n_r = number of players to keep for this role in the *reduced* game
            n_r = int(min(
                self.num_players_per_role[role_idx].item(),
                self.reduction_size_per_role.get(role_name, self.num_players_per_role[role_idx].item())
            ))

            k = len(strategies)
            if n_r == 0 or k == 0:
                continue

            # Use the cached integer partition helper:
            raw_partitions = _cached_distribute_players(k, n_r)
            # Convert each partition (tuple of length k) into a dict {strategy: count}
            distro_dicts: List[Dict[str, int]] = []
            for part in raw_partitions:
                # Only keep those that sum exactly to n_r (should always be true)
                if sum(part) == n_r:
                    dd = {strategies[i]: part[i] for i in range(k)}
                    distro_dicts.append(dd)

            if distro_dicts:
                role_profile_options.append((role_name, strategies, distro_dicts))

        if not role_profile_options:
            return []

        # 2) Cartesian‐product over each role’s distributions
        import itertools
        for combo in itertools.product(*[opts[2] for opts in role_profile_options]):
            full_profile: List[Tuple[str, str]] = []
            for (role_name, strategies, _), distro in zip(role_profile_options, combo):
                for strat, cnt in distro.items():
                    full_profile.extend([(role_name, strat)] * cnt)
            if full_profile:
                profiles.append(full_profile)

        return profiles
 

    def _n_players_for_role(self, role_name: str, role_idx: int) -> int:
        """
        Centralised lookup of the *current* player-count a reduced game
        should use for this role.  Falls back gracefully.
        """
       
        if hasattr(self, "reduction_size_per_role") and self.reduction_size_per_role:
            return int(self.reduction_size_per_role.get(
                role_name, self.num_players_per_role[role_idx]))

        
        if hasattr(self, "reduction_size") and self.reduction_size:
            return int(min(self.num_players_per_role[role_idx],
                           self.reduction_size))

       
        return int(self.num_players_per_role[role_idx])

    
    def _generate_role_symmetric_profiles(self, subgame: Dict[str, Set[str]]) -> List[List[Tuple[str, str]]]:
        """Generate role symmetric profiles for a subgame."""
        profiles = []
        
        # Generate all combinations of strategy assignments per role
        role_profile_options = []
        
        for role_idx, role_name in enumerate(self.role_names):
            if role_name not in subgame or not subgame[role_name]:
                continue
                
           
            role_name = self.role_names[role_idx]
            role_strategies = list(subgame[role_name])

            num_players_in_role = int(self.num_players_per_role[role_idx])



            
            if num_players_in_role == 0:
                continue
            
            # Generate all ways to distribute players among strategies in this role
            role_distributions = self._distribute_players_in_role(role_strategies, num_players_in_role)
            role_profile_options.append(role_distributions)
        
        if not role_profile_options:
            return []
        
        # Generate cartesian product of role distributions (one distribution per role)
        for role_distribution_combination in itertools.product(*role_profile_options):
            profile = []
            for role_idx, distribution in enumerate(role_distribution_combination):
                role_name = self.role_names[role_idx]
                for strategy, count in distribution.items():
                    for _ in range(count):
                        profile.append((role_name, strategy))
            
            if profile:  # Only add non-empty profiles
                profiles.append(profile)
        
        return profiles
    
    def _distribute_players_in_role(self, strategies: List[str], num_players: int) -> List[Dict[str, int]]:
        """Generate all ways to distribute players among strategies in a role."""
        if len(strategies) == 1:
            return [{strategies[0]: num_players}]
        
        distributions = []
        
        def generate_distributions(strategies_left, players_left, current_dist):
            if len(strategies_left) == 1:
                # Assign all remaining players to the last strategy
                final_dist = current_dist.copy()
                final_dist[strategies_left[0]] = players_left
                distributions.append(final_dist)
                return
            
            strategy = strategies_left[0]
            remaining_strategies = strategies_left[1:]
            
            # Try assigning 0 to all players to this strategy
            for assigned_to_this in range(players_left + 1):
                new_dist = current_dist.copy()
                new_dist[strategy] = assigned_to_this
                generate_distributions(remaining_strategies, players_left - assigned_to_this, new_dist)
        
        generate_distributions(strategies, num_players, {})
        return distributions
    
    def _generate_symmetric_profiles(self, strategies: List[str]) -> List[List[Tuple[str, str]]]:
        """Generate profiles for symmetric games (backward compatibility)."""
        profiles = []
        distributions = self._distribute_players(len(strategies), self.reduction_size)
        
        for distribution in distributions:
            profile = []
            for strategy_idx, count in enumerate(distribution):
                strategy = strategies[strategy_idx]
                for _ in range(count):
                    profile.append(("Player", strategy))
            profiles.append(profile)
        
        return profiles
    
    def _distribute_players(self, num_strategies: int, num_players: int) -> List[List[int]]:
        """
        Generate all ways to distribute players among strategies.
        
        Args:
            num_strategies: Number of strategies
            num_players: Number of players
            
        Returns:
            List of distributions, each a list of counts
        """
        if num_strategies == 1:
            return [[num_players]]
        
        result = []
        for i in range(num_players + 1):
            for sub_dist in self._distribute_players(num_strategies - 1, num_players - i):
                result.append([i] + sub_dist)
        
        return result
    
    def _select_equilibrium_candidates(self, game: Game, max_candidates: int = 10) -> List[np.ndarray]:
        """
        Select candidate equilibria from the game.
        
        Args:
            game: Game with existing data
            max_candidates: Maximum number of candidates to return
            
        Returns:
            List of candidate equilibrium mixtures
        """
        from marketsim.egta.solvers.equilibria import replicator_dynamics
        
        device = game.game.device if hasattr(game.game, 'device') else 'cpu'
        
        candidates = []
        
        for subgame in self.requested_subgames:
            if game.is_role_symmetric:
                # Create mixture for role symmetric game
                mixture = np.zeros(game.num_strategies)
                
                # Find indices for strategies in subgame and create role-normalized mixture
                global_strategy_idx = 0
                for role_idx, (role_name, role_strategies) in enumerate(zip(game.role_names, game.strategy_names_per_role)):
                    role_start_idx = global_strategy_idx
                    role_end_idx = global_strategy_idx + len(role_strategies)
                    
                    if role_name in subgame and subgame[role_name]:
                        # Create uniform mixture over strategies in this role's subgame
                        subgame_strategies_in_role = subgame[role_name]
                        num_subgame_strats = len(subgame_strategies_in_role)
                        
                        for strategy in role_strategies:
                            if strategy in subgame_strategies_in_role:
                                mixture[global_strategy_idx] = 1.0 / num_subgame_strats
                            global_strategy_idx += 1
                    else:
                        # If role not in subgame, create uniform mixture over all strategies in role
                        num_strats_in_role = len(role_strategies)
                        for strategy in role_strategies:
                            mixture[global_strategy_idx] = 1.0 / num_strats_in_role if num_strats_in_role > 0 else 0.0
                            global_strategy_idx += 1
            else:
                # Symmetric game
                strategy_mapping = {name: i for i, name in enumerate(game.strategy_names)}
                mixture = np.zeros(len(game.strategy_names))
                
                subgame_strategies = subgame.get("Player", set())
                subgame_indices = [strategy_mapping[s] for s in subgame_strategies if s in strategy_mapping]
                
                if subgame_indices:
                    mixture[subgame_indices] = 1.0 / len(subgame_indices)
            
            if mixture.sum() > 0:
                mixture_tensor = torch.tensor(mixture, dtype=torch.float32, device=device)
                eq_mixture = replicator_dynamics(game, mixture_tensor, iters=1000)
                candidates.append(eq_mixture.cpu().numpy())
        
        if not candidates:
            # Create uniform mixture that respects role structure
            if game.is_role_symmetric:
                mixture = np.zeros(game.num_strategies)
                global_strategy_idx = 0
                for role_strategies in game.strategy_names_per_role:
                    num_strats_in_role = len(role_strategies)
                    for _ in role_strategies:
                        mixture[global_strategy_idx] = 1.0 / num_strats_in_role if num_strats_in_role > 0 else 0.0
                        global_strategy_idx += 1
            else:
                mixture = np.ones(game.num_strategies) / game.num_strategies
            candidates.append(mixture)
        
        if len(candidates) > max_candidates:
            candidates = self.rand.sample(candidates, max_candidates)
        
        return candidates
    
    def scale_payoffs(self, payoffs: torch.Tensor) -> torch.Tensor:
        """
        Apply DPR scaling per role:  (N_r-1)/(n_r-1)
        """
        if not self.is_role_symmetric:                 # symmetric game – old path
            return payoffs * self.scaling_factor

        scaled = payoffs.clone()
        global_idx = 0
        for role_idx, role_name in enumerate(self.role_names):
            n_strats = len(self.strategy_names_per_role[role_idx])
            factor = self.scaling_factor_per_role[role_name]
            scaled[global_idx : global_idx + n_strats] *= factor
            global_idx += n_strats
        return scaled

    
    def _select_deviating_strategies(self, game: Game, mixture: np.ndarray, num_deviations: int = 2) -> Dict[str, Set[str]]:
        """
        Select strategies with highest deviation payoff for each role.
        
        Args:
            game: Game with existing data
            mixture: Mixture to analyze
            num_deviations: Number of deviating strategies to select per role
            
        Returns:
            Dictionary mapping role names to sets of deviating strategies
        """
        device = game.game.device if hasattr(game.game, 'device') else 'cpu'
        mixture_tensor = torch.tensor(mixture, dtype=torch.float32, device=device)
        
        payoffs = game.deviation_payoffs(mixture_tensor)
        scaled_payoffs = self.scale_payoffs(payoffs)
        
        deviating_strategies = {}
        
        if game.is_role_symmetric:
            # Select best deviations per role
            global_strategy_idx = 0
            for role_idx, (role_name, role_strategies) in enumerate(zip(game.role_names, game.strategy_names_per_role)):
                #role_payoffs = scaled_payoffs[global_strategy_idx:global_strategy_idx + len(role_strategies)]
                role_payoffs = torch.nan_to_num(
                    scaled_payoffs[global_strategy_idx:
                                global_strategy_idx + len(role_strategies)],
                    nan=np.inf,  # treat “missing” as best-possible payoff
                )
                # Get top strategies for this role
                sorted_indices = np.argsort(-role_payoffs.cpu().numpy())
                num_to_select = min(num_deviations, len(role_strategies))
                
                selected_strategies = set()
                for i in range(num_to_select):
                    strategy_idx = sorted_indices[i]
                    strategy_name = role_strategies[strategy_idx]
                    selected_strategies.add(strategy_name)
                
                deviating_strategies[role_name] = selected_strategies
                global_strategy_idx += len(role_strategies)
        else:
            # Symmetric game
            sorted_indices = np.argsort(-scaled_payoffs.cpu().numpy())
            deviating_indices = sorted_indices[:num_deviations]
            deviating_strategies["Player"] = {game.strategy_names[i] for i in deviating_indices}
        
        return deviating_strategies
    
    def _select_support_strategies(self, game: Game, mixture: np.ndarray, threshold: float = 0.01) -> Dict[str, Set[str]]:
        """
        Select strategies that are played with significant probability in each role.
        
        Args:
            game: Game instance
            mixture: Mixture to analyze
            threshold: Probability threshold
            
        Returns:
            Dictionary mapping role names to sets of support strategies
        """
        support_strategies = {}
        
        if game.is_role_symmetric:
            global_strategy_idx = 0
            for role_idx, (role_name, role_strategies) in enumerate(zip(game.role_names, game.strategy_names_per_role)):
                role_mixture = mixture[global_strategy_idx:global_strategy_idx + len(role_strategies)]
                
                support_indices = np.where(role_mixture > threshold)[0]
                support_strategies[role_name] = {role_strategies[i] for i in support_indices}
                
                global_strategy_idx += len(role_strategies)
        else:
            # Symmetric game
            support_indices = np.where(mixture > threshold)[0]
            support_strategies["Player"] = {game.strategy_names[i] for i in support_indices}
        
        return support_strategies
    
    def get_next_batch(self, game: Optional[Game] = None) -> List[List[Tuple[str, str]]]:
        """
        Get the next batch of role symmetric profiles to simulate.
        Args:
            game: Optional game with existing data
        Returns:
            List of role symmetric strategy profiles
        """
        if game is None:
            profiles_to_simulate = []
            for subgame in self.requested_subgames:
                profiles_to_simulate.extend(self._generate_profiles_for_subgame(subgame))
        else:
            self.game = game
            
            candidates = self._select_equilibrium_candidates(game)
            
            new_subgames = []
            for candidate in candidates:
                support_strategies = self._select_support_strategies(game, candidate)
                deviating_strategies = self._select_deviating_strategies(game, candidate)
                
                # Merge support and deviating strategies per role
                new_subgame = {}
                all_roles = set(support_strategies.keys()) | set(deviating_strategies.keys())
                
                for role_name in all_roles:
                    role_strategies = set()
                    if role_name in support_strategies:
                        role_strategies.update(support_strategies[role_name])
                    if role_name in deviating_strategies:
                        role_strategies.update(deviating_strategies[role_name])
                    
                    # Limit to subgame size per role
                    if len(role_strategies) > self.subgame_size:
                        role_strategies = set(list(role_strategies)[:self.subgame_size])
                    
                    new_subgame[role_name] = role_strategies
                
                new_subgames.append(new_subgame)
            
            self.requested_subgames.extend(new_subgames)
            
            profiles_to_simulate = []
            for subgame in new_subgames:
                profiles_to_simulate.extend(self._generate_profiles_for_subgame(subgame))
        
        # Filter out already scheduled profiles
        new_profiles = []
        for profile in profiles_to_simulate:
            sorted_profile = tuple(sorted(profile))
            
            if sorted_profile not in self.scheduled_profiles:
                new_profiles.append(profile)
                self.scheduled_profiles.add(sorted_profile)
        
        self.rand.shuffle(new_profiles)
        return new_profiles[:self.batch_size]
    
    def update(self, game: Game) -> None:
        """
        Update the scheduler with new game data.
        
        Args:
            game: Game with updated data
        """
        self.game = game
        
    def get_scaling_info(self) -> Dict[str, Any]:
        return {
            "full_game_players": self.num_players_per_role,
            "reduced_game_players": [self.reduction_size_per_role[r] for r in self.role_names],
            "scaling_factors": self.scaling_factor_per_role,
            "is_role_symmetric": self.is_role_symmetric,
        }
    '''
    def missing_deviations(self, mixture: np.ndarray, game: Game) -> List[List[Tuple[str,str]]]:
        """
        Return every unevaluated one-player deviation profile of σ.
        """
        support = self._select_support_strategies(game, mixture, threshold=1e-12)
        missing = []
        import itertools
        # build a pure profile over the support (one player per role strategy)
        base = []
        for role, strats in support.items():
            base.append([(role, s) for s in strats])
        for pure in itertools.product(*base):
            # `pure` is a compact list with ONE (role,strategy) per role
            pure = list(pure)

            for i, (role_i, strat_i) in enumerate(pure):
                for other in self.strategy_names_per_role[self.role_names.index(role_i)]:
                    if other == strat_i:
                        continue

                    # --- build compact dev profile ---------------------------
                    dev_compact = pure.copy()
                    dev_compact[i] = (role_i, other)

                    # --- EXPAND to a full-length profile --------------------
                    dev_full = []
                    for role_name, strat_name in dev_compact:
                        n_players = int(game.num_players_per_role[
                                        game.role_names.index(role_name)].item())
                        dev_full.extend([(role_name, strat_name)] * n_players)
                    # ---------------------------------------------------------
                    #print(f"Profiles to simulate: {dev_full}")
                    if not game.has_profile(dev_full):
                        missing.append(dev_full)

        return missing

        '''
@lru_cache(maxsize=128)
def _cached_distribute_players(k: int, n: int) -> Tuple[Tuple[int, ...], ...]:
    """
    Return all ways to assign `n` players over `k` strategies as tuples of length k.
    Cached so that (_cached_distribute_players(k,n)) is reused.
    """
    distributions = []

    def rec(strats_left: int, players_left: int, current: list):
        if strats_left == 1:
            # Assign all remaining players to the last strategy
            distributions.append(tuple(current + [players_left]))
            return
        for x in range(players_left + 1):
            rec(strats_left - 1, players_left - x, current + [x])

    rec(k, n, [])
    return tuple(distributions)

