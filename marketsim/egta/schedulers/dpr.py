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
import heapq



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
        # Min-heap of pending sub-games keyed by size.  Each item is a tuple
        #   (|support|, tie_break, sub_dict)
        self.requested_subgames: List[Tuple[int, float, Dict]] = []
        # Track the canonical key of sub-games already queued so we avoid
        # duplicates.
        self._queued_keys: Set[frozenset] = set()
        
        # Maximum number of profiles to emit from *one* sub-game in a single
        # call to get_next_batch.  Keeping this small prevents the first
        # scheduler batch from being dominated by the very first sub-game in
        # the queue, which can otherwise lead to poor initial coverage of the
        # strategy space.  The default of 5 can be tuned by callers.
        self.profiles_per_subgame: int = 0
        
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
            self.add_subgame(initial_subgame)
        else:
            # Symmetric game: choose strategies uniformly at random
            initial_strategies = set(self.rand.sample(self.strategies, self.subgame_size))
            self.add_subgame({"Player": initial_strategies})
    
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
            # Robustly convert player counts (can be torch scalar or plain int) to Python int
            full_N_r = self.num_players_per_role[role_idx]
            if hasattr(full_N_r, "item"):
                full_N_r = int(full_N_r.item())
            else:
                full_N_r = int(full_N_r)

            red_n_r = self.reduction_size_per_role.get(role_name, full_N_r)
            # Ensure the reduced count is also an int
            red_n_r = int(red_n_r.item()) if hasattr(red_n_r, "item") else int(red_n_r)

            n_r = min(full_N_r, red_n_r)

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

        # 2) Cartesian‐product over each role's distributions
        import itertools
        for combo in itertools.product(*[opts[2] for opts in role_profile_options]):
            full_profile: List[Tuple[str, str]] = []
            for (role_name, strategies, _), distro in zip(role_profile_options, combo):
                # Lookup full and reduced player counts for this role (computed earlier)
                role_idx = self.role_names.index(role_name)
                full_N_r = self.num_players_per_role[role_idx]
                if hasattr(full_N_r, "item"):
                    full_N_r = int(full_N_r.item())
                else:
                    full_N_r = int(full_N_r)

                red_n_r = self.reduction_size_per_role.get(role_name, full_N_r)
                red_n_r = int(red_n_r.item()) if hasattr(red_n_r, "item") else int(red_n_r)

                # Integer scaling factor (may be 1 if no reduction)
                scale = max(1, full_N_r // red_n_r)
                # First allocate scaled counts
                tmp_counts = {s: cnt * scale for s, cnt in distro.items()}
                # Handle any residual players due to floor division
                allocated = sum(tmp_counts.values())
                residual = full_N_r - allocated
                if residual > 0:
                    # Assign residual players to strategies with highest count first
                    sorted_strats = sorted(tmp_counts.items(), key=lambda x: -x[1])
                    idx = 0
                    while residual > 0 and sorted_strats:
                        strat_name, _ = sorted_strats[idx % len(sorted_strats)]
                        tmp_counts[strat_name] += 1
                        residual -= 1
                        idx += 1

                # Add to full profile list
                for strat, cnt in tmp_counts.items():
                    full_profile.extend([(role_name, strat)] * cnt)

            if full_profile:
                profiles.append(full_profile)

                # Early-exit if we have already generated too many profiles for
                # this sub-game.  Prevents combinatorial explosion when roles
                # have many players or strategies.
                MAX_PROFILES_PER_SUBGAME = getattr(self, "max_profiles_per_subgame", 1000)
                if len(profiles) >= MAX_PROFILES_PER_SUBGAME:
                    break  # stop gathering more

        return profiles
 
    
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
        
        for _size, _tie, subgame in self.requested_subgames:
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

    
    def _select_deviating_strategies(self, game: Game, mixture: np.ndarray, num_deviations: int = 16) -> Dict[str, Set[str]]:
        """
        Select strategies with highest deviation payoff for each role.
        
        Args:
            game: Game with existing data
            mixture: Mixture to analyze
            num_deviations: Number of deviating strategies to select per role
            
        Returns:
            Dictionary mapping role names to sets of deviating strategies
        """
        # ---------------------------------------------------------------
        # Use the *full* game (if available) when computing deviation
        # payoffs, so strategies outside the current restriction can still
        # be considered as best responses.
        # ---------------------------------------------------------------
        full_core = getattr(game.game, "full_game_reference", None)
        if full_core is not None:
            # Wrap in a lightweight facade to access deviation_payoffs etc.
            from marketsim.egta.core.game import Game as _GameWrap
            target_game = _GameWrap(full_core, game.metadata) if not isinstance(full_core, _GameWrap.__bases__) else game  # avoid double wrap
        else:
            target_game = game

        device = target_game.game.device if hasattr(target_game.game, 'device') else 'cpu'

        if len(mixture) < target_game.num_strategies:
            mix_full = np.zeros(target_game.num_strategies, dtype=np.float32)
            mix_full[: len(mixture)] = mixture
        else:
            mix_full = mixture

        mixture_tensor = torch.tensor(mix_full, dtype=torch.float32, device=device)
        
        payoffs = target_game.deviation_payoffs(mixture_tensor)
        scaled_payoffs = self.scale_payoffs(payoffs)
        
        deviating_strategies = {}
        
        if target_game.is_role_symmetric:
            # Select best deviations per role
            global_strategy_idx = 0
            for role_idx, (role_name, role_strategies) in enumerate(zip(target_game.role_names, target_game.strategy_names_per_role)):
                #role_payoffs = scaled_payoffs[global_strategy_idx:global_strategy_idx + len(role_strategies)]
                role_payoffs = torch.nan_to_num(
                    scaled_payoffs[global_strategy_idx:
                                global_strategy_idx + len(role_strategies)],
                    nan=np.inf,  # treat "missing" as best-possible payoff
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
            deviating_strategies["Player"] = {target_game.strategy_names[i] for i in deviating_indices}
        
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
        batch: List[List[Tuple[str, str]]] = []
        needed = self.batch_size

        # 1) Drain previously requested subgames up to batch_size
        while self.requested_subgames and needed > 0:
            size_, _tie, sub = heapq.heappop(self.requested_subgames)
            # remove canonical key so that if we re-queue later it is allowed
            self._queued_keys.discard(self._canonical_key(sub))
            emitted_from_sub = 0
            prof_list = self._generate_profiles_for_subgame(sub)
            # Randomise order so early slices are diverse.
            self.rand.shuffle(prof_list)

            # ---------- prepend pure profiles for coverage -----------------
            pure_profiles = []
            default_by_role = {r: next(iter(ss)) for r, ss in sub.items() if ss}

            for role_name, strat_set in sub.items():
                for strat in strat_set:
                    prof = []
                    for r_idx, r in enumerate(self.role_names):
                        choose = strat if r == role_name else default_by_role.get(r)
                        if choose is None:
                            choose = next(iter(self.strategy_names_per_role[r_idx]))
                        n_players = int(self.num_players_per_role[r_idx])
                        prof.extend([(r, choose)] * n_players)
                    pure_profiles.append(prof)

            # put pure profiles first
            prof_list = pure_profiles + prof_list

            for prof in prof_list:
                tpl = tuple(sorted(prof))
                if tpl in self.scheduled_profiles:
                    continue

                self.scheduled_profiles.add(tpl)
                batch.append(prof)
                emitted_from_sub += 1
                needed -= 1
                if needed == 0:
                    break

        # 2) If still need more, generate from new candidate subgames
        if game is not None and needed > 0:
            self.game = game
            candidates = self._select_equilibrium_candidates(game)

            for candidate in candidates:
                # Build support and deviator sets
                support = self._select_support_strategies(game, candidate)
                deviators = self._select_deviating_strategies(game, candidate)

                # Merge into a single subgame dict
                sub = {}
                roles = set(support) | set(deviators)
                for role in roles:
                    role_idx = game.role_names.index(role)
                    legal = set(game.strategy_names_per_role[role_idx])
                    s = (set(support.get(role, [])) | set(deviators.get(role, []))) & legal
                    if not s:
                        continue  # skip if nothing legal
                    if len(s) > self.subgame_size:
                        s = set(list(sorted(s))[:self.subgame_size])
                    sub[role] = s

                # Generate profiles for this subgame up to needed
                for prof in self._generate_profiles_for_subgame(sub):
                    tpl = tuple(sorted(prof))
                    if tpl not in self.scheduled_profiles:
                        self.scheduled_profiles.add(tpl)
                        batch.append(prof)
                        needed -= 1
                        if needed == 0:
                            break
                if needed == 0:
                    break

        # Shuffle and return at most batch_size profiles
        self.rand.shuffle(batch)
        return batch
    
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
    
    def missing_deviations(self, mixture: np.ndarray, game: Game) -> List[List[Tuple[str,str]]]:
        """
        return every unevaluated one-player deviation profile of the strategy profile.
        """
        support = self._select_support_strategies(game, mixture, threshold=1e-12)
        missing = []
        import itertools
        base = []
        for role, strats in support.items():
            base.append([(role, s) for s in strats])
        for pure in itertools.product(*base):
            pure = list(pure)

            for i, (role_i, strat_i) in enumerate(pure):
                for other in game.strategy_names_per_role[game.role_names.index(role_i)]:
                    if other == strat_i:
                        continue

                    dev_compact = pure.copy()
                    dev_compact[i] = (role_i, other)  

                    #expand: all but ONE player stay with original strat_i
                    dev_full = []
                    for _idx2, (role_name2, strat_name2) in enumerate(dev_compact):
                        # Determine player count for this role via role name, not list order.
                        role_index_in_game = game.role_names.index(role_name2)
                        n_players_raw2 = game.num_players_per_role[role_index_in_game]
                        n_players = int(n_players_raw2.item() if hasattr(n_players_raw2, 'item') else n_players_raw2)

                        if _idx2 == i: 
                            dev_full.append((role_name2, strat_name2))
                            orig_strat = pure[i][1]
                            dev_full.extend(
                                [(role_name2, orig_strat)] * (n_players - 1)
                            )
                        else:
                            dev_full.extend([(role_name2, strat_name2)] * n_players)
                     # ---------------------------------------------------------

                    
                    unplayed_threshold = 1e-3  

                    # Build name→index mapping from the *game* (safe for restricted games)
                    mapping = {}
                    g_idx = 0
                    for r_name, strats in zip(game.role_names, game.strategy_names_per_role):
                        for s in strats:
                            mapping[(r_name, s)] = g_idx
                            g_idx += 1

                    # Global index for the deviating strategy "other"
                    other_idx = mapping[(role_i, other)]

                    # If mixture vector comes from a *restricted* game it may
                    # be shorter than the full strategy list. In that case the
                    # deviating strategy is by definition unplayed in the
                    # current mixture -> schedule it.
                    if other_idx >= len(mixture):
                        is_unplayed = True
                    else:
                        is_unplayed = mixture[other_idx] < unplayed_threshold

                    if is_unplayed:
                        if not game.has_profile(dev_full):
                            missing.append(dev_full)

        return missing

    # ------------------------------------------------------------------
    # Sub-game queue helpers (priority by size ‑> breadth-first search)
    # ------------------------------------------------------------------
    def _canonical_key(self, sub: Dict[str, Set[str]]) -> frozenset:
        """Return immutable key identifying the sub-game support."""
        return frozenset((role, strat) for role, ss in sub.items() for strat in ss)

    def add_subgame(self, sub: Dict[str, Set[str]]):
        """Push *sub* onto the priority queue if not already present."""
        key = self._canonical_key(sub)
        if key in self._queued_keys:
            return  # already queued
        size = len(key)
        # random tie-break so heap pop of equal sizes is randomised
        heapq.heappush(self.requested_subgames, (size, self.rand.random(), sub))
        self._queued_keys.add(key)

        
@lru_cache(maxsize=256)
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

