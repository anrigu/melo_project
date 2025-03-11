import sys
import os
import logging
import torch
import numpy as np
import heapq
from collections import defaultdict, namedtuple
from typing import Dict, List, Tuple, Set, FrozenSet

# Fix imports - use absolute imports instead of relative
from marketsim.egta_test.subgame_search.symmetric_game import SymmetricGame
# If you have these in your codebase, use absolute imports
from marketsim.egta_test.subgame_search.utils.eq_computation import find_equilibria, replicator_dynamics
from marketsim.egta_test.subgame_search.utils.simplex_operations import simplex_normalize

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# A SubgameInfo object holds the necessary information for a subgame
SubgameInfo = namedtuple('SubgameInfo', [
    'strategies',    # Set of strategy indices in this subgame
    'priority',      # Priority for exploration (typically regret-based)
    'is_complete',   # Whether all required payoff data has been sampled
    'eq_mixture',    # The equilibrium mixture (if computed)
    'regret'         # Regret of the equilibrium (if computed)
])

class SubgameSearch:
    """
    Implementation of EGTA with subgame search (quiesce) for symmetric games.
    Based on the subgame search/quiesce algorithm:
    1. Start with small subgames
    2. Find equilibria in current subgames
    3. Check for beneficial deviations
    4. Expand subgames with profitable deviations
    5. Repeat until convergence
    """
    
    def __init__(self, 
                 strategy_names, 
                 simulator,
                 num_players,
                 regret_threshold=1e-3,
                 distance_threshold=0.1,
                 support_threshold=1e-4,
                 restricted_game_size=3,
                 eq_method="replicator_dynamics",
                 device="cpu"):
        """
        Initialize the SubgameSearch algorithm.
        """
        self.strategy_names = strategy_names
        self.num_strategies = len(strategy_names)
        self.strategy_to_idx = {name: i for i, name in enumerate(strategy_names)}
        
        # Handle both a simulator object and a direct simulator function
        if hasattr(simulator, 'simulate_profile'):
            self.simulator = simulator.simulate_profile
        else:
            self.simulator = simulator
            
        self.num_players = num_players
        self.regret_threshold = regret_threshold
        self.distance_threshold = distance_threshold
        self.support_threshold = support_threshold
        self.restricted_game_size = restricted_game_size
        self.eq_method = eq_method
        self.device = device
        
        # Game data structures
        self.full_game = None  # Will be constructed as profiles are sampled
        self.payoff_data = {}  # Dictionary mapping profile tuples to payoffs
        self.equilibria = []   # List of equilibria found with regrets
        
        # Subgame exploration data structures
        self.subgame_queue = []        # Priority queue of subgames to explore
        self.explored_subgames = set() # Set of explored subgame strategy sets
        self.backup_subgames = []      # Backup queue for when no deviations are found
        
        # Tracking for analytics
        self.total_profiles_sampled = 0
        self.total_iterations = 0
        self.subgames_explored = 0
    
    def initialize(self):
        """
        Initialize the subgame search with single-strategy subgames only.
        The algorithm will build up to larger subgames through the discovery
        of beneficial deviations.
        """
        logger.info("Initializing subgame search with single-strategy subgames")
        
        # Clear any existing data
        self.subgame_queue = []
        self.explored_subgames = set()
        self.backup_subgames = []
        self.total_profiles_sampled = 0
        self.subgames_explored = 0
        self.equilibria = []
        
        # Start with each pure strategy as a separate subgame
        for i in range(self.num_strategies):
            strategies = frozenset([i])
            subgame = SubgameInfo(
                strategies=strategies,
                priority=1.0,  # High priority for initial exploration
                is_complete=False,
                eq_mixture=None,
                regret=float('inf')
            )
            self._add_to_queue(subgame)
        
        logger.info(f"Added {len(self.subgame_queue)} initial single-strategy subgames")
        logger.info(f"The algorithm will build up to larger subgames as needed")
    
    def run(self, max_iterations=100):
        """
        Run the subgame search algorithm.
        """
        logger.info(f"Starting subgame search with {max_iterations} max iterations")
        
        # Initialize the subgame queue if it's empty
        if not self.subgame_queue:
            self.initialize()
        
        self.total_iterations = 0
        while self.total_iterations < max_iterations and (self.subgame_queue or self.backup_subgames):
            # Get the highest priority subgame or use backup if queue is empty
            if self.subgame_queue:
                subgame = heapq.heappop(self.subgame_queue)[1]
                logger.info(f"Iteration {self.total_iterations}: Exploring subgame with strategies {[self.strategy_names[i] for i in subgame.strategies]}")
            else:
                # If main queue is empty but we have backups, use one
                if self.backup_subgames:
                    logger.info(f"Main queue empty, using backup subgame")
                    subgame = heapq.heappop(self.backup_subgames)[1]
                    logger.info(f"Iteration {self.total_iterations}: Exploring backup subgame with strategies {[self.strategy_names[i] for i in subgame.strategies]}")
                else:
                    break  # No more subgames to explore
            
            # Check if this subgame has already been explored
            if subgame.strategies in self.explored_subgames:
                logger.info(f"Subgame already explored, skipping")
                continue
            
            # Process this subgame - sampling, solving, checking deviations
            try:
                self._process_subgame(subgame)
            except Exception as e:
                logger.error(f"Error processing subgame: {e}")
                import traceback
                traceback.print_exc()
                # Mark as explored to avoid infinite loop
                self.explored_subgames.add(subgame.strategies)
            
            self.total_iterations += 1
            self.subgames_explored += 1
            
            # Check if we've found beneficial deviations and have more subgames to explore
            if not self.subgame_queue and self.total_iterations >= max_iterations:
                logger.info(f"Reached max iterations ({max_iterations}), stopping search")
                break
        
        logger.info(f"Subgame search completed after {self.total_iterations} iterations")
        logger.info(f"Found {len(self.equilibria)} equilibria")
        logger.info(f"Explored {self.subgames_explored} subgames")
        logger.info(f"Sampled {self.total_profiles_sampled} unique profiles")
        
        return self.equilibria
    
    def _process_subgame(self, subgame):
        """
        Process a subgame: sample profiles, solve for equilibrium, check deviations.
        This is where the algorithm builds up to larger subgames.
        """
        # Ensure all required profiles are sampled for this subgame
        self._sample_subgame_profiles(subgame.strategies)
        logger.info(f"Successfully sampled profiles for subgame with {len(subgame.strategies)} strategies")
        
        # Handle single-strategy subgames specially
        if len(subgame.strategies) == 1:
            # For single-strategy subgames, the "equilibrium" is just using that strategy
            strategy_idx = next(iter(subgame.strategies))
            single_strat_mixture = torch.zeros(1, device=self.device)
            single_strat_mixture[0] = 1.0  # Pure strategy
            
            # Expand to full strategy space
            full_eq_mixture = self._expand_mixture(single_strat_mixture, subgame.strategies)
            
            # Add to equilibria (regret is 0 in this restricted game)
            self.equilibria.append((full_eq_mixture, 0.0))
            logger.info(f"Found single-strategy equilibrium: {self.strategy_names[strategy_idx]}")
            
            # Mark as explored
            self.explored_subgames.add(subgame.strategies)
            
            # If we don't have a full game yet, update it
            if self.full_game is None:
                self._update_full_game()
            
            # Skip deviation checks if full game is still not available
            if self.full_game is None:
                logger.warning("Full game not available yet, cannot check for beneficial deviations")
                return
            
            # Otherwise, look for beneficial deviations to expand the subgame
            best_responses = self._find_beneficial_deviations(full_eq_mixture, subgame.strategies)
            
            if best_responses:
                logger.info(f"Found {len(best_responses)} beneficial deviations from single-strategy equilibrium: {[self.strategy_names[br] for br in best_responses]}")
                
                # For each beneficial deviation, create a new expanded subgame
                for br in best_responses:
                    new_strategies = frozenset(subgame.strategies.union({br}))
                    # Only add if not already explored
                    if new_strategies not in self.explored_subgames:
                        # Priority is based on the gain from deviation
                        gains = self.full_game.deviation_gains(full_eq_mixture)
                        priority = gains[br].item()
                        new_subgame = SubgameInfo(
                            strategies=new_strategies,
                            priority=priority,
                            is_complete=False,
                            eq_mixture=None,
                            regret=float('inf')
                        )
                        logger.info(f"Building up: Adding expanded subgame with strategies {[self.strategy_names[i] for i in new_strategies]}")
                        self._add_to_queue(new_subgame)
            else:
                logger.info(f"No beneficial deviations found for single-strategy subgame")
            
            return
        
        # Create the empirical game for this subgame
        subgame_empirical = self._create_subgame(subgame.strategies)
        logger.info(f"Created empirical subgame with {len(subgame.strategies)} strategies")
        
        # Solve for equilibrium in the subgame
        eq_mixture, eq_regret, _ = find_equilibria(
            subgame_empirical, 
            method=self.eq_method,
            num_restarts=5,
            logging=True
        )
        logger.info(f"Found equilibrium in subgame with regret {eq_regret:.6f}")
        
        # Check if eq_mixture is None (equilibrium solving failed)
        if eq_mixture is None:
            logger.warning(f"Failed to find equilibrium in subgame with strategies {[self.strategy_names[i] for i in subgame.strategies]}")
            logger.warning("Using uniform mixture as a fallback")
        
        # The equilibrium mixture is for the subgame, expand it to the full strategy space
        full_eq_mixture = self._expand_mixture(eq_mixture, subgame.strategies)
        
        # Add to our list of equilibria
        self.equilibria.append((full_eq_mixture, eq_regret))
        
        # Mark this subgame as explored
        self.explored_subgames.add(subgame.strategies)
        
        # If we don't have a full game yet, update it
        if self.full_game is None:
            self._update_full_game()
            
        # Skip deviation checks if full game is still not available
        if self.full_game is None:
            logger.warning("Full game not available yet, cannot check for beneficial deviations")
            return
        
        # Check if this is a full-game equilibrium
        # Compute regret in the full game
        full_game_regret = self.full_game.regret(full_eq_mixture).item()
        logger.info(f"Subgame equilibrium regret: {eq_regret:.6f}, Full game regret: {full_game_regret:.6f}")
        
        # Find best responses and expand the subgame
        best_responses = self._find_beneficial_deviations(full_eq_mixture, subgame.strategies)
        
        if best_responses:
            logger.info(f"Found {len(best_responses)} beneficial deviations: {[self.strategy_names[br] for br in best_responses]}")
            
            # For each beneficial deviation, create a new expanded subgame
            for br in best_responses:
                new_strategies = frozenset(subgame.strategies.union({br}))
                # Only add if not already explored
                if new_strategies not in self.explored_subgames:
                    # Priority is based on the gain from deviation
                    gains = self.full_game.deviation_gains(full_eq_mixture)
                    priority = gains[br].item()
                    new_subgame = SubgameInfo(
                        strategies=new_strategies,
                        priority=priority,
                        is_complete=False,
                        eq_mixture=None,
                        regret=float('inf')
                    )
                    logger.info(f"Building up: Adding expanded subgame with strategies {[self.strategy_names[i] for i in new_strategies]}")
                    self._add_to_queue(new_subgame)
        
            # If we found any beneficial deviations, we're building up to a larger game
            if len(subgame.strategies) + 1 == self.num_strategies:
                logger.info(f"Building up to full game with all {self.num_strategies} strategies")
        else:
            # No beneficial deviations found, this is a local equilibrium
            logger.info(f"No beneficial deviations found for subgame - this is a local equilibrium")
            if full_game_regret < self.regret_threshold:
                # This is a confirmed equilibrium
                logger.info(f"Confirmed equilibrium with regret {full_game_regret:.6f}")
                
                # Check if this equilibrium is sufficiently different from existing ones
                is_new = True
                for existing_eq, _ in self.equilibria:
                    dist = torch.norm(existing_eq - full_eq_mixture).item()
                    if dist < self.distance_threshold:
                        is_new = False
                        break
                
                if is_new:
                    self.equilibria.append((full_eq_mixture, full_game_regret))
                    # Print the equilibrium mixture in readable form
                    eq_str = ", ".join([f"{self.strategy_names[i]}: {full_eq_mixture[i].item():.4f}" 
                                      for i in range(len(full_eq_mixture)) if full_eq_mixture[i] > 0.01])
                    logger.info(f"New equilibrium found: {eq_str}")
                else:
                    logger.info(f"Equilibrium is too similar to an existing one, skipping")
    
    def _add_to_queue(self, subgame):
        """Add a subgame to the priority queue."""
        # We negate priority because heapq is a min-heap, but we want max priority
        heapq.heappush(self.subgame_queue, (-subgame.priority, subgame))
        logger.info(f"Added subgame with strategies {[self.strategy_names[i] for i in subgame.strategies]}, priority {subgame.priority:.6f}")
    
    def _add_to_backup(self, subgame):
        """Add a subgame to the backup queue."""
        # We negate priority because heapq is a min-heap, but we want max priority
        heapq.heappush(self.backup_subgames, (-subgame.priority, subgame))
        logger.info(f"Added to backup: subgame with strategies {[self.strategy_names[i] for i in subgame.strategies]}, priority {subgame.priority:.6f}")
    
    def _generate_all_profiles(self, strategies):
        """
        Generate all possible profiles for a given set of strategies.
        Critically important to generate *all* valid distributions of players.
        """
        if not strategies:
            return []
            
        strategy_list = sorted(list(strategies))
        profiles = []
        
        # Handle the special case of a single strategy
        if len(strategy_list) == 1:
            profile = [0] * self.num_strategies
            profile[strategy_list[0]] = self.num_players
            profiles.append(profile)
            return profiles
        
        # For multiple strategies, we need to generate all distributions
        # Function to recursively generate all profiles
        def generate_recursive(remaining, idx, current_profile):
            if idx == len(strategy_list) - 1:
                # Last strategy gets the remaining players
                new_profile = current_profile.copy()
                new_profile[strategy_list[idx]] = remaining
                profiles.append(new_profile)
                return
                
            # Try different counts for the current strategy
            for count in range(remaining + 1):
                new_profile = current_profile.copy()
                new_profile[strategy_list[idx]] = count
                generate_recursive(remaining - count, idx + 1, new_profile)
        
        # Start with empty profile and fill recursively
        initial_profile = [0] * self.num_strategies
        generate_recursive(self.num_players, 0, initial_profile)
        
        # Log the generated profiles
        logger.info(f"Generated {len(profiles)} profiles for strategies {[self.strategy_names[i] for i in strategy_list]}")
        
        return profiles
    
    def _sample_subgame_profiles(self, strategies):
        """
        Ensure all required profiles for a subgame are sampled.
        Re-query the simulator if payoff data is missing.
        """
        logger.info(f"Sampling profiles for subgame with strategies {[self.strategy_names[i] for i in strategies]}")
        
        # Generate all possible profiles for this subgame
        profiles_to_sample = self._generate_all_profiles(strategies)
        
        if not profiles_to_sample:
            logger.warning(f"No profiles generated for strategies {[self.strategy_names[i] for i in strategies]}")
            return
        
        # Sample each profile
        for profile in profiles_to_sample:
            profile_tuple = tuple(profile)
            
            # Check if we need to sample this profile (not in cache or has invalid data)
            needs_sampling = False
            
            if profile_tuple not in self.payoff_data:
                needs_sampling = True
            else:
                # Check if any strategy has missing payoff data
                payoffs = self.payoff_data[profile_tuple]
                for i, count in enumerate(profile):
                    if count > 0:
                        strat_name = self.strategy_names[i]
                        if strat_name not in payoffs or payoffs[strat_name] is None:
                            needs_sampling = True
                            logger.info(f"Re-querying profile {profile_tuple} due to missing payoff for {strat_name}")
                            break
            
            if not needs_sampling:
                continue
                
            # Convert profile to strategy counts dictionary for the simulator
            strategy_counts = {self.strategy_names[idx]: count for idx, count in enumerate(profile) if count > 0}
            
            # Try to sample multiple times if needed
            max_attempts = 3
            for attempt in range(max_attempts):
                try:
                    # Run the simulation
                    logger.info(f"Simulating profile: {strategy_counts} (attempt {attempt+1}/{max_attempts})")
                    payoffs = self.simulator(strategy_counts)
                    
                    # Validate payoffs
                    valid_payoffs = True
                    for strat_name, count in strategy_counts.items():
                        if strat_name not in payoffs or payoffs[strat_name] is None:
                            valid_payoffs = False
                            logger.warning(f"Missing payoff for {strat_name} in profile {strategy_counts}")
                            break
                    
                    if valid_payoffs:
                        # Store the payoffs
                        self.payoff_data[profile_tuple] = payoffs
                        self.total_profiles_sampled += 1
                        
                        # Log the result
                        logger.info(f"Profile sampled successfully: {strategy_counts}, payoffs: {payoffs}")
                        break
                    else:
                        logger.warning(f"Invalid payoffs from simulation attempt {attempt+1}, retrying...")
                        if attempt == max_attempts - 1:
                            # On the last attempt, use zeros as fallback
                            logger.warning(f"Using zero payoffs as fallback for profile {strategy_counts}")
                            self.payoff_data[profile_tuple] = {name: 0.0 for name in strategy_counts.keys()}
                            self.total_profiles_sampled += 1
                except Exception as e:
                    logger.error(f"Error sampling profile {strategy_counts}: {e}")
                    if attempt == max_attempts - 1:
                        # On the last attempt, use zeros as fallback
                        logger.warning(f"Using zero payoffs as fallback for profile {strategy_counts}")
                        self.payoff_data[profile_tuple] = {name: 0.0 for name in strategy_counts.keys()}
                        self.total_profiles_sampled += 1
        
        # Update the full game with all data collected so far
        self._update_full_game()
    
    def _update_full_game(self):
        """
        Update the full game with all sampled profiles.
        Verify data completeness before updating.
        """
        # Check for profiles with incomplete data
        for profile_tuple, payoffs in list(self.payoff_data.items()):
            profile = list(profile_tuple)
            
            # Check if all used strategies have payoffs
            complete = True
            for i, count in enumerate(profile):
                if count > 0:
                    strat_name = self.strategy_names[i]
                    if strat_name not in payoffs or payoffs[strat_name] is None:
                        logger.warning(f"Profile {profile} has missing payoff for {strat_name}, re-querying")
                        complete = False
                        break
            
            if not complete:
                # Re-query this profile
                strategy_counts = {self.strategy_names[idx]: count for idx, count in enumerate(profile) if count > 0}
                try:
                    new_payoffs = self.simulator(strategy_counts)
                    self.payoff_data[profile_tuple] = new_payoffs
                except Exception as e:
                    logger.error(f"Error re-sampling profile {strategy_counts}: {e}")
                    # Use zero payoffs as fallback
                    self.payoff_data[profile_tuple] = {name: 0.0 for name in strategy_counts.keys()}
        
        # Now prepare data for SymmetricGame
        profiles = []
        payoffs_by_strategy = [[] for _ in range(self.num_strategies)]
        
        # Convert our payoff data to the format needed for SymmetricGame
        for profile_tuple, payoff_dict in self.payoff_data.items():
            profile = list(profile_tuple)
            profiles.append(profile)
            
            # For each strategy with players, record its payoff
            for strategy_idx, count in enumerate(profile):
                if count > 0:
                    strat_name = self.strategy_names[strategy_idx]
                    if strat_name in payoff_dict:
                        payoffs_by_strategy[strategy_idx].append(payoff_dict[strat_name])
                    else:
                        # Handle missing payoff data with zero as fallback
                        logger.warning(f"Missing payoff for strategy {strat_name} in profile {profile}, using 0.0")
                        payoffs_by_strategy[strategy_idx].append(0.0)
                else:
                    # No players using this strategy
                    payoffs_by_strategy[strategy_idx].append(0.0)
        
        # Create payoff and config tables
        if not profiles:
            logger.warning("No profiles to update the game with")
            return
            
        profiles_array = np.array(profiles)
        payoffs_array = np.zeros((self.num_strategies, len(profiles)))
        
        for i, payoffs in enumerate(payoffs_by_strategy):
            if payoffs:  # Only fill if we have data
                for j, p in enumerate(payoffs):
                    if j < len(profiles):
                        payoffs_array[i, j] = p
        
        # Create a new game or update the existing one
        if self.full_game is None:
            self.full_game = SymmetricGame(
                num_players=self.num_players,
                num_actions=self.num_strategies,
                config_table=profiles_array,
                payoff_table=payoffs_array,
                strategy_names=self.strategy_names,
                device=self.device
            )
            logger.info(f"Created full game with {len(profiles)} profiles")
        else:
            # Update existing game with new data
            raw_data = []
            for profile_tuple, payoff_dict in self.payoff_data.items():
                profile = list(profile_tuple)
                for strat_idx, count in enumerate(profile):
                    if count > 0:
                        strat_name = self.strategy_names[strat_idx]
                        if strat_name in payoff_dict:
                            # Add each player's payoff as a separate entry
                            for i in range(count):
                                raw_data.append((i, strat_name, payoff_dict[strat_name]))
            
            self.full_game.update_with_new_data(raw_data)
            logger.info(f"Updated full game with new profile data")
    
    def _create_subgame(self, strategies):
        """
        Create a subgame restricted to the given strategies.
        Re-query any missing payoff data.
        """
        if self.full_game is None:
            raise ValueError("Full game has not been initialized yet")
        
        # Check if we have complete data for all profiles in this subgame
        profiles_to_check = self._generate_all_profiles(strategies)
        missing_profiles = []
        
        for profile in profiles_to_check:
            profile_tuple = tuple(profile)
            if profile_tuple not in self.payoff_data:
                missing_profiles.append(profile)
                continue
            
            # Check if payoffs are complete
            payoffs = self.payoff_data[profile_tuple]
            for i, count in enumerate(profile):
                if count > 0:
                    strat_name = self.strategy_names[i]
                    if strat_name not in payoffs or payoffs[strat_name] is None:
                        missing_profiles.append(profile)
                        break
        
        # If we have missing profiles, re-sample them
        if missing_profiles:
            logger.info(f"Found {len(missing_profiles)} profiles with missing data, re-sampling...")
            
            for profile in missing_profiles:
                strategy_counts = {self.strategy_names[idx]: count for idx, count in enumerate(profile) if count > 0}
                try:
                    logger.info(f"Re-querying simulator for profile: {strategy_counts}")
                    payoffs = self.simulator(strategy_counts)
                    self.payoff_data[tuple(profile)] = payoffs
                except Exception as e:
                    logger.error(f"Error re-sampling profile {strategy_counts}: {e}")
                    # Use zero payoffs as fallback
                    self.payoff_data[tuple(profile)] = {name: 0.0 for name in strategy_counts.keys()}
            
            # Update the full game with new data
            self._update_full_game()
        
        # Convert strategies to tensor
        strategy_indices = torch.tensor(sorted(list(strategies)), device=self.device)
        
        try:
            # Create the restricted game
            restricted_game = self.full_game.restrict(strategy_indices)
            
            # Verify the restricted game has profiles
            if not hasattr(restricted_game, 'num_profiles') or restricted_game.num_profiles == 0:
                raise ValueError(f"Restricted game has no profiles for strategies {[self.strategy_names[i] for i in strategies]}")
            
            return restricted_game
        except Exception as e:
            logger.error(f"Error creating restricted game: {e}")
            raise ValueError(f"Failed to create subgame for strategies {[self.strategy_names[i] for i in strategies]}: {str(e)}")
    
    def _expand_mixture(self, mixture, subgame_strategies):
        """
        Expand a mixture from a restricted strategy space to the full strategy space.

        Args:
            mixture (torch.Tensor): The input mixture (or None if equilibrium wasn't found).
            subgame_strategies (list): The strategies in the subgame.

        Returns:
            torch.Tensor: The expanded mixture.
        """
        # Handle case where mixture is None (equilibrium not found)
        if mixture is None:
            # Return a uniform mixture over the given strategies
            full_mixture = torch.zeros(self.num_strategies, device=self.device)
            for i, s in enumerate(sorted(subgame_strategies)):
                full_mixture[s] = 1.0 / len(subgame_strategies)
            return full_mixture
            
        # Original code for when mixture is not None
        full_mixture = torch.zeros(self.num_strategies, device=self.device)
        for i, s in enumerate(sorted(subgame_strategies)):
            if i < len(mixture):
                full_mixture[s] = mixture[i]
        return full_mixture
    
    def _find_beneficial_deviations(self, mixture, strategies):
        """
        Find beneficial deviations from a mixture.
        This is key to the subgame search process - identifying strategies
        that should be added to the current subgame.
        """
        # Compute expected payoff of the mixture
        exp_payoff = self.full_game.expected_payoff(mixture).item()
        logger.info(f"Expected payoff of mixture: {exp_payoff:.6f}")
        
        # Compute payoffs for deviating to each pure strategy
        dev_payoffs = self.full_game.deviation_payoffs(mixture)
        
        # Find beneficial deviations (strategies not in the current subgame)
        beneficial_deviations = set()
        for strat_idx in range(self.num_strategies):
            if strat_idx not in strategies:
                # Check if deviation payoff exceeds expected payoff by more than threshold
                dev_payoff = dev_payoffs[strat_idx].item()
                
                if dev_payoff > exp_payoff + self.regret_threshold:
                    gain = dev_payoff - exp_payoff
                    beneficial_deviations.add(strat_idx)
                    logger.info(f"Found beneficial deviation to {self.strategy_names[strat_idx]}: "
                              f"payoff {dev_payoff:.6f} vs expected {exp_payoff:.6f} (gain: {gain:.6f})")
        
        return beneficial_deviations