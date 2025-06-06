"""
Empirical Game-Theoretic Analysis (EGTA) framework with Role Symmetric Game support.
"""
import os
import json
import time
from typing import Dict, List, Tuple, Any, Optional, Union, Set
import torch
import numpy as np
from datetime import datetime

from marketsim.egta.core.game import Game
from marketsim.egta.schedulers.base import Scheduler
from marketsim.egta.schedulers.random import RandomScheduler
from marketsim.egta.schedulers.dpr import DPRScheduler
from marketsim.egta.simulators.base import Simulator
from marketsim.egta.solvers.equilibria import quiesce, quiesce_sync, replicator_dynamics, regret
from marketsim.game.symmetric_game import SymmetricGame


class EGTA:
    """
    Empirical Game-Theoretic Analysis framework with Role Symmetric Game support.
    """
    
    def __init__(self, 
                simulator: Simulator,
                scheduler: Optional[Scheduler] = None,
                device: str = "cpu",
                output_dir: str = "results",
                max_profiles: int = 100,
                seed: Optional[int] = None):
        """
        Initialize the EGTA framework.
        
        Args:
            simulator: Simulator to use for evaluating profiles
            scheduler: Scheduler for determining which profiles to simulate
                If None, uses appropriate scheduler based on simulator type
            device: PyTorch device to use
            output_dir: Directory to save results
            max_profiles: Maximum number of profiles to simulate
            seed: Random seed
        """
        self.simulator = simulator
        self.device = device
        self.output_dir = output_dir
        self.max_profiles = max_profiles
        self.seed = seed
        
        # Detect if simulator supports role symmetric games
        self.is_role_symmetric = hasattr(simulator, 'get_role_info')
       
        
        if self.is_role_symmetric:
            # Role symmetric game setup
            self.role_names, self.num_players_per_role, self.strategy_names_per_role = simulator.get_role_info()
            self.num_players = sum(self.num_players_per_role)
            
            # Get all strategies for scheduler initialization
            all_strategies = []
            for strategies in self.strategy_names_per_role:
                all_strategies.extend(strategies)
        else:
            # Symmetric game setup
            self.role_names = ["Player"]
            self.num_players = simulator.get_num_players()
            self.num_players_per_role = [self.num_players]
            all_strategies = simulator.get_strategies()
            self.strategy_names_per_role = [all_strategies]
        
        # If scheduler is not provided, use appropriate default
        self.expected_strategies = set(           
            f"{r}:{s}" if self.is_role_symmetric else s
            for r, strats in zip(self.role_names, self.strategy_names_per_role)
            for s in strats
        )

        if scheduler is None:
            if self.is_role_symmetric:
                self.scheduler = DPRScheduler(
                    strategies=all_strategies,
                    num_players=self.num_players,
                    role_names=self.role_names,
                    num_players_per_role=self.num_players_per_role,
                    strategy_names_per_role=self.strategy_names_per_role,
                    seed=seed
                )
            else:
                self.scheduler = RandomScheduler(
                    strategies=all_strategies,
                    num_players=self.num_players,
                    seed=seed
                ) 
        else:
            self.scheduler = scheduler
        
        os.makedirs(output_dir, exist_ok=True)
        
        self.game = None
        self.payoff_data = []
        self.simulated_profiles = set()
        self.equilibria = []

    def has_profile(self, profile: List[Tuple[str,str]]) -> bool:
        key = tuple(sorted(profile))
        return key in self.game.payoff_table   # adjust to your storage structure

    
    def run(self, 
           max_iterations: int = 10, 
           profiles_per_iteration: int = 10,
           save_frequency: int = 1,
           verbose: bool = True,
           quiesce_kwargs: Optional[Dict] = None) -> Game:
        """
        Run the EGTA process.
        
        Args:
            max_iterations: Maximum number of iterations
            profiles_per_iteration: Number of profiles to simulate per iteration
            save_frequency: How often to save results (in iterations)
            verbose: Whether to print progress
            quiesce_kwargs: Optional dictionary of parameters to pass to quiesce_sync
            
        Returns:
            The final game
        """
        total_profiles = 0
        start_time = time.time()
        
        # Set default quiesce parameters if not provided
        if quiesce_kwargs is None:
            quiesce_kwargs = {
                'num_iters': 100,
                'num_random_starts': 10,
                'regret_threshold': 1e-3,
                'dist_threshold': 1e-2,
                'solver': 'replicator',
                'solver_iters': 5000
            }
        else:
            # Ensure all required parameters are present
            default_quiesce_kwargs = {
                'num_iters': 100,
                'num_random_starts': 60,
                'regret_threshold': 1e-2,
                'dist_threshold': 1e-2,
                'solver': 'replicator',
                'solver_iters': 5000
            }
            # Fill in any missing parameters with defaults
            for key, value in default_quiesce_kwargs.items():
                if key not in quiesce_kwargs:
                    quiesce_kwargs[key] = value
        
        # --------------------------------------------------------------------
        # BEGIN survey-compliant outer loop
        # --------------------------------------------------------------------
        for iteration in range(max_iterations):
            if verbose:
                print(f"\nIteration {iteration+1}/{max_iterations}")

            # ================================================================
            # 1)  MAKE SURE THE CURRENT RESTRICTED GAME IS COMPLETE
            # ================================================================
            while True:
                # get every profile the scheduler still wants
                batch = self.scheduler.get_next_batch(self.game)
                if not batch:
                    break  # nothing missing – restricted game is complete

                if verbose:
                    print(f"  Scheduling {len(batch)} new profiles")
                new_data = self.simulator.simulate_profiles(batch)

                # update game
                self.payoff_data.extend(new_data)
                if self.game is None:
                    self.game = Game.from_payoff_data(self.payoff_data, device=self.device)
                else:
                    self.game.update_with_new_data(new_data)

                self.scheduler.update(self.game)
                total_profiles += len(new_data)
                if total_profiles >= self.max_profiles:
                    break

            # ================================================================
            # 2)  BUILD CANDIDATE SET Ψ AND EXHAUSTIVELY EVALUATE UD(σ)
            # ================================================================
            candidates = self.scheduler._select_equilibrium_candidates(self.game, 50)
            confirmed, unconfirmed = [], []
            eps = quiesce_kwargs.get('regret_threshold', 1e-3)

            for σ in candidates:
                # ask scheduler for ALL missing one-player deviations of σ
                missing = self.scheduler.missing_deviations(σ, self.game)  # <- new helper
                if missing:
                    if verbose:
                        print(f"  {len(missing)} missing deviations → simulating")
                    new_data = self.simulator.simulate_profiles(missing)
                    self.payoff_data.extend(new_data)
                    self.game.update_with_new_data(new_data)
                    self.scheduler.update(self.game)
                    total_profiles += len(new_data)

                σ_reg = float(regret(self.game, torch.tensor(σ, dtype=torch.float32,
                                                            device=self.device)))
                if σ_reg <= eps:
                    confirmed.append((torch.tensor(σ,
                               dtype=torch.float32,
                               device=self.game.game.device),σ_reg))
                else:
                    unconfirmed.append((σ, σ_reg))

            if verbose:
                print(f"  Candidates – confirmed: {len(confirmed)}   "
                    f"unconfirmed: {len(unconfirmed)}")

           
            all_seen = (
                self.expected_strategies
                == self.game.strategies_present_in_payoff_table()   # <-- now a *real* test
            )
            if confirmed and not unconfirmed and all_seen:
                self.equilibria = confirmed
                if verbose:
                    print("  All strategies observed and all candidates confirmed – done.")
                break

           
            if verbose:
                print("  Expanding subgame …")

            # optional snapshot
            if iteration % save_frequency == 0:
                self._save_results(iteration)

            if total_profiles >= self.max_profiles:
                if verbose:
                    print(f"Reached profile budget ({self.max_profiles}).")
                break
        total_time = time.time() - start_time

        # --------------------------------------------------------------------
        # FINAL EQUILIBRIUM VERIFICATION VIA QUIESCE
        # --------------------------------------------------------------------
        try:
            if verbose:
                print("\nRunning QUIESCE final verification …")

            # Always pass a full-game reference. When DPR is not used, this is
            # just `self.game`; when DPR *is* used the reduced game should be
            # supplied as the first argument and the full game as `full_game`.
            # In our current architecture we only maintain the full game, so we
            # invoke quiesce on that object directly.

            quiesce_eqs = quiesce_sync(
                game=self.game,          # current empirical game (full)
                full_game=self.game,     # test deviations in the same game
                verbose=verbose,
                **quiesce_kwargs,
            )

            # quiesce_sync returns a list[(mixture, regret)]
            if quiesce_eqs:
                self.equilibria = quiesce_eqs
                if verbose:
                    print(f"QUIESCE found {len(self.equilibria)} equilibria")
        except Exception as e:
            if verbose:
                print(f"QUIESCE failed: {e}")

        # --------------------------------------------------------------------
        # PRINT SUMMARY AND RETURN
        # --------------------------------------------------------------------
        total_time = time.time() - start_time
        if verbose:
            print(f"\nEGTA completed in {total_time:.2f} seconds")
            print(f"Simulated {total_profiles} profiles")
            print(f"Found {len(self.equilibria)} equilibria")
        
        return self.game
    
    def _save_results(self, iteration: int):
        """
        Save current results to disk.
        
        Args:
            iteration: Current iteration
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save game
        game_path = os.path.join(self.output_dir, f"game_iter_{iteration}.json")
        self.game.save(game_path)
        
        # Save equilibria
        eq_path = os.path.join(self.output_dir, f"equilibria_iter_{iteration}.json")
        eq_data = []
        
        for i, (eq_mix, eq_regret) in enumerate(self.equilibria):
            eq_dict = {
                "id": i,
                "regret": float(eq_regret),
                "is_role_symmetric": self.is_role_symmetric
            }
            
            if self.is_role_symmetric:
                # Role symmetric equilibrium format
                eq_dict["mixture_by_role"] = {}
                global_idx = 0
                for role_name, role_strategies in zip(self.game.role_names, self.game.strategy_names_per_role):
                    role_mixture = {}
                    for strat_name in role_strategies:
                        if eq_mix[global_idx].item() > 0.001:
                            role_mixture[strat_name] = float(eq_mix[global_idx].item())
                        global_idx += 1
                    if role_mixture:
                        eq_dict["mixture_by_role"][role_name] = role_mixture
            else:
                # Symmetric equilibrium format
                eq_dict["mixture"] = {
                    name: float(eq_mix[j].item())
                    for j, name in enumerate(self.game.strategy_names)
                    if eq_mix[j].item() > 0.001
                }
            
            eq_data.append(eq_dict)
        
        with open(eq_path, 'w') as f:
            json.dump(eq_data, f, indent=2)
    
    def analyze_equilibria(self, verbose: bool = True) -> Dict[str, Any]:
        """
        Analyze the found equilibria.
        Args:
            verbose: Whether to print analysis
        Returns:
            Dictionary with analysis results
        """
        if not self.equilibria:
            if verbose:
                print("No equilibria found.")
            return {}
        
        support_sizes = []
        
        if self.is_role_symmetric:
            # Role symmetric analysis
            role_strategy_frequencies = {}
            for role_name in self.game.role_names:
                role_strategy_frequencies[role_name] = {}
                
            for eq_mix, _ in self.equilibria:
                global_idx = 0
                for role_name, role_strategies in zip(self.game.role_names, self.game.strategy_names_per_role):
                    for strat_name in role_strategies:
                        if strat_name not in role_strategy_frequencies[role_name]:
                            role_strategy_frequencies[role_name][strat_name] = 0.0
                        role_strategy_frequencies[role_name][strat_name] += eq_mix[global_idx].item() / len(self.equilibria)
                        global_idx += 1
                
                support = sum(1 for x in eq_mix if x.item() > 0.001)
                support_sizes.append(support)
        else:
            # Symmetric game analysis
            strategy_frequencies = {name: 0.0 for name in self.game.strategy_names}
            
            for eq_mix, _ in self.equilibria:
                support = sum(1 for x in eq_mix if x.item() > 0.001)
                support_sizes.append(support)
                
                for i, name in enumerate(self.game.strategy_names):
                    strategy_frequencies[name] += eq_mix[i].item() / len(self.equilibria)
        
        results = {
            "num_equilibria": len(self.equilibria),
            "regrets": [float(regret) for _, regret in self.equilibria],
            "avg_support_size": sum(support_sizes) / len(support_sizes),
            "min_support_size": min(support_sizes),
            "max_support_size": max(support_sizes),
            "is_role_symmetric": self.is_role_symmetric
        }
        
        if self.is_role_symmetric:
            results["role_strategy_frequencies"] = role_strategy_frequencies
        else:
            results["strategy_frequencies"] = strategy_frequencies
            sorted_strategies = sorted(
                strategy_frequencies.items(),
                key=lambda x: x[1],
                reverse=True
            )
            results["top_strategies"] = sorted_strategies[:3]
        
        if verbose:
            print("\nEquilibria Analysis")
            print(f"Number of equilibria: {results['num_equilibria']}")
            print(f"Average regret: {sum(results['regrets'])/len(results['regrets']):.6f}")
            print(f"Average support size: {results['avg_support_size']:.2f}")
            print(f"Support size range: {results['min_support_size']} - {results['max_support_size']}")
            
            if self.is_role_symmetric:
                print("\nStrategy frequencies by role:")
                for role_name, role_freqs in role_strategy_frequencies.items():
                    print(f"  {role_name}:")
                    sorted_role_strats = sorted(role_freqs.items(), key=lambda x: x[1], reverse=True)
                    for strat, freq in sorted_role_strats[:3]:
                        print(f"    {strat}: {freq:.4f}")
            else:
                print("\nTop strategies:")
                for strat, freq in results['top_strategies']:
                    print(f"  {strat}: {freq:.4f}")
        
        return results 

    def _create_reduced_game(self, full_game, reduction_size):
        """
        Create a reduced game with fewer players for DPR.
        
        Args:
            full_game: The full game with original number of players
            reduction_size: Number of players in the reduced game
            
        Returns:
            Reduced game with rescaled payoffs
        """
        if not isinstance(self.scheduler, DPRScheduler):
            return full_game
            
        if full_game.is_role_symmetric:
            # For role symmetric games, create reduced version by scaling payoffs
            # The underlying RSG should handle player reduction appropriately
            return full_game  # RSG already handles this internally
        else:
            # For symmetric games, use the original reduction logic
            scaling_factor = (full_game.num_players - 1) / (reduction_size - 1)
            
            reduced_sym_game = SymmetricGame(
                num_players=reduction_size,
                num_actions=full_game.game.num_actions,
                config_table=full_game.game.config_table.clone(),
                payoff_table=full_game.game.payoff_table.clone() / scaling_factor,  # Rescale payoffs
                strategy_names=full_game.game.strategy_names,
                device=full_game.game.device,
                offset=full_game.game.offset,
                scale=full_game.game.scale
            )
            
            # Create metadata for the reduced game
            metadata = {**full_game.metadata} if full_game.metadata else {}
            metadata['dpr_reduced'] = True
            metadata['original_players'] = full_game.num_players
            metadata['reduced_players'] = reduction_size
            metadata['scaling_factor'] = scaling_factor
            
            return Game(reduced_sym_game, metadata) 
    
  