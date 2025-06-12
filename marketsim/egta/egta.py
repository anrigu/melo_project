"""
Empirical Game-Theoretic Analysis (EGTA) framework with Role Symmetric Game support.
"""
import os
import json
import time
from typing import Dict, List, Tuple, Any, Optional, Union, Set, Sequence
import torch
import numpy as np
from datetime import datetime
from dataclasses import dataclass
from collections import Counter
from marketsim.egta.core.game import Game
from marketsim.egta.schedulers.base import Scheduler
from marketsim.egta.schedulers.random import RandomScheduler
from marketsim.egta.schedulers.dpr import DPRScheduler
from marketsim.egta.simulators.base import Simulator
from marketsim.egta.solvers.equilibria import quiesce, quiesce_sync, replicator_dynamics, regret
from marketsim.game.symmetric_game import SymmetricGame
import math
from marketsim.egta.stats import statistical_test, hoeffding_upper_bound
from tqdm.auto import tqdm          # nice Jupyter/terminal handling
from contextlib import redirect_stdout, redirect_stderr
import io



class silence_io:
    """Context-manager that discards everything printed to stdout *and* stderr."""
    def __enter__(self):
        self._buf_out, self._buf_err = io.StringIO(), io.StringIO()
        self._redir_out = redirect_stdout(self._buf_out)
        self._redir_err = redirect_stderr(self._buf_err)
        self._redir_out.__enter__()
        self._redir_err.__enter__()
        return self

    def __exit__(self, *exc):
        # close the redirections and drop the buffers
        self._redir_out.__exit__(*exc)
        self._redir_err.__exit__(*exc)



@dataclass
class Observation:
    """A *single* simulator run for one pure profile.

    Attributes
    ----------
    profile_key : Tuple[int, ...]
        Hashable representation of the pure‑strategy profile.
    payoffs : np.ndarray  shape = (num_players,)
        realised utilities this round.
    aux : Optional[Dict[str, Any]] – any extra features you might emit.
    """
    profile_key: Tuple[int, ...]
    payoffs:     np.ndarray
    aux:         Optional[Dict[str, Any]] = None

# -----------------------------------------------------------------------------
#  UTILITIES – one‑sided Hoeffding test   (no external dependencies)
# -----------------------------------------------------------------------------



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
        self._init_storage()
        
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
    def _init_storage(self):
        self._obs_by_profile: Dict[Tuple[int, ...], List[Observation]] = {}
        self.payoff_data = []

    def _record_observations(self, raw_samples: List[Any]):
        """
        • Keep variance data in self._obs_by_profile   (Observation objects)
        • Keep legacy per-agent rows in self.payoff_data  (list[tuple]) so
        Game.from_payoff_data still works unmodified.
        """
        for item in raw_samples:
            # ------------------------------------------------------------------
            # 1. Obtain / build an Observation  (needed for statistical tests)
            # ------------------------------------------------------------------
            if isinstance(item, Observation):
                obs = item
                legacy_row = None            # we'll build it below
            elif isinstance(item, list) and item and len(item[0]) == 4:
                # list[(pid,role,strat,payoff)…]  →  Observation
                profile_key = tuple(sorted((r, s) for _pid, r, s, _ in item))
                payoffs = np.asarray([p for *_xyz, p in item], dtype=float)
                obs = Observation(profile_key, payoffs)
                legacy_row = item            # we can reuse it as-is
            else:  # dict {"profile":…, "payoffs":…}
                profile_key, payoff_vec = item["profile"], item["payoffs"]
                obs = Observation(tuple(profile_key), np.asarray(payoff_vec, dtype=float))
                legacy_row = None            # will build below

            # ------------------------------------------------------------------
            # 2. Store Observation for variance / hypothesis tests
            # ------------------------------------------------------------------
            self._obs_by_profile.setdefault(obs.profile_key, []).append(obs)

            # ------------------------------------------------------------------
            # 3. Ensure self.payoff_data gets a *legacy* row
            # ------------------------------------------------------------------
            if legacy_row is None:
                # Need to fabricate the list[(pid,role,strat,pay)] form
                legacy_row = []
                for idx, (role, strat) in enumerate(obs.profile_key):
                    payoff = float(obs.payoffs[idx])
                    legacy_row.append((f"p{idx}", role, strat, payoff))
            self.payoff_data.append(legacy_row)



    def has_profile(self, profile: List[Tuple[str,str]]) -> bool:
        key = tuple(sorted((r, s) for r, s in profile))
        return key in self.game.payoff_table   # adjust to your storage structure

    
   
    def run(
        self,
        max_iterations: int = 10,
        profiles_per_iteration: int = 10,      # retained for API compatibility
        save_frequency: int = 1,
        verbose: bool = True,
        quiesce_kwargs: Optional[Dict[str, Any]] = None,
    ) -> "Game":

        total_profiles, start_time = 0, time.time()

        # ------------------------------------------------------------------
        # Defaults for the *final* QUIESCE pass
        # ------------------------------------------------------------------
        if quiesce_kwargs is None:
            quiesce_kwargs = dict(
                num_iters=1,
                num_random_starts=10,
                regret_threshold=1e-3,
                dist_threshold=1e-2,
                solver="replicator",
                solver_iters=5_000,
            )
        eps = quiesce_kwargs["regret_threshold"]

        # ================================================================
        #                       MAIN EGTA LOOP
        # ================================================================
        for it in range(max_iterations):
            if verbose:
                print(f"\nIteration {it + 1}/{max_iterations}")

            # ------------------------------------------------------------
            # (1)  Simulate new profiles from the scheduler
            # ------------------------------------------------------------
            while True:
                batch = self.scheduler.get_next_batch(self.game)
                if not batch:
                    break

                if verbose:
                    for idx, profile in enumerate(
                        tqdm(batch, desc="Simulating profiles",
                            unit="profile", leave=False)
                    ):
                        role_mode = getattr(self, "is_role_symmetric", False)
                        counts = Counter(profile)
                        if role_mode:  # profile is list[(role, strat)]
                            parts = [f"{r}:{s}×{c}" for (r, s), c in counts.items()]
                        else:          # symmetric game: list[strat]
                            parts = [f"{s}×{c}" for s, c in counts.items()]
                        print(f"  • ({idx + 1}/{len(batch)}) [{', '.join(parts)}]")

                t0 = time.time()
                with silence_io():                          # <── mute noisy internals
                    raw = self.simulator.simulate_profiles(batch)
                if verbose:
                    print(f"    ✓ finished simulation in {time.time() - t0:.1f}s")

                prev = len(self.payoff_data)
                self._record_observations(raw)
                legacy_rows = self.payoff_data[prev:]

                if self.game is None:
                    self.game = Game.from_payoff_data(legacy_rows,
                                                    device=self.device)
                else:
                    self.game.update_with_new_data(legacy_rows)

                self.scheduler.update(self.game)
                total_profiles += len(batch)

                if total_profiles >= self.max_profiles:
                    break

            # ------------------------------------------------------------
            # (2)  Evaluate equilibrium candidates
            # ------------------------------------------------------------
            true_game = (
                self.game.full_game_reference
                if hasattr(self.game, "full_game_reference")
                and self.game.full_game_reference is not None
                else self.game
            )

            candidates = self.scheduler._select_equilibrium_candidates(
                self.game, 50)

            # —— inject joint-pure CDA / MELO mixes ———————————————
            extra_pures, pure_cda, pure_melo = [], \
                torch.zeros(self.game.num_strategies, device=self.device), \
                torch.zeros(self.game.num_strategies, device=self.device)

            for gidx, sname in enumerate(self.game.strategy_names):
                if "CDA"  in sname.upper():
                    pure_cda[gidx] = 1.0
                if "MELO" in sname.upper():
                    pure_melo[gidx] = 1.0

            if pure_cda.sum():
                extra_pures.append(role_aware_normalize(pure_cda,  self.game).tolist())
            if pure_melo.sum():
                extra_pures.append(role_aware_normalize(pure_melo, self.game).tolist())

            for mix in extra_pures:
                if not any(np.allclose(mix, c, atol=1e-6) for c in candidates):
                    candidates.append(mix)
            # ————————————————————————————————————————————————

            confirmed, unconfirmed = [], []
            for σ in candidates:
                # be sure every deviation of σ is present
                missing = self.scheduler.missing_deviations(σ, self.game)
                if missing:
                    with silence_io():
                        raw = self.simulator.simulate_profiles(missing)
                    prev = len(self.payoff_data)
                    self._record_observations(raw)
                    self.game.update_with_new_data(self.payoff_data[prev:])

                σ_reg = float(regret(
                    true_game,
                    torch.tensor(σ, dtype=torch.float32, device=self.device))
                )
                (confirmed if σ_reg <= eps else unconfirmed).append((σ, σ_reg))

            if confirmed:
                self.equilibria = [
                    (torch.tensor(σ, dtype=torch.float32, device=self.device), r)
                    for σ, r in confirmed
                ]

            if verbose:
                print(f"  Candidates – confirmed: {len(confirmed)}   "
                    f"unconfirmed: {len(unconfirmed)}")

            # early-exit if everything is settled
            all_seen = (
                self.expected_strategies ==
                self.game.strategies_present_in_payoff_table()
            )
            if confirmed and not unconfirmed and all_seen:
                if verbose:
                    print("  All strategies observed and all candidates confirmed – done.")
                break

            # ------------------------------------------------------------
            # (3)  Snapshot
            # ------------------------------------------------------------
            if it % save_frequency == 0:
                self._save_results(it)

            if total_profiles >= self.max_profiles:
                if verbose:
                    print(f"Reached profile budget ({self.max_profiles}).")
                break

        # ----------------------------------------------------------------
        # (4)  Single final QUIESCE verification
        # ----------------------------------------------------------------
        if verbose:
            print("\nRunning QUIESCE final verification …")

        try:
            q_args = dict(quiesce_kwargs,
                        num_iters=1,
                        dist_threshold=1e-3,
                        num_random_starts=0)

            eqs = quiesce_sync(
                game=self.game,          # (possibly reduced) working game
                full_game=true_game,     # verify vs. true/full game
                obs_store=self._obs_by_profile,
                verbose=verbose,
                **q_args,
            )
            if eqs:
                self.equilibria = eqs
                if verbose:
                    print(f"QUIESCE found {len(self.equilibria)} equilibria")
        except Exception as e:
            if verbose:
                print(f"QUIESCE failed: {e}")

        # ----------------------------------------------------------------
        # (5)  Summary
        # ----------------------------------------------------------------
        if verbose:
            elapsed = time.time() - start_time
            print(f"\nEGTA completed in {elapsed:.2f}s – "
                f"simulated {total_profiles} profiles – "
                f"found {len(self.equilibria)} equilibria")

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
    
  