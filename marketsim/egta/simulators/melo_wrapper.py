"""
MELO simulator interface for EGTA with Role Symmetric Game support.
"""
import random
import numpy as np
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Any, Optional, Union
from tqdm import tqdm
from marketsim.egta.simulators.base import Simulator
from marketsim.simulator.melo_simulator import MELOSimulatorSampledArrival
from marketsim.agent.zero_intelligence_agent import ZIAgent
from marketsim.agent.melo_agent import MeloAgent
from marketsim.fourheap.constants import BUY, SELL
import concurrent.futures as cf  # ↳ for parallel repetitions


# ---------------------------------------------------------------------------
# Helper for parallel execution (must be top-level to be picklable)
# ---------------------------------------------------------------------------


def _run_melo_single_rep(args):
    """Run one MELO simulation repetition and return the `values` dict."""

    (
        rep_seed,
        sim_kwargs,
    ) = args

    # Local seeding to avoid identical streams across workers
    import random, numpy as np

    random.seed(rep_seed)
    np.random.seed(rep_seed & 0xFFFF_FFFF)  # within 32-bit range

    sim = MELOSimulatorSampledArrival(**sim_kwargs)
    sim.run()
    values = sim.end_sim()[0]  # first element is payoff dict
    return values


class MeloSimulator(Simulator):
    """
    Interface to the MELO simulator for EGTA with Role Symmetric Game support.
    Now supports strategic ZI agents that can choose market allocation.
    """
    
    def __init__(self, 
                num_strategic_mobi: int = 10,
                num_strategic_zi: int = 30,
                sim_time: int = 10000,
                num_assets: int = 1,
                lam: float = 6e-3,
                mean: float = 1e6,
                r: float = 0.05,
                shock_var: float = 1e4,
                q_max: int = 15,
                pv_var: float = 5e6,
                shade: Optional[List[float]] = None,
                eta: float = 0.5,
                lam_r: Optional[float] = None,
                holding_period: int = 10,
                lam_melo: float = 1e-3,
                # Background agents (non-strategic)
                num_background_zi: int = 0,
                num_background_hbl: int = 0,
                reps: int = 50,
                # Role-specific strategies
                mobi_strategies: Optional[List[str]] = None,
                zi_strategies: Optional[List[str]] = None,
                # Mode control
                force_symmetric: bool = False):
        """
        Initialize the MELO simulator interface for role symmetric games.
        
        Args:
            num_strategic_mobi: Number of strategic MOBI agents
            num_strategic_zi: Number of strategic ZI agents  
            sim_time: Simulation time
            num_assets: Number of assets
            lam: Arrival rate
            mean: Mean fundamental value
            r: Mean reversion rate
            shock_var: Shock variance
            q_max: Maximum quantity
            pv_var: Private value variance
            shade: Shade parameters
            eta: Eta parameter
            lam_r: Arrival rate for regular traders
            holding_period: Holding period
            lam_melo: Arrival rate for MELO traders
            num_background_zi: Number of non-strategic ZI agents
            num_background_hbl: Number of non-strategic HBL agents
            reps: Number of simulation repetitions
            mobi_strategies: Strategy names for MOBI role (if None, uses default)
            zi_strategies: Strategy names for ZI role (if None, uses default)
            force_symmetric: If True, forces symmetric game behavior (disables role symmetric mode)
        """
        self.num_strategic_mobi = num_strategic_mobi
        self.num_strategic_zi = num_strategic_zi
        self.sim_time = sim_time
        self.num_assets = num_assets
        self.lam = lam
        self.mean = mean
        self.r = r
        self.shock_var = shock_var
        self.q_max = q_max
        self.pv_var = pv_var
        self.shade = shade or [10, 30]
        self.eta = eta
        self.lam_r = lam_r or lam
        self.holding_period = holding_period
        self.lam_melo = lam_melo
        self.num_background_zi = num_background_zi
        self.num_background_hbl = num_background_hbl
        self.reps = reps
        self.order_quantity = 5  # Fixed order quantity for MOBI traders
        self.force_symmetric = force_symmetric
        
        # Define role names and player counts
        self.role_names = ["MOBI", "ZI"]
        self.num_players_per_role = [num_strategic_mobi, num_strategic_zi]
        
        # Define strategies for each role
        if mobi_strategies is None:
            self.mobi_strategies = [
                "MOBI_100_0",   # 100% CDA, 0% MELO
                "MOBI_75_25",   # 75% CDA, 25% MELO
                "MOBI_50_50",   # 50% CDA, 50% MELO
                "MOBI_25_75",   # 25% CDA, 75% MELO
                "MOBI_0_100",   # 0% CDA, 100% MELO
            ]
        else:
            self.mobi_strategies = mobi_strategies
            
        if zi_strategies is None:
            self.zi_strategies = [
                "ZI_100_0",     # 100% CDA, 0% MELO
                "ZI_75_25",     # 75% CDA, 25% MELO
                "ZI_50_50",     # 50% CDA, 50% MELO
                "ZI_25_75",     # 25% CDA, 75% MELO
                "ZI_0_100",     # 0% CDA, 100% MELO
            ]
        else:
            self.zi_strategies = zi_strategies
        
        self.strategy_names_per_role = [self.mobi_strategies, self.zi_strategies]
        
        # Define strategy parameters for both roles
        self.strategy_params = {}
        
        # MOBI strategy parameters
        for i, strategy in enumerate(self.mobi_strategies):
            if "100_0" in strategy:
                self.strategy_params[strategy] = {"cda_proportion": 1.0, "melo_proportion": 0.0}
            elif "75_25" in strategy:
                self.strategy_params[strategy] = {"cda_proportion": 0.75, "melo_proportion": 0.25}
            elif "50_50" in strategy:
                self.strategy_params[strategy] = {"cda_proportion": 0.5, "melo_proportion": 0.5}
            elif "25_75" in strategy:
                self.strategy_params[strategy] = {"cda_proportion": 0.25, "melo_proportion": 0.75}
            elif "0_100" in strategy:
                self.strategy_params[strategy] = {"cda_proportion": 0.0, "melo_proportion": 1.0}
        
        # ZI strategy parameters (same allocation logic)
        for i, strategy in enumerate(self.zi_strategies):
            if "100_0" in strategy:
                self.strategy_params[strategy] = {"cda_proportion": 1.0, "melo_proportion": 0.0}
            elif "75_25" in strategy:
                self.strategy_params[strategy] = {"cda_proportion": 0.75, "melo_proportion": 0.25}
            elif "50_50" in strategy:
                self.strategy_params[strategy] = {"cda_proportion": 0.5, "melo_proportion": 0.5}
            elif "25_75" in strategy:
                self.strategy_params[strategy] = {"cda_proportion": 0.25, "melo_proportion": 0.75}
            elif "0_100" in strategy:
                self.strategy_params[strategy] = {"cda_proportion": 0.0, "melo_proportion": 1.0}
        
        # In symmetric mode, remove role symmetric capabilities
        if self.force_symmetric:
            # Remove the get_role_info method to make EGTA treat this as a symmetric game
            if hasattr(self, 'get_role_info'):
                delattr(self, 'get_role_info')
    
    def get_num_players(self) -> int:
        """Get the total number of strategic players."""
        if self.force_symmetric:
            # In symmetric mode, return just the MOBI players (the strategic players we care about)
            return self.num_strategic_mobi
        else:
            # In role symmetric mode, return total strategic players
            return sum(self.num_players_per_role)
    
    def get_role_info(self) -> Tuple[List[str], List[int], List[List[str]]]:
        """Get role information for role symmetric games."""
        return self.role_names, self.num_players_per_role, self.strategy_names_per_role
    
    def get_strategies(self) -> List[str]:
        """Get all strategies across all roles (for backward compatibility)."""
        if self.force_symmetric:
            # In symmetric mode, return only MOBI strategies (the strategic players)
            return self.mobi_strategies
        else:
            # In role symmetric mode, return all strategies
            all_strategies = []
            for strategies in self.strategy_names_per_role:
                all_strategies.extend(strategies)
            return all_strategies
    
    def simulate_profile(self, profile: List[Tuple[str, str]]) -> List[Tuple[str, str, str, float]]:
        """
        Simulate a profile - handles both role symmetric and symmetric game formats.
        
        Args:
            profile: List of (role_name, strategy_name) tuples
            
        Returns:
            List of (player_id, role_name, strategy_name, payoff) tuples or 
            List of (player_id, strategy_name, payoff) tuples for symmetric mode
        """
        if self.force_symmetric:
            # In symmetric mode, treat all strategies as MOBI strategies
            # and all players as "Player" role
            strategy_counts = Counter(strategy_name for _, strategy_name in profile)
            
            # Map to role strategy counts - all MOBI, ZI stays as background
            role_strategy_counts = {"MOBI": strategy_counts, "ZI": Counter()}
            is_role_symmetric_profile = False  # Force symmetric output format
        else:
            # Detect if this is a role symmetric profile or symmetric profile
            profile_roles = set(role_name for role_name, _ in profile)
            is_role_symmetric_profile = len(profile_roles) > 1 or "Player" not in profile_roles
            
            if is_role_symmetric_profile:
                # Handle role symmetric profile format
                role_strategy_counts = {role: Counter() for role in self.role_names}
                for role_name, strategy_name in profile:
                    if role_name in role_strategy_counts:
                        role_strategy_counts[role_name][strategy_name] += 1
                    else:
                        print(f"Warning: Unknown role '{role_name}' in profile. Expected roles: {self.role_names}")
                        return []
            else:
                # Handle symmetric profile format - convert to role symmetric internally
                # All players are using strategies, but we need to map them to MOBI/ZI roles
                strategy_counts = Counter(strategy_name for _, strategy_name in profile)
                
                # Split strategies between MOBI and ZI based on strategy names
                role_strategy_counts = {"MOBI": Counter(), "ZI": Counter()}
                
                for strategy_name, count in strategy_counts.items():
                    if strategy_name.startswith("MOBI_"):
                        # This is a MOBI strategy
                        role_strategy_counts["MOBI"][strategy_name] = count
                    elif strategy_name.startswith("ZI_"):
                        # This is a ZI strategy  
                        role_strategy_counts["ZI"][strategy_name] = count
                    else:
                        # For strategies that don't have role prefix, try to map them
                        # For backward compatibility, assume they could be either type
                        # We'll assign them to MOBI role for simplicity
                        role_strategy_counts["MOBI"][strategy_name] = count
        
        # ------------------------------------------------------------------
        # PARALLEL REPETITIONS: spawn up to CPU count workers
        # ------------------------------------------------------------------

        # Common kwargs for every repetition
        num_background = self.num_background_zi + self.num_background_hbl

        base_kwargs = dict(
            num_background_agents=num_background,
            sim_time=self.sim_time,
            num_zi=self.num_background_zi,
            num_hbl=self.num_background_hbl,
            num_strategic=self.get_num_players(),
            num_assets=self.num_assets,
            lam=self.lam,
            mean=self.mean,
            r=self.r,
            shock_var=self.shock_var,
            q_max=self.q_max,
            pv_var=self.pv_var,
            shade=self.shade,
            eta=self.eta,
            lam_r=self.lam_r,
            holding_period=self.holding_period,
            lam_melo=self.lam_melo,
            role_strategy_counts=role_strategy_counts,
            strategy_params=self.strategy_params,
            role_names=self.role_names,
        )

        # Dispatch repetitions in parallel
        with cf.ProcessPoolExecutor() as pool:
            seeds = [random.randint(0, 2**32 - 1) for _ in range(self.reps)]
            iter_args = [(s, base_kwargs) for s in seeds]
            values_per_rep = list(tqdm(pool.map(_run_melo_single_rep, iter_args), total=self.reps))

        # ------------------------------------------------------------------
        # Aggregate results across repetitions
        # ------------------------------------------------------------------

        aggregated_results = []
        
        if not values_per_rep:
            print("Warning: All simulation repetitions failed! Returning default payoffs of 0.")
            player_id = 0
            if self.force_symmetric:
                # Symmetric mode: iterate through MOBI strategies only
                for role_name, strategy_name in profile:
                    # For symmetric games, use 3-element format: (player_id, strategy_name, payoff)
                    aggregated_results.append((f"player_{player_id}", strategy_name, 0.0))
                    player_id += 1
            else:
                for role_name, strategy_name in profile:
                    if is_role_symmetric_profile:
                        # Role symmetric format: (player_id, role_name, strategy_name, payoff)
                        aggregated_results.append((f"player_{player_id}", role_name, strategy_name, 0.0))
                    else:
                        # Symmetric format: (player_id, strategy_name, payoff)
                        aggregated_results.append((f"player_{player_id}", strategy_name, 0.0))
                    player_id += 1
            return aggregated_results
        
        # Calculate average payoffs
        player_id = 0
        if self.force_symmetric:
            # Symmetric mode: only process MOBI strategies
            for role_name, strategy_name in profile:
                # Get payoffs for this player across all repetitions
                payoffs = []
                for values in values_per_rep:
                    agent_id = num_background + player_id
                    if agent_id in values:
                        payoff = values[agent_id]
                        if isinstance(payoff, (int, float)) and not np.isnan(payoff) and not np.isinf(payoff):
                            payoffs.append(float(payoff))
                
                # Calculate average payoff
                if payoffs:
                    avg_payoff = sum(payoffs) / len(payoffs)
                else:
                    # agent made no trades; treat as zero-profit
                    avg_payoff = 0.0
                
                # Symmetric format: (player_id, strategy_name, payoff)
                aggregated_results.append((f"player_{player_id}", strategy_name, avg_payoff))
                player_id += 1
        else:
            # Original role symmetric/symmetric processing
            for role_name, strategy_name in profile:
                # Get payoffs for this player across all repetitions
                payoffs = []
                for values in values_per_rep:
                    agent_id = num_background + player_id
                    if agent_id in values:
                        payoff = values[agent_id]
                        if isinstance(payoff, (int, float)) and not np.isnan(payoff) and not np.isinf(payoff):
                            payoffs.append(float(payoff))
                
                # Calculate average payoff
                if payoffs:
                    avg_payoff = sum(payoffs) / len(payoffs)
                else:
                    # agent made no trades; treat as zero-profit
                    avg_payoff = 0.0
                
                if is_role_symmetric_profile:
                    # Role symmetric format: (player_id, role_name, strategy_name, payoff)
                    aggregated_results.append((f"player_{player_id}", role_name, strategy_name, avg_payoff))
                else:
                    # Symmetric format: (player_id, strategy_name, payoff)
                    aggregated_results.append((f"player_{player_id}", strategy_name, avg_payoff))
                player_id += 1
        
        return aggregated_results
    
    def simulate_profiles(self, profiles: List[List[Tuple[str, str]]]) -> List[List[Tuple[str, str, str, float]]]:
        """
        Simulate multiple role symmetric profiles.
        
        Args:
            profiles: List of role symmetric profiles
            
        Returns:
            List of lists of (player_id, role_name, strategy_name, payoff) tuples
        """
        results = []
        for i, profile in enumerate(profiles):
            print(f"Simulating role symmetric profile {i+1}/{len(profiles)}: {profile}")
            profile_results = self.simulate_profile(profile)
            results.append(profile_results)
        return results 