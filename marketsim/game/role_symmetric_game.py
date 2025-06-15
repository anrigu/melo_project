import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from functools import lru_cache
import time
from abc import ABC, abstractmethod
from .abstract_game import AbstractGame
from collections import defaultdict, Counter
from marketsim.game.symmetric_game import SymmetricGame
# ← add this
from marketsim.egta.utils.observations import Observation



class RoleSymmetricGame(AbstractGame):
    """
    Represents a role-symmetric game.
    """
    def __init__(self,
                 role_names: List[str],
                 num_players_per_role: List[int],
                 strategy_names_per_role: List[List[str]],
                 rsg_config_table: Optional[torch.Tensor] = None,
                 rsg_payoff_table: Optional[torch.Tensor] = None,
                 device: str = "cpu",
                 offset: float = 0.0,
                 scale: float = 1.0):
        """
        Initialize a RoleSymmetricGame.

        Args:
            role_names: List of role names.
            num_players_per_role: List of integers, number of players for each role.
            strategy_names_per_role: List of lists of strings, strategy names for each role.
            rsg_config_table: Tensor of RSG configurations/profiles.
                              Shape: (num_rsg_configs, num_total_strategies)
            rsg_payoff_table: Tensor of payoffs for each strategy in each RSG configuration.
                              Shape: (num_total_strategies, num_rsg_configs)
            device: PyTorch device.
            offset: Offset for normalizing payoffs.
            scale: Scale for normalizing payoffs.
        """
        if not (len(role_names) == len(num_players_per_role) == len(strategy_names_per_role)):
            raise ValueError("Mismatch in lengths of role_names, num_players_per_role, and strategy_names_per_role")

        self.role_names: List[str] = role_names
        self.num_players_per_role: torch.Tensor = torch.tensor(num_players_per_role, dtype=torch.long, device=device)
        self.strategy_names_per_role: List[List[str]] = strategy_names_per_role
        self.device: str = device
        self.is_role_symmetric: bool = True

        self.num_roles: int = len(role_names)
        self.num_strategies_per_role: np.ndarray = np.array([len(s) for s in strategy_names_per_role], dtype=int)
        
        if np.any(self.num_strategies_per_role == 0):
            raise ValueError("Each role must have at least one strategy.")

        self.num_strategies: int = int(self.num_strategies_per_role.sum())

        self.role_starts: np.ndarray = np.concatenate(([0], self.num_strategies_per_role[:-1].cumsum())).astype(int)
        self.role_indices: np.ndarray = np.arange(self.num_roles).repeat(self.num_strategies_per_role)
        self.num_actions: int = self.num_strategies 

        self.rsg_config_table: Optional[torch.Tensor] = rsg_config_table.to(device) if rsg_config_table is not None else None
        self.rsg_payoff_table: Optional[torch.Tensor] = rsg_payoff_table.to(device) if rsg_payoff_table is not None else None
        
        if self.rsg_config_table is not None and self.rsg_config_table.shape[1] != self.num_strategies:
            raise ValueError(f"rsg_config_table num_strategies mismatch: expected {self.num_strategies}, got {self.rsg_config_table.shape[1]}")
        if self.rsg_payoff_table is not None and self.rsg_payoff_table.shape[0] != self.num_strategies:
            raise ValueError(f"rsg_payoff_table num_strategies mismatch: expected {self.num_strategies}, got {self.rsg_payoff_table.shape[0]}")
        if self.rsg_config_table is not None and self.rsg_payoff_table is not None and \
           self.rsg_config_table.shape[0] != self.rsg_payoff_table.shape[1]:
            raise ValueError("Mismatch in number of configurations between rsg_config_table and rsg_payoff_table")

        self.offset: float = offset
        self.scale: float = scale

    @classmethod
    def from_observations(
        cls,
        repo: "ObservationRepo",
        *,
        min_obs: int = 1,
        device: str = "cpu",
        normalize: bool = True,
    ) -> "RoleSymmetricGame":
        """Aggregate valid observations in *repo* into RSG tables."""

        role_names, num_players_per_role, strategy_names_per_role = repo.get_metadata()

        # Build global index map (role,strat) -> col idx
        global_idx: Dict[Tuple[str, str], int] = {}
        g_counter = 0
        for r, strat_list in zip(role_names, strategy_names_per_role):
            for s in strat_list:
                global_idx[(r, s)] = g_counter
                g_counter += 1
        n_strats = g_counter

        configs: List[Tuple[int, ...]] = []
        payoff_rows: List[List[float]] = []  # list over configs, each is len=n_strats

        for prof_key in repo.profiles():
            obs_list = repo.get(prof_key)
            if len(obs_list) < min_obs:
                continue

            # aggregate counts for this profile key once (they are identical across obs)
            counts = [0] * n_strats
            for (role, strat) in prof_key:
                counts[global_idx[(role, strat)]] += 1
            configs.append(tuple(counts))

            # collect payoffs per strat across observations
            by_strat: List[List[float]] = [[] for _ in range(n_strats)]
            for o in obs_list:
                for (role, strat), payoff in zip(o.profile, o.payoffs):
                    by_strat[global_idx[(role, strat)]].append(float(payoff))
            payoff_rows.append([np.mean(lst) if lst else np.nan for lst in by_strat])

        if not configs:
            raise ValueError("No profiles with ≥min_obs observations in repo.")

        payoff_arr = torch.tensor(payoff_rows, device=device, dtype=torch.float32)
        if normalize:
            mean = payoff_arr.nanmean(dim=0, keepdim=True)
            std = payoff_arr.nanstd(dim=0, keepdim=True).clamp_min_(1.0)
            payoff_arr = (payoff_arr - mean) / std
            offset, scale = float(mean.mean()), float(std.mean())
        else:
            offset, scale = 0.0, 1.0

        config_tensor = torch.tensor(configs, device=device, dtype=torch.float32)
        payoff_tensor = payoff_arr.transpose(0, 1)  # shape (n_strats, n_configs)

        return cls(
            role_names=role_names,
            num_players_per_role=num_players_per_role,
            strategy_names_per_role=strategy_names_per_role,
            rsg_config_table=config_tensor,
            rsg_payoff_table=payoff_tensor,
            device=device,
            offset=offset,
            scale=scale,
        )

    @classmethod
    def from_payoff_data_rsg(
        cls,
        payoff_data: List[List[Tuple[str, str, str, float]]], # Each item: (player_id_str, role_name, strategy_name, payoff_val)
        role_names: List[str],
        num_players_per_role: List[int],
        strategy_names_per_role: List[List[str]],
        device: str = "cpu",
        normalize_payoffs: bool = True
    ) -> 'RoleSymmetricGame':
        """
        Create a RoleSymmetricGame from raw payoff data.

        Args:
            payoff_data: List of simulated profiles. Each profile is a list of tuples:
                         (player_id_str, role_name, strategy_name, payoff_value).
                         Payoffs should be raw, unnormalized values.
            role_names: Definitive list of role names.
            num_players_per_role: Definitive list of player counts for each role, matching role_names.
            strategy_names_per_role: Definitive list of lists of strategy names, matching role_names.
            device: PyTorch device.
            normalize_payoffs: Whether to normalize payoffs (subtract mean, divide by std).

        Returns:
            A RoleSymmetricGame instance populated with config and payoff tables.
        """
        # Create a temporary instance to use its helper properties for mapping
        # This instance won't have payoff tables initially.
        temp_rsg_for_mapping = cls(
            role_names=role_names,
            num_players_per_role=num_players_per_role,
            strategy_names_per_role=strategy_names_per_role,
            device=device
        )
        
        role_to_idx = {name: i for i, name in enumerate(temp_rsg_for_mapping.role_names)}
        # Global strategy index mapping
        global_strat_to_idx: Dict[Tuple[str, str], int] = {}
        global_idx_counter = 0
        for r_idx, r_name in enumerate(temp_rsg_for_mapping.role_names):
            for s_name in temp_rsg_for_mapping.strategy_names_per_role[r_idx]:
                global_strat_to_idx[(r_name, s_name)] = global_idx_counter
                global_idx_counter += 1

        # Store aggregated payoffs and counts for each unique configuration
        # Key: tuple of global strategy counts (length num_total_strategies)
        # Value: dict {"payoffs": [list of payoffs for strat0, list for strat1, ...], "profile_count": int}
        processed_profiles = defaultdict(lambda: {
            "payoffs": [[] for _ in range(temp_rsg_for_mapping.num_strategies)],
            "profile_count": 0
        })

        all_payoffs_for_norm = []

        for sim_profile in payoff_data:
            # Validate profile structure for this RSG
            current_profile_role_player_counts = Counter([p_data[1] for p_data in sim_profile]) # Counts players per role_name
            expected_total_players = temp_rsg_for_mapping.num_players_per_role.sum().item()
            
            if len(sim_profile) != expected_total_players:
                # print(f"Warning: Skipping profile with {len(sim_profile)} players, expected {expected_total_players}.")
                continue # Skip malformed profile
            
            valid_profile = True
            for r_idx, r_name in enumerate(temp_rsg_for_mapping.role_names):
                if current_profile_role_player_counts[r_name] != temp_rsg_for_mapping.num_players_per_role[r_idx].item():
                    # print(f"Warning: Skipping profile with incorrect player count for role '{r_name}'. Got {current_profile_role_player_counts[r_name]}, expected {temp_rsg_for_mapping.num_players_per_role[r_idx].item()}.")
                    valid_profile = False
                    break
            if not valid_profile:
                continue

            # Determine global strategy counts for this simulation profile
            current_config_counts = [0] * temp_rsg_for_mapping.num_strategies
            for _, r_name, s_name, _ in sim_profile:
                if (r_name, s_name) not in global_strat_to_idx:
                    # print(f"Warning: Strategy '{s_name}' for role '{r_name}' in payoff data not found in game definition. Skipping profile.")
                    valid_profile = False; break
                global_s_idx = global_strat_to_idx[(r_name, s_name)]
                current_config_counts[global_s_idx] += 1
            if not valid_profile:
                continue
            
            config_tuple = tuple(current_config_counts)
            processed_profiles[config_tuple]["profile_count"] += 1

            for _, r_name, s_name, payoff_val in sim_profile:
                if payoff_val is not None and not np.isnan(payoff_val) and not np.isinf(payoff_val):
                    if (r_name, s_name) in global_strat_to_idx: # Should always be true if profile was valid
                        global_s_idx = global_strat_to_idx[(r_name, s_name)]
                        processed_profiles[config_tuple]["payoffs"][global_s_idx].append(payoff_val)
                        if normalize_payoffs:
                            all_payoffs_for_norm.append(payoff_val)
        
        if not processed_profiles:
            # print("Warning: No valid profiles found in payoff_data for RSG construction.")
            # Return an empty game or game with no payoff tables
            return cls(role_names, num_players_per_role, strategy_names_per_role, device=device)

        # Calculate normalization constants if needed
        payoff_mean = 0.0
        payoff_std = 1.0
        if normalize_payoffs and all_payoffs_for_norm:
            payoff_mean = np.mean(all_payoffs_for_norm)
            payoff_std = np.std(all_payoffs_for_norm)
            if payoff_std < 1e-9: # Avoid division by zero or too small std dev
                payoff_std = 1.0
        
        # Populate rsg_config_table and rsg_payoff_table
        num_unique_configs = len(processed_profiles)
        rsg_config_table_np = np.zeros((num_unique_configs, temp_rsg_for_mapping.num_strategies), dtype=np.float32)
        # payoff table: (num_total_strategies, num_unique_configs)
        rsg_payoff_table_np = np.full((temp_rsg_for_mapping.num_strategies, num_unique_configs), np.nan, dtype=np.float32)

        for i, (config_counts_tuple, data) in enumerate(processed_profiles.items()):
            rsg_config_table_np[i, :] = list(config_counts_tuple)
            for s_idx in range(temp_rsg_for_mapping.num_strategies):
                if data["payoffs"][s_idx]: # If there are payoffs for this strategy in this config
                    avg_payoff = np.mean(data["payoffs"][s_idx])
                    if normalize_payoffs:
                        rsg_payoff_table_np[s_idx, i] = (avg_payoff - payoff_mean) / payoff_std
                    else:
                        rsg_payoff_table_np[s_idx, i] = avg_payoff
                # Else, it remains NaN, indicating no payoff data for this strategy in this config

        rsg_config_table_tensor = torch.tensor(rsg_config_table_np, device=device, dtype=torch.float32)
        rsg_payoff_table_tensor = torch.tensor(rsg_payoff_table_np, device=device, dtype=torch.float32)

        return cls(
            role_names=role_names,
            num_players_per_role=num_players_per_role,
            strategy_names_per_role=strategy_names_per_role,
            rsg_config_table=rsg_config_table_tensor,
            rsg_payoff_table=rsg_payoff_table_tensor,
            device=device,
            offset=payoff_mean if normalize_payoffs else 0.0,
            scale=payoff_std if normalize_payoffs else 1.0
        )

    def _ensure_role_normalized_mixture(self, mixture: torch.Tensor, epsilon: float = 1e-9) -> torch.Tensor:
        """
        Ensures the mixture is normalized per role.
        Returns a new tensor, does not modify input mixture tensor unless it was not a tensor.
        """
        if not torch.is_tensor(mixture):
            # If not a tensor, convert and it will be a new object, so modification is fine.
            mixture_out = torch.tensor(mixture, dtype=torch.float64, device=self.device)
        elif mixture.dtype != torch.float64:
            mixture_out = mixture.to(dtype=torch.float64) # Ensure float64 for precision
        else:
            mixture_out = mixture.clone() # Clone to avoid modifying cached tensor or external input

        if mixture_out.shape != (self.num_strategies,):
            original_shape = mixture_out.shape
            mixture_out = mixture_out.squeeze()
            if mixture_out.shape != (self.num_strategies,):
                 raise ValueError(f"Mixture shape {original_shape} (squeezed to {mixture_out.shape}) incompatible with num_strategies {self.num_strategies}")

        for r_idx in range(self.num_roles):
            role_slice = slice(self.role_starts[r_idx], self.role_starts[r_idx] + self.num_strategies_per_role[r_idx])
            num_strats_in_role = self.num_strategies_per_role[r_idx]

            if num_strats_in_role == 0:
                continue

            role_part = mixture_out[role_slice]
            role_part = torch.clamp(role_part, min=0.0)
            role_sum = role_part.sum()

            if role_sum > epsilon:
                mixture_out[role_slice] = role_part / role_sum
            else:
                mixture_out[role_slice] = 1.0 / num_strats_in_role
        return mixture_out
        
    # ------------------------------------------------------------------
    # Internal helper that actually computes deviation payoffs (no Jacobian)
    # This code block is exactly the former implementation moved into a
    # separate function so we can reuse it for autograd-based Jacobian
    # computation without duplicating everything.
    # ------------------------------------------------------------------

    @lru_cache(maxsize=None)
    def _deviation_payoffs_internal(self, mixture_flat: torch.Tensor) -> torch.Tensor:  # noqa: C901
        """Role-aware deviation payoffs (no Jacobian)."""
        if self.rsg_config_table is None or self.rsg_payoff_table is None:
            return torch.zeros(self.num_strategies, device=self.device, dtype=torch.float32)

        mixture_internal = self._ensure_role_normalized_mixture(mixture_flat)

        dev_payoffs_output = torch.zeros(self.num_strategies, device=self.device, dtype=torch.float64)

        for s_k_idx in range(self.num_strategies):
            r_k = self.role_indices[s_k_idx]

            num_players_others_list = self.num_players_per_role.cpu().tolist()
            if num_players_others_list[r_k] < 1:
                dev_payoffs_output[s_k_idx] = 0.0
                continue
            num_players_others_list[r_k] -= 1
            num_players_others = torch.tensor(num_players_others_list, dtype=torch.long, device=self.device)

            expected_payoff_sk = 0.0

            for c_idx in range(self.rsg_config_table.shape[0]):
                profile_c = self.rsg_config_table[c_idx].to(dtype=torch.float64)
                if profile_c[s_k_idx] < 1.0 - 1e-9:
                    continue

                profile_others = profile_c.clone()
                profile_others[s_k_idx] -= 1.0

                # verify counts per role
                valid = True
                for r_idx in range(self.num_roles):
                    sl = slice(self.role_starts[r_idx], self.role_starts[r_idx] + self.num_strategies_per_role[r_idx])
                    if not torch.isclose(profile_others[sl].sum(), num_players_others[r_idx].to(torch.float64), atol=1e-6):
                        valid = False; break
                    if torch.any(profile_others[sl] < -1e-9):
                        valid = False; break
                if not valid:
                    continue

                log_p = 0.0
                can_form = True
                for r_idx in range(self.num_roles):
                    sl = slice(self.role_starts[r_idx], self.role_starts[r_idx] + self.num_strategies_per_role[r_idx])
                    counts_role = torch.round(profile_others[sl]).to(torch.float32)
                    probs_role  = mixture_internal[sl]

                    tot_players_role = int(num_players_others[r_idx].item())
                    if tot_players_role == 0:
                        if counts_role.sum() < 1e-9:
                            continue
                        else:
                            can_form = False; break

                    if torch.any(counts_role < -1e-9):
                        can_form = False; break

                    if not torch.isclose(counts_role.sum(), torch.tensor(float(tot_players_role), device=self.device)):
                        can_form = False; break

                    if torch.any(probs_role < -1e-9):
                        can_form = False; break

                    try:
                        m = torch.distributions.Multinomial(total_count=tot_players_role, probs=probs_role.to(torch.float32))
                        log_p += m.log_prob(counts_role)
                    except (ValueError, RuntimeError):
                        can_form = False; break

                if not can_form or torch.isinf(log_p) or torch.isnan(log_p):
                    continue

                p = torch.exp(log_p)
                if p > 1e-12:
                    payoff_val = self.rsg_payoff_table[s_k_idx, c_idx]
                    if not torch.isnan(payoff_val):
                        expected_payoff_sk += p * payoff_val.to(dtype=torch.float64)

            dev_payoffs_output[s_k_idx] = expected_payoff_sk

        return dev_payoffs_output.to(dtype=torch.float32)

    # ------------------------------------------------------------------
    # Public API: deviation_payoffs with optional Jacobian
    # ------------------------------------------------------------------
    def deviation_payoffs(self, mixture: torch.Tensor, *, jacobian: bool = False, ignore_incomplete: bool = False):
        """Deviation payoffs, optionally with Jacobian.

        Parameters
        ----------
        mixture : torch.Tensor
            Flat tensor of strategy probabilities (one per global strategy).
        jacobian : bool, default False
            If True, also returns the Jacobian matrix \(d dev / d mix\).
        """

        if not jacobian and not ignore_incomplete:
            # Fast path with caching
            return self._deviation_payoffs_internal(mixture)

        if not jacobian and ignore_incomplete:
            # Renormalise over observed-payoff mass only (no caching).
            return self._deviation_payoffs_internal_renorm(mixture)

        # --- Jacobian via autograd -----------------------------------
        mix_var = mixture.clone().detach().to(self.device).to(torch.float64)
        mix_var.requires_grad_(True)

        devs = self._deviation_payoffs_internal(mix_var)

        jac_rows = []
        for i in range(self.num_strategies):
            if mix_var.grad is not None:
                mix_var.grad.zero_()
            devs[i].backward(retain_graph=True)
            jac_rows.append(mix_var.grad.detach().clone())

        jac = torch.stack(jac_rows, dim=0).to(torch.float32)
        return devs.to(dtype=torch.float32).detach(), jac

    # ------------------------------------------------------------------
    # Helper that renormalises if some payoff entries are missing (NaN).
    #  *No* LRU cache because the result now depends on both `mixture` and the
    #    pattern of NaNs, but we keep it efficient by sharing most of the code.
    # ------------------------------------------------------------------

    def _deviation_payoffs_internal_renorm(self, mixture_flat: torch.Tensor) -> torch.Tensor:
        """Deviation payoffs with probability renormalisation.

        Any configuration where the focal strategy's payoff is NaN is *dropped*.
        The remaining probability mass is renormalised so that expectations are
        conditional on the payoff being observed.
        """

        if self.rsg_config_table is None or self.rsg_payoff_table is None:
            return torch.zeros(self.num_strategies, device=self.device, dtype=torch.float32)

        mixture_internal = self._ensure_role_normalized_mixture(mixture_flat)

        dev_payoffs = torch.full((self.num_strategies,), float('nan'), device=self.device, dtype=torch.float64)

        for s_k_idx in range(self.num_strategies):
            r_k = self.role_indices[s_k_idx]

            # players other than the deviator per role
            num_players_others_list = self.num_players_per_role.cpu().tolist()
            if num_players_others_list[r_k] < 1:
                dev_payoffs[s_k_idx] = 0.0; continue
            num_players_others_list[r_k] -= 1
            num_players_others = torch.tensor(num_players_others_list, dtype=torch.long, device=self.device)

            num = 0.0  # weighted payoff sum
            denom = 0.0  # total prob mass with finite payoff

            for c_idx in range(self.rsg_config_table.shape[0]):
                profile_c = self.rsg_config_table[c_idx].to(dtype=torch.float64)
                if profile_c[s_k_idx] < 1.0 - 1e-9:
                    continue

                profile_others = profile_c.clone(); profile_others[s_k_idx] -= 1.0

                # verify counts per role
                valid = True
                for r_idx in range(self.num_roles):
                    sl = slice(self.role_starts[r_idx], self.role_starts[r_idx] + self.num_strategies_per_role[r_idx])
                    if not torch.isclose(profile_others[sl].sum(), num_players_others[r_idx].to(torch.float64), atol=1e-6):
                        valid = False; break
                    if torch.any(profile_others[sl] < -1e-9):
                        valid = False; break
                if not valid:
                    continue

                # probability of seeing this profile under mixture (Multinomial × roles)
                log_p = 0.0; can_form = True
                for r_idx in range(self.num_roles):
                    sl = slice(self.role_starts[r_idx], self.role_starts[r_idx] + self.num_strategies_per_role[r_idx])
                    counts_role = torch.round(profile_others[sl]).to(torch.float32)
                    probs_role  = mixture_internal[sl]

                    tot_players_role = int(num_players_others[r_idx].item())
                    if tot_players_role == 0:
                        if counts_role.sum() < 1e-9:
                            continue
                        can_form = False; break

                    if torch.any(counts_role < -1e-9):
                        can_form = False; break
                    if not torch.isclose(counts_role.sum(), torch.tensor(float(tot_players_role), device=self.device)):
                        can_form = False; break

                    try:
                        m = torch.distributions.Multinomial(total_count=tot_players_role, probs=probs_role.to(torch.float32))
                        log_p += m.log_prob(counts_role)
                    except (ValueError, RuntimeError):
                        can_form = False; break

                if not can_form or torch.isinf(log_p) or torch.isnan(log_p):
                    continue

                payoff_val = self.rsg_payoff_table[s_k_idx, c_idx]
                if torch.isnan(payoff_val):
                    # skip but don't accumulate probability
                    continue

                p = torch.exp(log_p)
                if p <= 1e-12:
                    continue

                num += p * payoff_val.to(dtype=torch.float64)
                denom += p

            if denom > 1e-12:
                dev_payoffs[s_k_idx] = num / denom

        return dev_payoffs.to(dtype=torch.float32)

    # ------------------------------------------------------------------
    # Expected payoffs of the mixture itself (one value per role)
    # ------------------------------------------------------------------
    def mixture_values(self, mixture: torch.Tensor) -> torch.Tensor:
        """Return a length-`num_roles` tensor of expected payoffs.

        Each entry is the expected payoff for that role when every player
        sticks with *mixture* (i.e. \(\sum_s \pi_{r,s} \cdot \text{dev}_{r,s}(\pi)\)).
        """
        mix_norm = self._ensure_role_normalized_mixture(mixture).to(torch.float32)
        devs = self._deviation_payoffs_internal(mix_norm)

        role_vals = []
        offset = 0
        for r in range(self.num_roles):
            n_s = self.num_strategies_per_role[r]
            seg_mix  = mix_norm[offset:offset + n_s]
            seg_devs = devs[offset:offset + n_s]
            role_vals.append((seg_mix * seg_devs).sum())
            offset += n_s

        return torch.stack(role_vals)

    def get_strategy_name(self, global_strategy_index: int) -> str:
        """Returns the name of a strategy given its global index."""
        if not (0 <= global_strategy_index < self.num_strategies):
            raise ValueError(f"Invalid global_strategy_index: {global_strategy_index}")
        
        role_idx = self.role_indices[global_strategy_index]
        strategy_idx_within_role = global_strategy_index - self.role_starts[role_idx]
        return self.strategy_names_per_role[role_idx][strategy_idx_within_role]

    def get_role_and_strategy_index(self, global_strategy_index: int) -> Tuple[int, int]:
        """Returns the role index and strategy index within that role for a global strategy index."""
        if not (0 <= global_strategy_index < self.num_strategies):
            raise ValueError(f"Invalid global_strategy_index: {global_strategy_index}")
        
        role_idx = self.role_indices[global_strategy_index]
        strategy_idx_within_role = global_strategy_index - self.role_starts[role_idx]
        return role_idx, strategy_idx_within_role

    def regret(self, mixture: torch.Tensor) -> float:
        """
        Calculate the regret of a given mixture.
        Regret is the maximum gain from unilaterally deviating to a pure strategy.
        This needs to be role-aware. This overrides the AbstractGame.regret method.
        """
        dev_payoffs = self.deviation_payoffs(mixture)

        # If any payoff is NaN or Inf we cannot trust the regret; return large value
        if torch.isnan(dev_payoffs).any() or torch.isinf(dev_payoffs).any():
            return float('inf')

        # Ensure mixture is float32 for calculations here if it comes from outside
        if not torch.is_tensor(mixture):
            mixture = torch.tensor(mixture, dtype=torch.float32, device=self.device)
        elif mixture.dtype != torch.float32:
            mixture = mixture.to(dtype=torch.float32)

        normalized_mixture_for_regret = self._ensure_role_normalized_mixture(mixture).to(torch.float32)

        max_regret_val = 0.0 # Use a different name to avoid conflict with function name
        for r_idx in range(self.num_roles):
            role_slice = slice(self.role_starts[r_idx], self.role_starts[r_idx] + self.num_strategies_per_role[r_idx])
            if self.num_strategies_per_role[r_idx] == 0: continue

            role_mixture_segment = normalized_mixture_for_regret[role_slice]
            role_dev_payoffs = dev_payoffs[role_slice]
            
            expected_payoff_for_role = torch.sum(role_mixture_segment * torch.nan_to_num(role_dev_payoffs, nan=0.0))
            max_dev_payoff_for_role = torch.max(role_dev_payoffs) 
            if torch.isnan(max_dev_payoff_for_role) or torch.isinf(max_dev_payoff_for_role):
                # If max dev payoff is problematic, but expected is fine, regret is effectively infinite or undefined.
                # Or, if all strategies in role have NaN payoffs, treat max dev payoff as 0 for regret calc if E[payoff] is 0.
                if torch.isnan(expected_payoff_for_role) or torch.isinf(expected_payoff_for_role) or expected_payoff_for_role.item() != 0.0:
                    # If expected payoff is also NaN/inf, or not zero when max_dev is problematic, this role's regret is problematic
                    # Fallback to a high regret value to indicate issue or skip if this role has no actual plays in mixture
                    if role_mixture_segment.sum() > 1e-6 : # If this role is actually played
                        max_regret_val = max(max_regret_val, 1.0) # Assign high regret
                    continue # Or handle more gracefully
                else: # Expected payoff is 0, max_dev is problematic (e.g. all NaNs for this role)
                    max_dev_payoff_for_role = torch.tensor(0.0, device=self.device, dtype=torch.float32)
            
            current_role_regret = (max_dev_payoff_for_role - expected_payoff_for_role).item()
            if current_role_regret > max_regret_val:
                max_regret_val = current_role_regret
        
        return max_regret_val

    def best_responses(self, mixture: torch.Tensor, atol: float = 1e-8) -> torch.Tensor:
        """
        Find best responses to a mixture for each role.
        Overrides AbstractGame.best_responses to be role-aware.
        
        Args:
            mixture: Strategy mixture (flat tensor, will be role-normalized internally).
            atol: Absolute tolerance for considering strategies as best responses.
            
        Returns:
            Boolean tensor indicating which global strategies are best responses 
            (i.e., part of a best response for their respective role).
        """
        # Ensure mixture is valid and role-normalized for calculations
        # _ensure_role_normalized_mixture ensures it's float64, so convert dev_payoffs to float64 for comparison
        # However, deviation_payoffs returns float32. For consistency, work in float32 or ensure types match.
        # Let's assume deviation_payoffs returns float32 as per its current last line.
        # The mixture for best_responses should also be float32.

        if not torch.is_tensor(mixture):
            mixture_internal = torch.tensor(mixture, dtype=torch.float32, device=self.device)
        elif mixture.dtype != torch.float32:
            mixture_internal = mixture.to(dtype=torch.float32)
        else:
            mixture_internal = mixture # Use as is if already correct type

        # We need deviation payoffs based on the potentially non-normalized input mixture passed to dev_payoffs
        # because dev_payoffs handles its own internal normalization via _ensure_role_normalized_mixture.
        dev_payoffs = self.deviation_payoffs(mixture_internal) # Should be float32

        best_response_mask = torch.zeros_like(dev_payoffs, dtype=torch.bool)

        for r_idx in range(self.num_roles):
            role_slice = slice(self.role_starts[r_idx], self.role_starts[r_idx] + self.num_strategies_per_role[r_idx])
            if self.num_strategies_per_role[r_idx] == 0:
                continue

            role_dev_payoffs = dev_payoffs[role_slice]
            
            # Handle cases where all payoffs in a role might be NaN or -inf
            if torch.all(torch.isnan(role_dev_payoffs)) or torch.all(torch.isneginf(role_dev_payoffs)):
                # If all payoffs are problematic, no strategy is a best response, or all are if count is 1 and payoff is same NaN/-inf.
                # Or, could make all true if num_strats_in_role is 1.
                # For now, if all are NaN/inf, treat as no BR unless it's a single strategy role (then it is BR by definition).
                if self.num_strategies_per_role[r_idx] == 1:
                    best_response_mask[role_slice] = True
                continue 
            
            # Find max payoff for this role, ignoring NaNs if any exist (assuming non-NaN is better than NaN)
            # If all are NaN, this will be NaN. If some are NaN, max of non-NaNs.
            max_payoff_for_role = torch.max(role_dev_payoffs[~torch.isnan(role_dev_payoffs)])
            if torch.isnan(max_payoff_for_role) or torch.isinf(max_payoff_for_role): # if max is still nan/inf (e.g. all were nan/inf)
                 # If after filtering NaNs, max is still NaN/Inf, or if there were no non-NaN values.
                 # This case is tricky. If all payoffs are -inf, any is a BR.
                 # If all are NaN, it's undefined. For now, if max_payoff_for_role is still bad, mark no BRs unless single strat.
                 if self.num_strategies_per_role[r_idx] == 1 and (torch.isnan(role_dev_payoffs[0]) or torch.isinf(role_dev_payoffs[0])):
                     best_response_mask[role_slice] = True # Single strategy is BR by default
                 continue

            best_response_mask[role_slice] = torch.isclose(role_dev_payoffs, max_payoff_for_role, atol=atol, rtol=0) # rtol=0 for absolute
            # Ensure that NaNs in role_dev_payoffs do not become True due to isclose(NaN, value)
            best_response_mask[role_slice] = best_response_mask[role_slice] & (~torch.isnan(role_dev_payoffs))

        return best_response_mask

    def restrict(self, restriction_indices: List[int]) -> 'RoleSymmetricGame':
        """
        Creates a new game restricted to a subset of strategies.
        This is a complex operation that involves re-calculating roles, strategies,
        and filtering/re-indexing configuration and payoff tables.

        Args:
            restriction_indices: A list of GLOBAL strategy indices to keep.

        Returns:
            A new RoleSymmetricGame instance for the restricted game.
        """
        if not restriction_indices:
            raise ValueError("Restriction indices cannot be empty.")
        
        restriction_set = set(restriction_indices)
        if len(restriction_set) == self.num_strategies and sorted(list(restriction_set)) == list(range(self.num_strategies)):
            # No actual restriction, return a copy (or self if immutability is guaranteed, but safer to copy for tables)
            return RoleSymmetricGame(
                role_names=self.role_names.copy(),
                num_players_per_role=self.num_players_per_role.cpu().tolist(), # Keep as list for constructor
                strategy_names_per_role=[s_list.copy() for s_list in self.strategy_names_per_role],
                rsg_config_table=self.rsg_config_table.clone() if self.rsg_config_table is not None else None,
                rsg_payoff_table=self.rsg_payoff_table.clone() if self.rsg_payoff_table is not None else None,
                device=self.device,
                offset=self.offset,
                scale=self.scale
            )

        new_role_names: List[str] = []
        new_num_players_per_role: List[int] = []
        new_strategy_names_per_role: List[List[str]] = []
        
        # Mappings for the new restricted game
        # Global indices in the original game that are kept, and their new global index in the restricted game
        old_global_to_new_global_idx: Dict[int, int] = {}
        new_global_idx_counter = 0

        # Determine new role/strategy structure
        for r_idx, role_name in enumerate(self.role_names):
            current_role_original_strats = self.strategy_names_per_role[r_idx]
            restricted_role_strats: List[str] = []
            
            original_role_start_idx = self.role_starts[r_idx]
            
            for s_local_idx, strat_name in enumerate(current_role_original_strats):
                original_global_idx = original_role_start_idx + s_local_idx
                if original_global_idx in restriction_set:
                    restricted_role_strats.append(strat_name)
                    old_global_to_new_global_idx[original_global_idx] = new_global_idx_counter
                    new_global_idx_counter += 1
            
            if restricted_role_strats: # Only include role if it has strategies left
                new_role_names.append(role_name)
                new_num_players_per_role.append(self.num_players_per_role[r_idx].item())
                new_strategy_names_per_role.append(restricted_role_strats)

        if not new_role_names: # No roles left after restriction
            # print("Warning: Restriction results in a game with no roles/strategies.")
            return RoleSymmetricGame([], [], [], device=self.device, offset=self.offset, scale=self.scale)

        # New number of total strategies in the restricted game
        new_total_strategies = sum(len(s_list) for s_list in new_strategy_names_per_role)
        if new_total_strategies == 0:
             return RoleSymmetricGame(new_role_names, new_num_players_per_role, new_strategy_names_per_role, device=self.device, offset=self.offset, scale=self.scale)


        # --- Filter rsg_config_table and rsg_payoff_table --- 
        if self.rsg_config_table is None or self.rsg_payoff_table is None:
            # print("Warning: Original game has no payoff tables, restricted game will also have none.")
            return RoleSymmetricGame(
                new_role_names, new_num_players_per_role, new_strategy_names_per_role,
                device=self.device, offset=self.offset, scale=self.scale
            )

        # 1. Select columns from rsg_config_table for kept strategies
        kept_original_global_indices = sorted(list(old_global_to_new_global_idx.keys()))
        # Ensure indices are tensors for gather
        kept_original_global_indices_tensor = torch.tensor(kept_original_global_indices, dtype=torch.long, device=self.device) 

        # Create a temporary RSG object for the new structure to get its properties easily
        temp_restricted_rsg = RoleSymmetricGame(new_role_names, new_num_players_per_role, new_strategy_names_per_role, device=self.device)

        filtered_rsg_config_table_cols = self.rsg_config_table.index_select(1, kept_original_global_indices_tensor)

        valid_config_rows_mask = torch.ones(filtered_rsg_config_table_cols.shape[0], dtype=torch.bool, device=self.device)
        for i in range(filtered_rsg_config_table_cols.shape[0]):
            current_counts = filtered_rsg_config_table_cols[i, :]
            for r_new_idx in range(temp_restricted_rsg.num_roles):
                new_slice = slice(
                    temp_restricted_rsg.role_starts[r_new_idx],
                    temp_restricted_rsg.role_starts[r_new_idx] + temp_restricted_rsg.num_strategies_per_role[r_new_idx],
                )
                expected = temp_restricted_rsg.num_players_per_role[r_new_idx]
                actual = current_counts[new_slice].sum()
                if not torch.isclose(actual, expected.to(actual.dtype), atol=1e-6, rtol=1e-4):
                    valid_config_rows_mask[i] = False
                    break
        
        final_rsg_config_table = filtered_rsg_config_table_cols[valid_config_rows_mask, :]
        
        filtered_rsg_payoff_table_rows = self.rsg_payoff_table.index_select(0, kept_original_global_indices_tensor)
        final_rsg_payoff_table = filtered_rsg_payoff_table_rows[:, valid_config_rows_mask]

        if final_rsg_config_table.shape[0] == 0: #NOTE should never occur
            
            return RoleSymmetricGame(
                new_role_names, new_num_players_per_role, new_strategy_names_per_role, 
                device=self.device, offset=self.offset, scale=self.scale
            )

        return RoleSymmetricGame(
            role_names=new_role_names,
            num_players_per_role=new_num_players_per_role,
            strategy_names_per_role=new_strategy_names_per_role,
            rsg_config_table=final_rsg_config_table,
            rsg_payoff_table=final_rsg_payoff_table,
            device=self.device,
            offset=self.offset, 
            scale=self.scale
        )

    def to_symmetric_game_for_solver(self) -> SymmetricGame:
        """
        This cane be removed
        """
        from marketsim.game.symmetric_game import SymmetricGame # Local import to avoid circular dependency
        
        # print("Warning: Converting RoleSymmetricGame to a SymmetricGame for solver. This is a temporary hack.")

        flat_strategy_names = []
        for r_idx in range(self.num_roles):
            for s_idx_in_role in range(self.num_strategies_per_role[r_idx]):
                flat_strategy_names.append(f"{self.role_names[r_idx]}_{self.strategy_names_per_role[r_idx][s_idx_in_role]}")
        
        total_players = int(self.num_players_per_role.sum().item())

        sym_game = SymmetricGame(
            num_players=total_players,
            num_actions=self.num_strategies, 
            strategy_names=flat_strategy_names,
            device=self.device
        )
        return sym_game

    def __repr__(self):
        return (f"RoleSymmetricGame(roles={self.role_names}, players_per_role={self.num_players_per_role.tolist()}, "
                f"strats_per_role={self.num_strategies_per_role.tolist()})")

    def update_with_new_data(
        self,
        new_payoff_data: List[List[Tuple[str, str, str, float]]], 
        normalize_payoffs: bool = True 
    ):
        """
        Update the game with new payoff data for Role-Symmetric Games.
        This method will add new configurations or update existing ones with new payoff observations.
        If normalize_payoffs is True, it will re-calculate normalization constants (offset/scale)
        based on ALL data (old + new) and re-normalize the entire payoff table.

        Args:
            new_payoff_data: List of new simulated profiles. Each profile is
                             List[(player_id_str, role_name, strategy_name, payoff_value)].
            normalize_payoffs: Whether to (re-)normalize payoffs.
        """
        if not new_payoff_data:
            return

        # --- Step 1: Aggregate all current known raw payoffs and new raw payoffs --- 
        # This structure will hold all raw payoffs for every (config_tuple, strategy_index) pair
        # It allows for correct re-calculation of means and global normalization constants.
        all_aggregated_raw_payoffs = defaultdict(lambda: [[] for _ in range(self.num_strategies)])
        # This will store profile counts for each configuration found in old or new data.
        all_config_profile_counts = Counter()

        # Global strategy index mapping (re-created here for clarity, could be a helper)
        global_strat_to_idx: Dict[Tuple[str, str], int] = {}
        _idx_counter = 0
        for r_idx, r_name in enumerate(self.role_names):
            for s_name in self.strategy_names_per_role[r_idx]:
                global_strat_to_idx[(r_name, s_name)] = _idx_counter
                _idx_counter += 1

        # De-normalize and collect existing payoffs from the current payoff table
        if self.rsg_config_table is not None and self.rsg_payoff_table is not None:
            num_existing_configs = self.rsg_config_table.shape[0]
            for i in range(num_existing_configs):
                config_counts_tuple = tuple(self.rsg_config_table[i].cpu().long().tolist()) # Use long().tolist() for exact counts
             
                for s_idx in range(self.num_strategies):
                    if not torch.isnan(self.rsg_payoff_table[s_idx, i]):
                        # This is the stored (potentially normalized) mean payoff.
                        stored_mean_payoff = self.rsg_payoff_table[s_idx, i].item()
                        # De-normalize it. This is an estimate of one raw data point.
                        raw_payoff_estimate = stored_mean_payoff * self.scale + self.offset
                        all_aggregated_raw_payoffs[config_counts_tuple][s_idx].append(raw_payoff_estimate)
                
        for sim_profile in new_payoff_data:
            current_profile_role_player_counts = Counter([p_data[1] for p_data in sim_profile])
            expected_total_players = self.num_players_per_role.sum().item()
            if len(sim_profile) != expected_total_players: continue
            valid_profile = True
            for r_idx, r_name in enumerate(self.role_names):
                if current_profile_role_player_counts[r_name] != self.num_players_per_role[r_idx].item():
                    valid_profile = False; break
            if not valid_profile: continue

            current_config_counts = [0] * self.num_strategies
            valid_current_sim_profile = True
            raw_payoffs_for_this_sim_profile_by_strat = [[] for _ in range(self.num_strategies)]

            for _, r_name, s_name, payoff_val in sim_profile:
                if (r_name, s_name) not in global_strat_to_idx:
                    valid_current_sim_profile = False; break
                global_s_idx = global_strat_to_idx[(r_name, s_name)]
                current_config_counts[global_s_idx] += 1
                if payoff_val is not None and not np.isnan(payoff_val) and not np.isinf(payoff_val):
                    raw_payoffs_for_this_sim_profile_by_strat[global_s_idx].append(payoff_val)
                    # Collect for global normalization if chosen
                    # all_raw_payoffs_for_re_norm.append(payoff_val) # This should be done *after* deciding if to normalize
            if not valid_current_sim_profile: continue
            
            config_tuple = tuple(current_config_counts)
            all_config_profile_counts[config_tuple] += 1 # Count occurrences of this config
            for s_idx in range(self.num_strategies):
                all_aggregated_raw_payoffs[config_tuple][s_idx].extend(raw_payoffs_for_this_sim_profile_by_strat[s_idx])

        new_offset = self.offset
        new_scale = self.scale
        if normalize_payoffs:
            temp_all_raw_payoffs_list = []
            for config_key in all_aggregated_raw_payoffs:
                for s_idx_payoffs_list in all_aggregated_raw_payoffs[config_key]:
                    temp_all_raw_payoffs_list.extend(s_idx_payoffs_list)
            
            if temp_all_raw_payoffs_list: # Only if there's any data at all
                new_offset = np.mean(temp_all_raw_payoffs_list)
                new_scale = np.std(temp_all_raw_payoffs_list)
                if new_scale < 1e-9: new_scale = 1.0
            else: # No data to normalize from, use neutral values
                new_offset = 0.0
                new_scale = 1.0
        else: # Not normalizing, ensure offset/scale are neutral for new calculations
            new_offset = 0.0
            new_scale = 1.0

        # --- Step 3: Rebuild tables using the combined data and new normalization --- 
        if not all_aggregated_raw_payoffs: # No data at all (neither old nor new was valid)
            return 

        unique_configs_list = sorted(list(all_aggregated_raw_payoffs.keys())) # Ensure consistent order
        num_final_unique_configs = len(unique_configs_list)

        final_rsg_config_table_np = np.array(unique_configs_list, dtype=np.float32)
        final_rsg_payoff_table_np = np.full((self.num_strategies, num_final_unique_configs), np.nan, dtype=np.float32)

        for i, config_counts_tuple in enumerate(unique_configs_list):
            payoff_lists_for_config = all_aggregated_raw_payoffs[config_counts_tuple]
            for s_idx in range(self.num_strategies):
                if payoff_lists_for_config[s_idx]: # If there are payoffs for this strategy in this config
                    avg_raw_payoff = np.mean(payoff_lists_for_config[s_idx])
                    if normalize_payoffs: # Apply the newly calculated global normalization
                        final_rsg_payoff_table_np[s_idx, i] = (avg_raw_payoff - new_offset) / new_scale
                    else:
                        final_rsg_payoff_table_np[s_idx, i] = avg_raw_payoff
        
        self.rsg_config_table = torch.tensor(final_rsg_config_table_np, device=self.device, dtype=torch.float32)
        self.rsg_payoff_table = torch.tensor(final_rsg_payoff_table_np, device=self.device, dtype=torch.float32)
        self.offset = new_offset
        self.scale = new_scale
        
        # Clear lru_cache for deviation_payoffs as tables have changed
        if hasattr(self, 'deviation_payoffs') and hasattr(self.deviation_payoffs, 'cache_clear'):
            self.deviation_payoffs.cache_clear()

    def has_profile(self, profile: List[Tuple[str,str]]) -> bool:
        """
        Shortcut so callers can ask the RSG object directly.
        """
        # We already implemented the logic in the wrapper – just delegate.
        from marketsim.egta.core.game import Game   # avoid circular at top
        return Game(self).has_profile(profile)

