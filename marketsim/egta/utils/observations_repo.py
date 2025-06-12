"""In‑memory repository that stores every Observation and basic game metadata."""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple, Iterable
from statistics import stdev, mean

from observations import Observation, ProfileKey

class ObservationRepo:
    """Grow‑only store – append observations, aggregate on demand."""

    def __init__(
        self,
        role_names: List[str] | None = None,
        num_players_per_role: List[int] | None = None,
        strategy_names_per_role: List[List[str]] | None = None,
    ) -> None:
        self._store: Dict[ProfileKey, List[Observation]] = defaultdict(list)
        self._meta = {
            "role_names": role_names,
            "num_players_per_role": num_players_per_role,
            "strategy_names_per_role": strategy_names_per_role,
        }

    # ---------------- write ----------------
    def add(self, obs: Observation):
        self._store[obs.profile].append(obs)

    # ---------------- read helpers ----------------
    def get(self, profile: ProfileKey, *, valid_only: bool = True) -> List[Observation]:
        obs = self._store.get(profile, [])
        if valid_only:
            obs = [o for o in obs if o.valid]
        return obs

    def n_obs(self, profile: ProfileKey, *, valid_only: bool = True) -> int:
        return len(self.get(profile, valid_only=valid_only))

    def profiles(self) -> Iterable[ProfileKey]:
        return self._store.keys()

    # ---------------- metadata ----------------
    def set_metadata(
        self,
        role_names: List[str],
        num_players_per_role: List[int],
        strategy_names_per_role: List[List[str]],
    ) -> None:
        self._meta = {
            "role_names": role_names,
            "num_players_per_role": num_players_per_role,
            "strategy_names_per_role": strategy_names_per_role,
        }

    def get_metadata(self):
        if None in self._meta.values():
            raise ValueError("ObservationRepo metadata has not been initialised; call set_metadata().")
        return (
            self._meta["role_names"],
            self._meta["num_players_per_role"],
            self._meta["strategy_names_per_role"],
        )

