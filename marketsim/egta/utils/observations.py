"""Lightweight record for a single simulator execution."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import time, uuid

# Canonical identity of a pure‑strategy profile
ProfileKey = Tuple[Tuple[str, str], ...]  # sorted (role,strategy) pairs

@dataclass
class Observation:
    profile: ProfileKey                    # immutable key
    payoffs: List[float]                   # len == #players (strategic only)
    meta: Dict[str, float] = field(default_factory=dict)
    valid: bool = True
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    ts: float = field(default_factory=time.time)

    def __post_init__(self):
        if not isinstance(self.profile, tuple):
            raise TypeError("profile must be a tuple of tuples, sorted")
        if not self.payoffs:
            raise ValueError("payoffs list may not be empty")