"""
Empirical Game-Theoretic Analysis (EGTA) framework.
This package provides tools for game-theoretic analysis of strategic interactions.

Avoid importing heavy submodules at import-time to prevent circular-import
problems (e.g. `marketsim.game.role_symmetric_game` ⇄ `marketsim.egta`).
Instead, expose the public symbols lazily the first time they are accessed.
"""

from importlib import import_module
from types import ModuleType
from typing import Any

__all__ = [
    # Core classes
    "Game", "EGTA",

    # Solvers
    "replicator_dynamics", "fictitious_play", "gain_descent",
    "quiesce", "regret", "best_responses",

    # Visualization helpers
    "plot_equilibrium_payoffs", "plot_strategy_traces",
    "run_solver_with_trace", "compare_solvers", "visualize_quiesce_results",
]


# ---------------------------------------------------------------------------
# Lazy attribute resolver
# ---------------------------------------------------------------------------


def __getattr__(name: str) -> Any:  # noqa: D401 (simple signature is fine)
    """Dynamically import sub-symbols on first access.

    This keeps the top-level package light-weight and prevents circular-import
    issues while still providing the convenient dotted access
    (`marketsim.egta.Game`, `marketsim.egta.quiesce`, …).
    """

    if name == "Game":
        mod: ModuleType = import_module("marketsim.egta.core.game")
        return getattr(mod, "Game")

    if name == "EGTA":
        mod = import_module("marketsim.egta.egta")
        return getattr(mod, "EGTA")

    if name in {
        "replicator_dynamics", "fictitious_play", "gain_descent",
        "quiesce", "regret", "best_responses",
    }:
        mod = import_module("marketsim.egta.solvers.equilibria")
        return getattr(mod, name)

    if name in {
        "plot_equilibrium_payoffs", "plot_strategy_traces",
        "run_solver_with_trace", "compare_solvers", "visualize_quiesce_results",
    }:
        mod = import_module("marketsim.egta.visualization")
        return getattr(mod, name)

    raise AttributeError(f"module 'marketsim.egta' has no attribute '{name}'")


# NOTE: mypy & IDEs can still discover the public API via __all__, and at
# runtime the lazy loader above will resolve the symbols on-demand. 