"""
Empirical Game-Theoretic Analysis (EGTA) framework – lightweight top-level
initializer.

To avoid heavy dependencies (e.g. seaborn, ipywidgets) and circular imports, we
**lazily** expose public symbols the first time they are accessed instead of
importing all sub-modules at import time.
"""
from importlib import import_module
from types import ModuleType
from typing import Any, List

__all__: List[str] = [
    # Core classes
    "Game", "EGTA",
    
    # Solver helpers
    "replicator_dynamics", "fictitious_play", "gain_descent",
    "quiesce", "regret", "best_responses",
    
    # Visualization helpers (imported lazily – avoids seaborn dependency unless requested)
    "plot_equilibrium_payoffs", "plot_strategy_traces",
    "run_solver_with_trace", "compare_solvers", "visualize_quiesce_results",
]

# ---------------------------------------------------------------------------
# Lazy attribute resolver
# ---------------------------------------------------------------------------

def __getattr__(name: str) -> Any:  # noqa: D401
    """Dynamically load sub-modules on first access."""

    if name == "Game":
        mod: ModuleType = import_module("marketsim.egta.core.game")
        return getattr(mod, "Game")

    if name == "EGTA":
        mod = import_module("marketsim.egta.egta")
        return getattr(mod, "EGTA")

    # Solver utilities
    if name in {
        "replicator_dynamics", "fictitious_play", "gain_descent",
        "quiesce", "regret", "best_responses",
    }:
        mod = import_module("marketsim.egta.solvers.equilibria")
        return getattr(mod, name)

    # Visualization utilities (heavy – defer import until explicitly used)
    if name in {
        "plot_equilibrium_payoffs", "plot_strategy_traces",
        "run_solver_with_trace", "compare_solvers", "visualize_quiesce_results",
    }:
        mod = import_module("marketsim.egta.visualization")
        return getattr(mod, name)

    raise AttributeError(f"module 'marketsim.egta' has no attribute '{name}'")