"""
Empirical Game-Theoretic Analysis (EGTA) framework.
This package provides tools for game-theoretic analysis of strategic interactions.
"""

# Import core classes
from marketsim.egta.core.game import Game
from marketsim.egta.egta import EGTA

# Import solvers
from marketsim.egta.solvers.equilibria import (
    replicator_dynamics,
    fictitious_play,
    gain_descent,
    quiesce,
    regret,
    best_responses
)

# Import visualization functions
from marketsim.egta.visualization import (
    plot_equilibrium_payoffs,
    plot_strategy_traces,
    run_solver_with_trace,
    compare_solvers,
    visualize_quiesce_results
)

# Expose key functionality at the package level
__all__ = [
    # Core classes
    'Game', 'EGTA',
    
    # Solvers
    'replicator_dynamics', 'fictitious_play', 'gain_descent', 
    'quiesce', 'regret', 'best_responses',
    
    # Visualization
    'plot_equilibrium_payoffs', 'plot_strategy_traces',
    'run_solver_with_trace', 'compare_solvers', 'visualize_quiesce_results'
] 