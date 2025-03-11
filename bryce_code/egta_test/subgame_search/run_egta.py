#!/usr/bin/env python3
"""
Main script to run EGTA with subgame search on the MELO simulator.

This script demonstrates how to use the subgame search algorithm
to analyze multi-agent strategies in the MELO market simulator.
"""

import os
import sys
import logging
import argparse
import torch
import numpy as np
from typing import Dict, List, Tuple

# Use absolute imports
from marketsim.egta_test.subgame_search.subgame_search import SubgameSearch
from marketsim.egta_test.subgame_search.melo_simulator_adapter import MeloSimulatorAdapter
from marketsim.egta_test.subgame_search.utils.eq_computation import find_equilibria
from marketsim.egta_test.subgame_search.reductions.dpr import DPRGAME

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('egta_subgame_search.log')
    ]
)

logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run EGTA with subgame search on the MELO simulator')
    
    parser.add_argument('--num-players', type=int, default=4,
                        help='Number of players in the game')
    parser.add_argument('--reduced-players', type=int, default=None,
                        help='Number of players to use in DPR reduction (if applicable)')
    parser.add_argument('--strategies', type=str, nargs='+', default=['ZI', 'MELO'],
                        help='Strategy names to analyze')
    parser.add_argument('--sim-time', type=int, default=500,
                        help='Simulation time steps')
    parser.add_argument('--num-simulations', type=int, default=5,
                        help='Number of simulations per profile for statistical significance')
    parser.add_argument('--max-iterations', type=int, default=20,
                        help='Maximum number of subgame search iterations')
    parser.add_argument('--regret-threshold', type=float, default=1e-3,
                        help='Maximum regret to consider an equilibrium valid')
    parser.add_argument('--eq-method', type=str, default='replicator_dynamics',
                        choices=['replicator_dynamics', 'ficticious_play', 'gain_descent', 'iterated_better_response'],
                        help='Equilibrium computation method')
    parser.add_argument('--use-dpr', action='store_true',
                        help='Use DPR (Deviation Preserving Reduction) if set')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda'],
                        help='Device to use for tensor operations')
    
    return parser.parse_args()

def print_equilibrium_info(eq_mixture, regret, strategy_names):
    """Print information about an equilibrium."""
    logger.info("=" * 50)
    logger.info("Equilibrium found with regret: %.6f", regret)
    
    # Format mixture as percentage
    mixture_pct = [f"{p * 100:.2f}%" for p in eq_mixture.numpy()]
    
    # Create a table of strategy probabilities
    table = []
    for i, (strat, prob) in enumerate(zip(strategy_names, mixture_pct)):
        table.append([i, strat, prob])
    
    # Print the table
    logger.info("Strategy mixture:")
    logger.info("  Index  |  Strategy  |  Probability")
    logger.info("---------|------------|-------------")
    for row in table:
        logger.info("  %d      |  %s      |  %s", row[0], row[1], row[2])
    logger.info("=" * 50)

def main():
    """Main function to run EGTA with subgame search."""
    args = parse_args()
    
    # Set up device
    device = torch.device(args.device)
    
    # Create the simulator adapter
    simulator_adapter = MeloSimulatorAdapter(
        num_background_agents=15,  # Placeholder, adjust as needed
        sim_time=args.sim_time,
        num_simulations=args.num_simulations,
        # Add any other simulator parameters here
    )
    
    # Create the subgame search instance
    subgame_search = SubgameSearch(
        strategy_names=args.strategies,
        simulator=simulator_adapter.simulate_profile,
        num_players=args.num_players,
        regret_threshold=args.regret_threshold,
        eq_method=args.eq_method,
        device=device
    )
    
    # Run the subgame search
    logger.info("Starting EGTA with subgame search...")
    equilibria = subgame_search.run(max_iterations=args.max_iterations)
    
    # Print results
    if equilibria:
        logger.info("Found %d equilibria:", len(equilibria))
        for i, (eq_mixture, regret) in enumerate(equilibria):
            logger.info("Equilibrium %d:", i+1)
            print_equilibrium_info(eq_mixture, regret, args.strategies)
    else:
        logger.info("No equilibria found.")
    
    # If DPR is requested and we have a full game, run DPR analysis
    if args.use_dpr and args.reduced_players and subgame_search.full_game:
        logger.info("Running DPR analysis...")
        
        # Create a DPR reduction of the full game
        reduced_players = args.reduced_players
        dpr_game = DPRGAME(
            full_game=subgame_search.full_game,
            reduced_players=reduced_players,
            device=device
        )
        
        # Compute equilibrium in the reduced game
        reduced_eq, reduced_regret, _ = find_equilibria(
            dpr_game,
            method=args.eq_method,
            num_restarts=5,
            logging=True
        )
        
        # Expand to the full game
        full_eq = dpr_game.expand_mixture(reduced_eq)
        
        logger.info("DPR Equilibrium:")
        print_equilibrium_info(full_eq, reduced_regret, args.strategies)
        
        # Validate in the full game
        if subgame_search.full_game:
            full_regret = subgame_search.full_game.regret(full_eq).item()
            logger.info("DPR Equilibrium regret in full game: %.6f", full_regret)
    
    logger.info("EGTA with subgame search completed.")

if __name__ == "__main__":
    main() 