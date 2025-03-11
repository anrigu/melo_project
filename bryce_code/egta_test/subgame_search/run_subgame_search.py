import os
import sys
import torch
import logging
import numpy as np
import argparse
import traceback
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the parent directory to the path
sys.path.append(str(Path(__file__).parent.parent.parent))

from marketsim.egta_test.subgame_search.melo_simulator_adapter import MeloSimulatorAdapter
from marketsim.egta_test.subgame_search.subgame_search import SubgameSearch

def main(args):
    """
    Run the subgame search algorithm with the MELO simulator.
    """
    try:
        # Define the strategies for MELO agents - these must match what's in MeloSimulatorAdapter
        strategy_names = [
            "MELO_Only",      # Always chooses MELO
            "CDA_Only",       # Always chooses CDA
            "Balanced",       # 50/50 choice between markets
            "MELO_Biased",    # 75% MELO, 25% CDA
            "CDA_Biased"      # 25% MELO, 75% CDA
        ]
        
        logger.info(f"Creating simulator with {args.num_simulations} simulations per profile")
        
        # Create the simulator adapter
        simulator = MeloSimulatorAdapter(
            num_background_agents=args.background_agents,
            sim_time=args.sim_time,
            num_assets=args.num_assets,
            num_simulations=args.num_simulations
        )
        
        # Test the simulator with a simple profile
        test_profile = {strategy_names[0]: args.num_players}
        logger.info(f"Testing simulator with profile: {test_profile}")
        test_result = simulator.simulate_profile(test_profile)
        logger.info(f"Test simulation successful: {test_result}")
    except Exception as e:
        logger.error(f"Error testing simulator: {str(e)}")
        return
        
    logger.info(f"Creating subgame search with {args.num_players} players")
        
    # Create the subgame search object
    search = SubgameSearch(
            strategy_names=strategy_names,
            simulator=simulator,
            num_players=args.num_players,
            regret_threshold=args.regret_threshold,
            distance_threshold=args.distance_threshold,
            support_threshold=args.support_threshold,
            eq_method=args.eq_method,
            device=args.device
        )
        
        # Initialize and run the search
    logger.info("Initializing subgame search...")
    search.initialize()
        
    logger.info(f"Running subgame search with maximum {args.max_iterations} iterations...")
    equilibria = search.run(max_iterations=args.max_iterations)
        
    # Print results
    logger.info("Subgame search completed")
    logger.info(f"Found {len(equilibria)} equilibria")
        
    if not equilibria:
            logger.info("No equilibria found. This could be due to:")
            logger.info("1. Insufficient iterations to find equilibria")
            logger.info("2. Issues with the simulator or strategy space")
            logger.info("3. No pure equilibria exist in this game")
        
    for i, (eq_mixture, regret) in enumerate(equilibria):
            logger.info(f"Equilibrium {i+1}:")
            for j, prob in enumerate(eq_mixture):
                if prob > args.support_threshold:
                    logger.info(f"  {strategy_names[j]}: {prob.item():.4f}")
            logger.info(f"  Regret: {regret:.6f}")
        
    # Save results to file if specified
    if args.output_file:
            save_results(equilibria, strategy_names, args.output_file, args.support_threshold)
            logger.info(f"Results saved to {args.output_file}")
            


def save_results(equilibria, strategy_names, output_file, support_threshold=1e-4):
    """
    Save the equilibria to a file.
    """
    with open(output_file, 'w') as f:
        f.write("# MELO Market Game Equilibria\n\n")
        
        if not equilibria:
            f.write("No equilibria found.\n")
            return
        
        for i, (eq_mixture, regret) in enumerate(equilibria):
            f.write(f"## Equilibrium {i+1}\n")
            f.write(f"Regret: {regret:.6f}\n\n")
            f.write("Strategy | Probability\n")
            f.write("---------|------------\n")
            
            for j, prob in enumerate(eq_mixture):
                if prob > support_threshold:
                    f.write(f"{strategy_names[j]} | {prob.item():.4f}\n")
            
            f.write("\n")

def run_subgame_search(
    strategy_names,
    simulator,
    num_players,
    max_iterations=10,
    regret_threshold=1e-3,
    distance_threshold=0.1,
    support_threshold=1e-4,
    restricted_game_size=3,
    eq_method="replicator_dynamics",
    device="cpu",
    output_dir=None,
    verbose=True
):
    """
    Run the subgame search algorithm.
    
    Args:
        strategy_names: List of strategy names
        simulator: Simulator adapter
        num_players: Number of players
        max_iterations: Maximum number of iterations to run
        regret_threshold: Regret threshold for equilibria
        distance_threshold: Distance threshold for distinguishing equilibria
        support_threshold: Support threshold for equilibria
        restricted_game_size: Size of restricted games
        eq_method: Equilibrium finding method
        device: Device to use for tensor operations
        output_dir: Directory to save output
        verbose: Whether to print verbose output
        
    Returns:
        List of equilibria with regrets
    """
    # Create the subgame search object with the given parameters
    search = SubgameSearch(
        strategy_names=strategy_names,
        simulator=simulator,
        num_players=num_players,
        regret_threshold=regret_threshold,
        distance_threshold=distance_threshold,
        support_threshold=support_threshold,
        restricted_game_size=restricted_game_size,
        eq_method=eq_method,
        device=device
    )
    
    # Initialize and run the search
    search.initialize()
    
    # Make sure max_iterations is at least equal to the number of strategies
    # to give the algorithm a chance to build up larger subgames
    max_iterations = max(max_iterations, len(strategy_names) * 2)
    
    logger.info(f"Running subgame search with {max_iterations} iterations")
    equilibria = search.run(max_iterations=max_iterations)
    
    # Print the results
    if verbose:
        logger.info(f"Subgame search completed with {len(equilibria)} equilibria")
        for i, (eq_mixture, regret) in enumerate(equilibria):
            eq_str = ", ".join([f"{strategy_names[i]}: {eq_mixture[i].item():.4f}" 
                              for i in range(len(eq_mixture)) 
                              if eq_mixture[i] > 0.01])
            logger.info(f"Equilibrium {i+1}: {eq_str} (regret: {regret:.6f})")
            
        logger.info(f"Explored {search.subgames_explored} subgames")
        logger.info(f"Sampled {search.total_profiles_sampled} unique profiles")
    
    # Save output if requested
    if output_dir:
        save_results(search, equilibria, strategy_names, output_dir)
    
    return equilibria, search

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run subgame search for MELO market game")
    
    # MELO Simulator parameters
    parser.add_argument("--background-agents", type=int, default=15, 
                        help="Number of background agents in the simulation")
    parser.add_argument("--sim-time", type=int, default=500, 
                        help="Simulation time steps")
    parser.add_argument("--num-assets", type=int, default=1, 
                        help="Number of assets in the simulation")
    parser.add_argument("--num-simulations", type=int, default=10, 
                        help="Number of simulations to run for each profile")
    
    # Subgame search parameters
    parser.add_argument("--num-players", type=int, default=5, 
                        help="Number of strategic players in the game")
    parser.add_argument("--regret-threshold", type=float, default=1e-3, 
                        help="Maximum regret to consider an equilibrium valid")
    parser.add_argument("--distance-threshold", type=float, default=0.1, 
                        help="Minimum distance between equilibria to consider them distinct")
    parser.add_argument("--support-threshold", type=float, default=1e-4, 
                        help="Strategies with probabilities below this threshold will be truncated")
    parser.add_argument("--max-iterations", type=int, default=20, 
                        help="Maximum number of iterations for subgame search")
    parser.add_argument("--eq-method", type=str, default="replicator_dynamics", 
                        choices=["replicator_dynamics", "gain_descent", "iterated_better_response", "ficticious_play"],
                        help="Method to use for computing equilibria")
    parser.add_argument("--device", type=str, default="cpu", 
                        help="Device to use for torch operations")
    
    # Output parameters
    parser.add_argument("--output-file", type=str, default="subgame_search_results.md", 
                        help="File to save results to")
    
    # Error handling
    parser.add_argument("--skip-on-error", action="store_true", 
                        help="Continue execution even if errors occur")
    
    args = parser.parse_args()
    main(args) 