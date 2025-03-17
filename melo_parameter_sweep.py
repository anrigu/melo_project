"""
MELO Parameter Sweep Experiment

This script runs a parameter sweep to find MELO parameters that make it more competitive with CDA.
"""
import numpy as np
from marketsim.egta.simulators.melo_wrapper import MeloSimulator
from marketsim.egta.core.game import Game
import time
import json
from tqdm import tqdm

def run_single_experiment(params, num_players=10, reps=10):
    """Run a single experiment with the given parameters."""
    simulator = MeloSimulator(
        num_strategic=num_players,
        sim_time=params["sim_time"],
        lam=params["lam"],
        lam_melo=params["lam_melo"],
        mean=params["mean"],
        r=params["r"],
        q_max=params["q_max"],
        holding_period=params["holding_period"],
        reps=reps
    )
    
    strategies = simulator.get_strategies()
    
    profiles = [

        ["MELO_100_0"] * num_players,
   
        ["MELO_0_100"] * num_players
    ]
    
    # Simulate profiles
    print(f"Simulating profiles with parameters: {params}")
    results = simulator.simulate_profiles(profiles)
    
    # Calculate average payoffs for each strategy
    cda_payoffs = [payoff for _, strat, payoff in results[0] if strat == "MELO_100_0"]
    melo_payoffs = [payoff for _, strat, payoff in results[1] if strat == "MELO_0_100"]
    
    cda_avg = sum(cda_payoffs) / len(cda_payoffs) if cda_payoffs else 0
    melo_avg = sum(melo_payoffs) / len(melo_payoffs) if melo_payoffs else 0
    
    # Record results
    result = {
        "params": params,
        "cda_avg_payoff": cda_avg,
        "melo_avg_payoff": melo_avg,
        "ratio_melo_to_cda": melo_avg / cda_avg if cda_avg > 0 else 0,
        "difference": melo_avg - cda_avg
    }
    
    return result

def main():
    param_grid = {
        "sim_time": [1000],
        "lam": [0.05, 0.1],
        "lam_melo": [0.2, 0.5, 1.0],
        "mean": [1e6],
        "r": [0.01, 0.05],
        "q_max": [15],
        "holding_period": [5, 10, 20]
    }
    
    param_keys = list(param_grid.keys())
    param_values = list(param_grid.values())
    
    all_experiments = []
    for lam in param_grid["lam"]:
        for lam_melo in param_grid["lam_melo"]:
            for r in param_grid["r"]:
                for holding_period in param_grid["holding_period"]:
                    params = {
                        "sim_time": 1000,
                        "lam": lam,
                        "lam_melo": lam_melo,
                        "mean": 1e6,
                        "r": r,
                        "q_max": 15,
                        "holding_period": holding_period
                    }
                    all_experiments.append(params)
    
    print(f"Running {len(all_experiments)} parameter combinations")
    
    # Run experiments
    results = []
    start_time = time.time()
    
    for params in tqdm(all_experiments):
        try:
            result = run_single_experiment(params)
            results.append(result)
            
            print(f"CDA Avg: {result['cda_avg_payoff']:.2f}, MELO Avg: {result['melo_avg_payoff']:.2f}, Ratio: {result['ratio_melo_to_cda']:.4f}")
            
            with open("melo_parameter_sweep_results.json", "w") as f:
                json.dump(results, f, indent=2)
                
        except Exception as e:
            print(f"Error with parameters {params}: {e}")
    
    results.sort(key=lambda x: x["ratio_melo_to_cda"], reverse=True)
    
    print("\nBest parameter sets (highest MELO/CDA ratio):")
    for i, result in enumerate(results[:5]):
        params = result["params"]
        print(f"\n{i+1}. MELO/CDA Ratio: {result['ratio_melo_to_cda']:.4f}")
        print(f"   CDA Avg: {result['cda_avg_payoff']:.2f}, MELO Avg: {result['melo_avg_payoff']:.2f}")
        print(f"   Parameters: lam={params['lam']}, lam_melo={params['lam_melo']}, r={params['r']}, holding_period={params['holding_period']}")
    
    total_time = time.time() - start_time
    print(f"\nTotal time: {total_time:.1f} seconds")

if __name__ == "__main__":
    main() 