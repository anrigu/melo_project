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

        #["MELO_100_0"] * num_players,
   
        ["MELO_0_100"] * num_players
    ]
    
    # Simulate profiles
    print(f"Simulating profiles with parameters: {params}")
    results = simulator.simulate_profiles(profiles)
    
    # --- Add accounting print statements --- 
    if hasattr(simulator, 'last_sim_instance') and simulator.last_sim_instance and hasattr(simulator.last_sim_instance, 'meloMarket'): 
        market = simulator.last_sim_instance.meloMarket
        order_book = market.order_book

        if hasattr(market, 'get_placed_count') and \
           hasattr(order_book, 'get_withdrawn_count') and \
           hasattr(order_book, 'get_unexecuted_info'):
            
            placed_count = market.get_placed_count()
           
            matched_qty = sum(mo.order.quantity for mo in market.matched_orders)
            withdrawn_count = order_book.get_withdrawn_count()
            unexecuted_info = order_book.get_unexecuted_info()
            unexecuted_qty = unexecuted_info['unexecuted_qty']
            unexecuted_count = unexecuted_info['unexecuted_count'] #number of distinct orders unexecuted

            print("--- Simulation Accounting (MELO Market) ---") #clarify scope
            print(f"  Orders Placed (in MELO): {placed_count}")
            matched_order_count = order_book.get_matched_order_count()
            print(f"  Orders Fully Matched (in MELO): {matched_order_count}") 
            print(f"  Total Quantity Matched (in MELO): {matched_qty}") #still print total quantity
            print(f"  Orders Withdrawn (from MELO): {withdrawn_count}")
            print(f"  Unexecuted Orders Remaining (in MELO): {unexecuted_count}")
            print(f"  Unexecuted Quantity Remaining (in MELO): {unexecuted_qty}")
            
            if placed_count != (matched_order_count + withdrawn_count + unexecuted_count):
                print(f"ACCOUNTING CHECK FAILED: Placed ({placed_count}) != Matched ({matched_order_count}) + Withdrawn ({withdrawn_count}) + Unexecuted ({unexecuted_count})")
            else:
                print(f"ACCOUNTING CHECK PASSED: Placed ({placed_count}) == Matched ({matched_order_count}) + Withdrawn ({withdrawn_count}) + Unexecuted ({unexecuted_count})")
        else:
            print("Accounting Warning: MeloMarket or its OrderBook missing required counting methods ---")
    else:
        print("Accounting Warning: Could not access simulator.last_sim_instance.meloMarket object ---") # Updated warning

    cda_payoffs = [payoff for _, strat, payoff in results[0] if strat == "MELO_100_0"]
    melo_payoffs = [payoff for _, strat, payoff in results[0] if strat == "MELO_0_100"]
    
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
        "sim_time": [100],
        "lam": [0.1],
        "lam_melo": [0.2],
        "mean": [1e6],
        "r": [0.01],
        "q_max": [15],
        "holding_period": [5]
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


 # calculate matched quantity specifically from the MeloMarket's matched_orders
            # Note: This assumes matched_orders only contains MELO matches. If CDA matches need 
            # to be included, we'd need to access simulator.last_sim_instance.market.matched_orders too.
            # Sum the quantity from the nested Order object within MatchedOrder