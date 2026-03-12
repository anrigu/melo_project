import torch
import numpy as np
import os
import sys
import time
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
import seaborn as sns
import json
import pandas as pd
from datetime import datetime
from matplotlib.lines import Line2D


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from marketsim.egta.simulators.melo_wrapper import MeloSimulator
from marketsim.egta.core.game import Game
from marketsim.egta.egta import EGTA
from marketsim.egta.schedulers.dpr import DPRScheduler
from marketsim.egta.schedulers.random import RandomScheduler
from marketsim.egta.solvers.equilibria import replicator_dynamics, regret

def analyze_basins_of_attraction(game, num_points=100, iters=5000):

    print("\nAnalyzing basins of attraction...")
    
    if game.num_strategies < 2:
        print("Need at least 2 strategies to analyze basins of attraction")
        return None
    
    if game.num_strategies == 2:
        grid_mixtures = []
        for i in range(num_points + 1):
            p = i / num_points
            mix = torch.zeros(game.num_strategies, device=game.game.device)
            mix[0] = p  
            mix[1] = 1 - p  
            grid_mixtures.append(mix)
        
        # Run replicator dynamics from each starting point
        final_states = []
        converged_to_eq = []
        traces = []
        
        for mix in grid_mixtures:
            final_mix, trace = replicator_dynamics(
                game, 
                mix.clone(), 
                iters=iters, 
                converge_threshold=1e-3,
                return_trace=True
            )
            final_states.append(final_mix)
            traces.append(trace)
            
            # Calculate regret for the final state
            final_regret = regret(game, final_mix).item()
            
            # Determine which equilibrium it converged to (if any)
            converged_to = "None"
            if final_regret < 1e-3:  # Low regret indicates equilibrium
                if final_mix[0] > 0.9:  # Converged to all MELO
                    converged_to = "All MELO"
                elif final_mix[1] > 0.9:  # Converged to all CDA
                    converged_to = "All CDA"
                else:
                    converged_to = "Mixed"
            
            converged_to_eq.append(converged_to)
        
        plt.figure(figsize=(12, 6))
        x_vals = [mix[0].item() for mix in grid_mixtures]
        
        colors = {"All MELO": "blue", "All CDA": "red", "Mixed": "purple", "None": "gray"}
        point_colors = [colors[eq] for eq in converged_to_eq]
        
        plt.scatter(x_vals, [1] * len(x_vals), c=point_colors, s=50)
        plt.yticks([])
        plt.xlabel('Initial MELO Probability')
        plt.title('Basin of Attraction (Initial → Final State)')
        
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=colors["All MELO"], label='All MELO', markersize=10),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=colors["All CDA"], label='All CDA', markersize=10),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=colors["Mixed"], label='Mixed', markersize=10),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=colors["None"], label='None', markersize=10)
        ]
        plt.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=4)
        plt.tight_layout()
        plt.savefig('basin_of_attraction.png')
        plt.show()
        
        # Plot 2: Sample trajectories in 2D
        plt.figure(figsize=(10, 6))
        num_traces_to_plot = min(10, len(traces))
        step = max(1, len(traces) // num_traces_to_plot)
        for i in range(0, len(traces), step):
            if i < len(traces):  # Safety check
                trace = traces[i]
                trace_vals = [t[0].item() for t in trace]
                plt.plot(range(len(trace_vals)), trace_vals, 
                         label=f"Initial MELO: {x_vals[i]:.2f}", alpha=0.7)
        
        plt.xlabel('Iteration')
        plt.ylabel('MELO Probability')
        plt.title('Replicator Dynamics Trajectories (Sample)')
        plt.legend()
        plt.tight_layout()
        plt.savefig('trajectory_samples.png')
        plt.show()

        max_trace_len = max(len(trace) for trace in traces)
        
        trajectory_data = np.zeros((len(traces), max_trace_len))
        
        # Fill in trajectory data - probability of MELO
        for i, trace in enumerate(traces):
            trace_vals = [t[0].item() for t in trace]
            trajectory_data[i, :len(trace_vals)] = trace_vals    
            
            # Fill any remaining positions with the final value     
            if len(trace_vals) < max_trace_len:
                trajectory_data[i, len(trace_vals):] = trace_vals[-1]
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(trajectory_data, cmap="viridis", cbar_kws={'label': 'MELO Probability'})
        plt.xlabel('Iteration')
        plt.ylabel('Initial MELO Probability')
        plt.title('Replicator Dynamics Trajectories')
        plt.tight_layout()
        plt.savefig('basin_heatmap.png')
        plt.show()
        
        # Calculate the size of each basin
        basin_counts = {}
        for eq in converged_to_eq:
            if eq not in basin_counts:
                basin_counts[eq] = 0
            basin_counts[eq] += 1
        
        # Convert to percentages
        total_points = len(converged_to_eq)
        basin_percentages = {eq: (count / total_points) * 100 for eq, count in basin_counts.items()}
        
        print("Basin of Attraction Analysis:")
        for eq, percentage in basin_percentages.items():
            print(f"  {eq}: {percentage:.1f}% of initial conditions")
        
        return grid_mixtures, final_states, converged_to_eq
    
    else:
        print("Basin visualization for >2 strategies not implemented yet")
        return None

if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
    print("Using GPU for simulations")
else:
    device = torch.device("cpu")
    print("Using CPU for simulations")

def run_mobi_zi_egta():
    num_mobi_agents = 10
    num_zi_agents = 30
    holding_periods = [160, 180]
    
    # Store results for all holding periods
    all_results = []
    
    for holding_period in holding_periods:
        print(f"\n{'='*60}")
        print(f"RUNNING EXPERIMENT: HOLDING PERIOD {holding_period}")
        print(f"{'='*60}")
    
        sim_time = 10000  # 1×10^4 timesteps as specified
        num_iterations = 10
        num_strategies = 2 
        
        # Set reasonable limits instead of calculating all possible profiles
        # DPR will intelligently sample the most important profiles
        max_profiles = 50  # Reasonable limit for efficient exploration
        profiles_per_iteration = 10  # Profiles to simulate per iteration
        
        print(f"Max profiles to simulate: {max_profiles}")
        print(f"Profiles per iteration: {profiles_per_iteration}")
        
        print(f"Running EGTA with {num_mobi_agents} MOBI and {num_zi_agents} ZI agents")
        print(f"Holding period: {holding_period}")
        print(f"Simulation time: {sim_time}")
        
        # Create simulator with paper-specified parameters
        simulator = MeloSimulator(
            num_strategic_mobi=num_mobi_agents, 
            sim_time=sim_time,
            lam=0.006,           
            mean=1000000.0,      
            r=0.001,             
            shock_var=100,       
            q_max=10,            
            pv_var=5000000,      
            shade=[10, 30],      
            holding_period=holding_period,  
            lam_melo=0.001,      
            num_background_zi=num_zi_agents,  
            #num_hbl=0,
            mobi_strategies= [
            "MOBI_100_0",  
            #"MOBI_50_50", 
            "MOBI_0_100"   
            ],
            reps=10,           
            force_symmetric=True  
        )
        
        strategies = simulator.get_strategies()
        print(f"Available strategies: {strategies}")
        
        for strategy, params in simulator.strategy_params.items():
            print(f"{strategy}: CDA={params['cda_proportion']}, MELO={params['melo_proportion']}")
        
        # Use DPRScheduler for both symmetric and role symmetric games
        scheduler = DPRScheduler(
            strategies=strategies,
            num_players=num_mobi_agents,
            batch_size=max_profiles,
            reduction_size=4, # num_mobi - 1 / reduction_size - 1 needs to be an integer
            seed=42
        )
        
        egta = EGTA(
            simulator=simulator,
            scheduler=scheduler,
            device=device,
            output_dir=f"results/mobi_zi_egta/holding_period_{holding_period}",
            max_profiles=max_profiles,
            seed=42
        )
        
        print("Running EGTA...")
        start_time = time.time()
        game = egta.run(
            max_iterations=num_iterations,
            profiles_per_iteration=profiles_per_iteration,
            save_frequency=1,
            verbose=True,
            quiesce_kwargs={
                'num_iters': 50,
                'num_random_starts': 20,
                'regret_threshold': 1e-3,
                'dist_threshold': 1e-2,
                'solver': 'replicator',
                'solver_iters': 5000,
                'restricted_game_size': 4
            }
        )
        end_time = time.time()
        print(f"EGTA completed in {end_time - start_time:.2f} seconds")
        
        print("\nGame Details:")
        print(f"Strategy names: {game.strategy_names}")
        
        # Get the payoff matrix
        payoff_matrix = game.game.payoff_table.cpu().numpy()
        print("\nPayoff Matrix:")
        for i, strategy in enumerate(game.strategy_names):
            print(f"{strategy}: {payoff_matrix[i]}")
        
        welfare_data = []
        labels = []
        eq_mixture = None
        
        if egta.equilibria:
            eq_mixture, eq_regret = egta.equilibria[0]  
            
            print("\nEquilibria found by EGTA:")
            print("\nWelfare Analysis of Equilibria:")
            
            strategy_to_idx = {name: i for i, name in enumerate(game.strategy_names)}
            
            profile_payoffs = {}
            for profile_data in egta.payoff_data:
                profile_str = str([(player_id, strat) for player_id, strat, _ in profile_data])
                if profile_str not in profile_payoffs:
                    profile_payoffs[profile_str] = []
                profile_payoffs[profile_str].extend([payoff for _, _, payoff in profile_data])
            
            for i, (mixture, regret_val) in enumerate(egta.equilibria):
                expected_welfare = 0.0
                weight_sum = 0.0
                
                for profile_str, payoffs in profile_payoffs.items():
                    profile_list = eval(profile_str)
                    strategy_counts = {}
                    for _, strat in profile_list:
                        if strat not in strategy_counts:
                            strategy_counts[strat] = 0
                        strategy_counts[strat] += 1
                    
                    profile_prob = 1.0
                    for strat, count in strategy_counts.items():
                        if strat in strategy_to_idx:
                            strat_idx = strategy_to_idx[strat]
                            profile_prob *= mixture[strat_idx].item() ** count
                    
                    avg_payoff = sum(payoffs) / len(payoffs) if payoffs else 0
                    
                    expected_welfare += profile_prob * avg_payoff
                    weight_sum += profile_prob
                
                if weight_sum > 0:
                    expected_welfare /= weight_sum
                
                welfare_data.append(expected_welfare)
                labels.append(f"Eq {i+1}")
                
                print(f"\nEquilibrium {i+1} (regret: {regret_val:.6f}):")
                print(f"  Expected MOBI Agent Welfare: {expected_welfare:.6f}")
                for j, strategy in enumerate(game.strategy_names):
                    print(f"  {strategy}: {mixture[j].item():.6f}")
            
            print("\nDetailed analysis of best equilibrium:")
            for i, strategy in enumerate(game.strategy_names):
                print(f"{strategy}: {eq_mixture[i].item():.6f}")
            
            regret_val = eq_regret if isinstance(eq_regret, float) else eq_regret.item()
            print(f"Regret: {regret_val:.6f}")
            
            # Determine strategy dominance
            cda_idx = game.strategy_names.index("MOBI_100_0") if "MOBI_100_0" in game.strategy_names else 0
            cda_probability = eq_mixture[cda_idx].item()
            
            if cda_probability > 0.99:
                dominant_strategy = "CDA"
                print("The equilibrium is predominantly CDA")
            elif cda_probability < 0.01:
                dominant_strategy = "MELO"
                print("The equilibrium is predominantly MELO")
            else:
                dominant_strategy = "Mixed"
                print("⚖️ The equilibrium is mixed strategy")
        
        # Create experiment parameters dictionary
        experiment_params = {
            "holding_period": holding_period,
            "sim_time": sim_time,
            "num_mobi_agents": num_mobi_agents,
            "num_zi_agents": num_zi_agents,
            "num_iterations": num_iterations,
            "num_strategies": num_strategies,
            "max_profiles": max_profiles,
            "profiles_per_iteration": profiles_per_iteration,
            "simulator_params": simulator.strategy_params
        }
        
        # Save results for this holding period
        print(f"\n💾 Saving results for holding period {holding_period}...")
        results_dir = save_comprehensive_results(
            egta, game, welfare_data, labels, experiment_params, 
            output_dir=f"results/mobi_zi_egta/holding_period_{holding_period}"
        )
        
        period_summary = {
            "holding_period": holding_period,
            "best_welfare": welfare_data[0] if welfare_data else None,
            "best_regret": regret_val if egta.equilibria else None,
            "dominant_strategy": dominant_strategy if egta.equilibria else "None",
            "cda_probability": cda_probability if egta.equilibria else None,
            "num_equilibria": len(egta.equilibria),
            "results_dir": results_dir
        }
        all_results.append(period_summary)
        
        print(f"Completed holding period {holding_period}")
        print(f"Results saved to: {results_dir}")
    
    print(f"\n{'='*60}")
    print("CROSS-PERIOD ANALYSIS SUMMARY")
    print(f"{'='*60}")
    
    summary_file = "results/mobi_zi_egta/cross_period_summary.json"
    os.makedirs("results/mobi_zi_egta", exist_ok=True)
    with open(summary_file, 'w') as f:
        json.dump({
            "analysis_timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "holding_periods_analyzed": holding_periods,
            "results": all_results
        }, f, indent=2)
    
    print(f"Cross-period summary saved to: {summary_file}")
    
    # Print summary table
    print("SUMMARY TABLE:")
    print("Holding Period | Dominant Strategy | CDA Probability | Welfare")
    print("-" * 65)
    for result in all_results:
        hp = result["holding_period"]
        strategy = result["dominant_strategy"]
        cda_prob = result["cda_probability"]
        welfare = result["best_welfare"]
        cda_str = f"{cda_prob:.3f}" if cda_prob is not None else "N/A"
        welfare_str = f"{welfare:.1f}" if welfare is not None else "N/A"
        print(f"{hp:14d} | {strategy:16s} | {cda_str:11s} | {welfare_str}")
    
    return game, eq_mixture, egta, welfare_data, labels, experiment_params

def convert_tensors_to_lists(obj):

    
    if isinstance(obj, torch.Tensor):
        return obj.cpu().numpy().tolist()
    elif isinstance(obj, dict):
        return {key: convert_tensors_to_lists(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_tensors_to_lists(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_tensors_to_lists(item) for item in obj)
    else:
        return obj

def save_comprehensive_results(egta, game, welfare_data, labels, experiment_params, output_dir="results/mobi_zi_egta", basin_results=None):
   
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    results_dir = os.path.join(output_dir, f"comprehensive_results_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    
    params_file = os.path.join(results_dir, "experiment_parameters.json")
    with open(params_file, 'w') as f:
        json.dump(experiment_params, f, indent=2)
    print(f"Saved experiment parameters to {params_file}")
    
    equilibria_detailed = []
    for i, (mixture, regret_val) in enumerate(egta.equilibria):
        eq_dict = {
            "equilibrium_id": i + 1,
            "regret": float(regret_val),
            "welfare": welfare_data[i] if i < len(welfare_data) else None,
            "mixture_dict": {name: float(mixture[j].item()) for j, name in enumerate(game.strategy_names)},
            "mixture_vector": mixture.tolist(),
            "support": [name for j, name in enumerate(game.strategy_names) if mixture[j].item() > 0.001],
            "support_size": sum(1 for x in mixture if x.item() > 0.001),
            "is_pure_strategy": sum(1 for x in mixture if x.item() > 0.99) == 1,
            "dominant_strategy": game.strategy_names[torch.argmax(mixture).item()] if len(game.strategy_names) > 0 else None,
            "dominant_probability": float(torch.max(mixture).item())
        }
        equilibria_detailed.append(eq_dict)
    
    eq_file = os.path.join(results_dir, "equilibria_detailed.json")
    with open(eq_file, 'w') as f:
        json.dump(equilibria_detailed, f, indent=2)
    print(f"Saved detailed equilibria to {eq_file}")
    
    eq_df = pd.DataFrame(equilibria_detailed)
    eq_csv = os.path.join(results_dir, "equilibria_summary.csv")
    eq_df.to_csv(eq_csv, index=False)
    print(f"Saved equilibria CSV to {eq_csv}")
    
    welfare_analysis = {
        "welfare_data": welfare_data,
        "labels": labels,
        "best_equilibrium_welfare": max(welfare_data) if welfare_data else None,
        "worst_equilibrium_welfare": min(welfare_data) if welfare_data else None,
        "welfare_variance": np.var(welfare_data) if len(welfare_data) > 1 else 0,
        "welfare_comparison": [
            {
                "equilibrium": labels[i],
                "welfare": welfare_data[i],
                "welfare_rank": sorted(welfare_data, reverse=True).index(welfare_data[i]) + 1
            } for i in range(len(welfare_data))
        ]
    }
    
    welfare_file = os.path.join(results_dir, "welfare_analysis.json")
    with open(welfare_file, 'w') as f:
        json.dump(welfare_analysis, f, indent=2)
    print(f"Saved welfare analysis to {welfare_file}")
    
    # 5. Save game details and payoff matrix
    game_details = {
        "strategy_names": game.strategy_names,
        "num_strategies": game.num_strategies,
        "num_players": game.num_players,
        "payoff_matrix": game.game.payoff_table.cpu().numpy().tolist(),
        "game_metadata": game.metadata if hasattr(game, 'metadata') and game.metadata else {},
        "total_profiles_simulated": len(egta.payoff_data),
        "profiles_per_strategy_combination": {}
    }
    
    # Count profiles per strategy combination
    for profile_data in egta.payoff_data:
        strategies_in_profile = [strat for _, strat, _ in profile_data]
        strategy_counts = {}
        for strat in set(strategies_in_profile):
            strategy_counts[strat] = strategies_in_profile.count(strat)
        profile_key = str(sorted(strategy_counts.items()))
        if profile_key not in game_details["profiles_per_strategy_combination"]:
            game_details["profiles_per_strategy_combination"][profile_key] = 0
        game_details["profiles_per_strategy_combination"][profile_key] += 1
    
    game_file = os.path.join(results_dir, "game_details.json")
    with open(game_file, 'w') as f:
        json.dump(game_details, f, indent=2)
    print(f"Saved game details to {game_file}")
    
    # 6. Save raw payoff data for future analysis
    payoff_data_processed = []
    for i, profile_data in enumerate(egta.payoff_data):
        profile_dict = {
            "profile_id": i,
            "agents": [{"agent_id": agent_id, "strategy": strat, "payoff": float(payoff)} 
                      for agent_id, strat, payoff in profile_data],
            "average_payoff": np.mean([payoff for _, _, payoff in profile_data]),
            "strategy_distribution": {}
        }
        
        strategies_in_profile = [strat for _, strat, _ in profile_data]
        for strat in set(strategies_in_profile):
            profile_dict["strategy_distribution"][strat] = strategies_in_profile.count(strat)
        
        payoff_data_processed.append(profile_dict)
    
    payoff_file = os.path.join(results_dir, "raw_payoff_data.json")
    with open(payoff_file, 'w') as f:
        json.dump(payoff_data_processed, f, indent=2)
    print(f"Saved raw payoff data to {payoff_file}")
    
    if basin_results is not None:
        basin_file = os.path.join(results_dir, "basin_analysis.json")
        basin_analysis = {
            "basin_results": convert_tensors_to_lists(basin_results),
            "analysis_timestamp": timestamp,
            "analysis_type": "replicator_dynamics_basins"
        }
        with open(basin_file, 'w') as f:
            json.dump(basin_analysis, f, indent=2)
        print(f"Saved basin analysis to {basin_file}")
    
    # 8. Create summary report
    summary_report = {
        "experiment_summary": {
            "timestamp": timestamp,
            "holding_period": experiment_params.get("holding_period"),
            "num_equilibria_found": len(egta.equilibria),
            "best_equilibrium": equilibria_detailed[0] if equilibria_detailed else None,
            "predominant_strategy": None,
            "strategy_frequencies": {}
        },
        "key_findings": [],
        "files_generated": [
            "experiment_parameters.json",
            "equilibria_detailed.json", 
            "equilibria_summary.csv",
            "welfare_analysis.json",
            "game_details.json",
            "raw_payoff_data.json"
        ]
    }
    
    for strategy in game.strategy_names:
        total_freq = 0
        for eq_dict in equilibria_detailed:
            total_freq += eq_dict["mixture_dict"][strategy]
        summary_report["experiment_summary"]["strategy_frequencies"][strategy] = total_freq / len(equilibria_detailed) if equilibria_detailed else 0
    
    # Determine predominant strategy
    if summary_report["experiment_summary"]["strategy_frequencies"]:
        predominant = max(summary_report["experiment_summary"]["strategy_frequencies"].items(), key=lambda x: x[1])
        summary_report["experiment_summary"]["predominant_strategy"] = predominant[0]
    
    # Add key findings
    if equilibria_detailed:
        best_eq = equilibria_detailed[0]
        summary_report["key_findings"].append(f"Found {len(equilibria_detailed)} equilibria")
        summary_report["key_findings"].append(f"Best equilibrium has regret {best_eq['regret']:.6f}")
        summary_report["key_findings"].append(f"Dominant strategy: {best_eq['dominant_strategy']} ({best_eq['dominant_probability']:.1%})")
        
        if best_eq["is_pure_strategy"]:
            summary_report["key_findings"].append("Best equilibrium is a pure strategy")
        else:
            summary_report["key_findings"].append("Best equilibrium is a mixed strategy")
    
    if basin_results is not None:
        summary_report["files_generated"].append("basin_analysis.json")
    
    summary_file = os.path.join(results_dir, "experiment_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary_report, f, indent=2)
    print(f"Saved experiment summary to {summary_file}")
    
    print(f"Comprehensive results saved to: {results_dir}")
    print(f"📁 Generated {len(summary_report['files_generated'])} result files")
    
    return results_dir

if __name__ == "__main__":
    os.makedirs("results/mobi_zi_egta", exist_ok=True)
    game, eq_mixture, egta, welfare_data, labels, experiment_params = run_mobi_zi_egta() 
    results_dir = save_comprehensive_results(egta, game, welfare_data, labels, experiment_params, basin_results=analyze_basins_of_attraction(game, num_points=100, iters=5000)) 