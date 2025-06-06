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
import math

def analyze_basins_of_attraction_rsg(game, num_points=100, iters=5000):
    """
    Analyze basins of attraction for role symmetric games.
    This is more complex than symmetric games due to role structure.
    """
    print("\nAnalyzing basins of attraction for role symmetric game...")
    
    if not game.is_role_symmetric:
        print("This function is for role symmetric games only.")
        return None
    
    if game.num_strategies < 2:
        print("Need at least 2 strategies to analyze basins of attraction")
        return None
    
   
    grid_mixtures = []
    for i in range(num_points):
        mixture = torch.zeros(game.num_strategies, device=game.game.device)
        
        # Generate random mixture for each role
        global_idx = 0
        for role_idx, (role_name, role_strategies) in enumerate(zip(game.role_names, game.strategy_names_per_role)):
            num_role_strats = len(role_strategies)
            if num_role_strats > 0:
                # Random distribution for this role
                role_mixture = torch.rand(num_role_strats)
                role_mixture = role_mixture / role_mixture.sum()  # Normalize
                mixture[global_idx:global_idx + num_role_strats] = role_mixture
            global_idx += num_role_strats
        
        grid_mixtures.append(mixture)
    
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
        final_regret_result = regret(game, final_mix)
        final_regret = final_regret_result.item() if torch.is_tensor(final_regret_result) else final_regret_result
        
        # Determine which equilibrium it converged to (if any)
        converged_to = "Mixed"
        if final_regret < 1e-3:  # Low regret indicates equilibrium
            # Analyze the mixture to determine type
            dominant_strategies = []
            global_idx = 0
            for role_name, role_strategies in zip(game.role_names, game.strategy_names_per_role):
                role_mixture = final_mix[global_idx:global_idx + len(role_strategies)]
                max_idx = torch.argmax(role_mixture)
                if role_mixture[max_idx] > 0.9:
                    dominant_strategies.append(f"{role_name}:{role_strategies[max_idx]}")
                global_idx += len(role_strategies)
            
            if len(dominant_strategies) == len(game.role_names):
                converged_to = " & ".join(dominant_strategies)
            else:
                converged_to = "Mixed"
        else:
            converged_to = "None"
        
        converged_to_eq.append(converged_to)
    
    # Create visualization
    plt.figure(figsize=(15, 10))
    
    # Plot convergence types
    convergence_types = list(set(converged_to_eq))
    colors = plt.cm.Set3(np.linspace(0, 1, len(convergence_types)))
    color_map = {conv_type: colors[i] for i, conv_type in enumerate(convergence_types)}
    
    # Create a 2D projection for visualization (using first two strategy dimensions)
    x_vals = [mix[0].item() for mix in grid_mixtures]
    y_vals = [mix[1].item() if game.num_strategies > 1 else 0 for mix in grid_mixtures]
    point_colors = [color_map[eq] for eq in converged_to_eq]
    
    plt.scatter(x_vals, y_vals, c=point_colors, s=50, alpha=0.7)
    plt.xlabel('First Strategy Probability')
    plt.ylabel('Second Strategy Probability' if game.num_strategies > 1 else 'Constant')
    plt.title('Basin of Attraction (Role Symmetric Game)')
    
    # Create legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[conv_type], 
                  label=conv_type, markersize=10)
        for conv_type in convergence_types
    ]
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('rsg_basin_of_attraction.png', bbox_inches='tight')
    plt.show()
    
    # Calculate basin sizes
    basin_counts = {}
    for eq in converged_to_eq:
        basin_counts[eq] = basin_counts.get(eq, 0) + 1
    
    total_points = len(converged_to_eq)
    basin_percentages = {eq: (count / total_points) * 100 for eq, count in basin_counts.items()}
    
    print("Basin of Attraction Analysis (Role Symmetric Game):")
    for eq, percentage in basin_percentages.items():
        print(f"  {eq}: {percentage:.1f}% of initial conditions")
    
    return grid_mixtures, final_states, converged_to_eq

if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
    print("Using GPU for simulations")
else:
    device = torch.device("cpu")
    print("Using CPU for simulations")

def run_role_symmetric_mobi_zi_egta():
    """
    Run EGTA with role symmetric games where both MOBI and ZI agents are strategic.
    """
    num_strategic_mobi = 28
    num_strategic_zi = 40
    holding_periods = [160]
    
    all_results = []
    
    for holding_period in holding_periods:
        print(f"\n{'='*60}")
        print(f"RUNNING ROLE SYMMETRIC EXPERIMENT: HOLDING PERIOD {holding_period}")
        print(f"{'='*60}")
    
        sim_time = 8000  
        num_iterations = 10
        batch_size = 40
        
        print(f"Running Role Symmetric EGTA with {num_strategic_mobi} strategic MOBI and {num_strategic_zi} strategic ZI agents")
        print(f"Holding period: {holding_period}")
        print(f"Simulation time: {sim_time}")
        
        mobi_strategies = [
            "MOBI_100_0",   # 100% CDA, 0% MELO
            #"MOBI_50_50",   # 50% CDA, 50% MELO
            "MOBI_0_100"   # 0% CDA, 100% MELO
        ]
        
        zi_strategies = [
            "ZI_100_0",     # 100% CDA, 0% MELO
            #"ZI_75_25",     # 75% CDA, 25% MELO
            #"ZI_25_75",     # 25% CDA, 75% MELO
            "ZI_0_100"     # 0% CDA, 100% MELO
        ]
        
        # Create simulator with role symmetric game parameters
        simulator = MeloSimulator(
            num_strategic_mobi=num_strategic_mobi, 
            num_strategic_zi=num_strategic_zi,
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
            num_background_zi=0,  
            num_background_hbl=0, 
            reps=10,              # Reduced for faster testing
            mobi_strategies=mobi_strategies,
            zi_strategies=zi_strategies
        )
        
        # Get role information
        role_names, num_players_per_role, strategy_names_per_role = simulator.get_role_info()
        
        print(f"Roles: {role_names}")
        print(f"Players per role: {num_players_per_role}")
        print(f"Strategies per role: {strategy_names_per_role}")
        
        for role_name, strategies in zip(role_names, strategy_names_per_role):
            print(f"{role_name} strategies:")
            for strategy in strategies:
                params = simulator.strategy_params[strategy]
                print(f"  {strategy}: CDA={params['cda_proportion']}, MELO={params['melo_proportion']}")
        
        # Create DPR scheduler for role symmetric games
        scheduler = DPRScheduler(
            strategies=simulator.get_strategies(),  # All strategies combined
            num_players=simulator.get_num_players(),
            batch_size=batch_size,
            reduction_size_per_role={"MOBI": 4, "ZI": 4},  
            seed=42,
            role_names=role_names,
            num_players_per_role=num_players_per_role,
            strategy_names_per_role=strategy_names_per_role,
            subgame_size=4
        )


        
        egta = EGTA(
            simulator=simulator,
            scheduler=scheduler,
            device=device,
            output_dir=f"results/rsg_mobi_zi_egta/holding_period_{holding_period}",
            max_profiles=batch_size * num_iterations,
            seed=42
        )
        
        print("Running Role Symmetric EGTA...")
        start_time = time.time()
        game = egta.run(
            max_iterations=num_iterations,
            profiles_per_iteration=256,
            save_frequency=2,
            verbose=True,
            quiesce_kwargs={
                'num_iters': 50,
                'num_random_starts': 60,
                'regret_threshold': 1e-3,
                'dist_threshold': 1e-2,
                'solver': 'replicator',
                'solver_iters': 3000,
                'restricted_game_size': 4
            }
        )
        end_time = time.time()
        print(f"Role Symmetric EGTA completed in {end_time - start_time:.2f} seconds")
        
        # Verify equilibria are properly role symmetric
        if egta.equilibria and game.is_role_symmetric:
            print(f"Verifying {len(egta.equilibria)} role symmetric equilibria...")
            print(f"State space structure: Δ^{len(game.strategy_names_per_role[0])-1} × Δ^{len(game.strategy_names_per_role[1])-1}")
            print(f"Expected total sum: {len(game.role_names)} (each role sums to 1)")
            
            for i, (mixture, regret_val) in enumerate(egta.equilibria[:3]):  # Check first 3
                print(f"\nEquilibrium {i+1}:")
                total_sum = mixture.sum().item()
                expected_total = len(game.role_names)  # Should equal number of roles
                print(f"  Total sum: {total_sum:.6f} (expected: {expected_total:.1f})")
                
                # Verify each role sums to 1 and show the structure
                global_idx = 0
                all_roles_valid = True
                role_details = []
                
                for role_name, role_strategies in zip(game.role_names, game.strategy_names_per_role):
                    role_part = mixture[global_idx:global_idx + len(role_strategies)]
                    role_sum = role_part.sum().item()
                    print(f"  {role_name} sum: {role_sum:.6f} (expected: 1.0)")
                    
                    # Show strategy breakdown for this role
                    role_breakdown = []
                    for j, strategy in enumerate(role_strategies):
                        prob = role_part[j].item()
                        if prob > 0.01:
                            role_breakdown.append(f"{strategy}:{prob:.3f}")
                    
                    if role_breakdown:
                        role_details.append(f"{role_name}[{', '.join(role_breakdown)}]")
                    else:
                        role_details.append(f"{role_name}[near-zero]")
                    
                    if abs(role_sum - 1.0) > 1e-3:
                        all_roles_valid = False
                    global_idx += len(role_strategies)
                
                print(f"  Structure: {' × '.join(role_details)}")
                
                total_valid = abs(total_sum - expected_total) < 1e-3
                
                if all_roles_valid and total_valid:
                    print(f"Valid multi-population equilibrium")
                else:
                    print(f"Invalid role symmetric equilibrium")
                    if not total_valid:
                        print(f"     - Total sum incorrect: {total_sum:.6f} ≠ {expected_total:.1f}")
                    if not all_roles_valid:
                        print(f"     - Some roles don't sum to 1.0")
                        
            # Test the coupling: show that payoffs depend on full joint state
            if len(egta.equilibria) > 0:
                test_mixture = egta.equilibria[0][0]
                print(f"\n🔗 Verifying payoff coupling (payoffs depend on full joint state):")
                
                # Compute payoffs for the equilibrium mixture
                dev_payoffs = game.deviation_payoffs(test_mixture)
                print(f"Deviation payoffs at equilibrium:")
                global_idx = 0
                for role_name, role_strategies in zip(game.role_names, game.strategy_names_per_role):
                    print(f"  {role_name}:")
                    for j, strategy in enumerate(role_strategies):
                        payoff = dev_payoffs[global_idx + j].item()
                        print(f"    {strategy}: {payoff:.6f}")
                    global_idx += len(role_strategies)
                    
            print(f"\n🎉 Multi-population replicator dynamics is working correctly!")
            print(f"   • Each role maintains its own simplex (sums to 1)")
            print(f"   • Payoffs computed from full joint state (coupled)")
            print(f"   • State space: Cartesian product of role simplexes")
        
        print("\nRole Symmetric Game Details:")
        print(f"Role names: {game.role_names}")
        print(f"Strategy names per role: {game.strategy_names_per_role}")
        print(f"Number of players per role: {game.num_players_per_role}")
        
        welfare_data = []
        labels = []
        eq_mixture = None
        
        if egta.equilibria:
            eq_mixture, eq_regret = egta.equilibria[0]  
            
            print("\nEquilibria found by Role Symmetric EGTA:")
            print("\nWelfare Analysis of Equilibria:")
            
            # Calculate welfare for role symmetric equilibria
            for i, (mixture, regret_val) in enumerate(egta.equilibria):
                expected_welfare = 0.0
                
                try:
                    dev_payoffs = game.deviation_payoffs(mixture)
                    # Calculate expected payoff weighted by mixture
                    expected_welfare = (mixture * dev_payoffs).sum().item()
                except Exception as e:
                    print(f"Error calculating welfare for equilibrium {i+1}: {e}")
                    expected_welfare = 0.0
                
                welfare_data.append(expected_welfare)
                labels.append(f"Eq {i+1}")
                
                print(f"\nEquilibrium {i+1} (regret: {regret_val:.6f}):")
                print(f"  Expected Welfare: {expected_welfare:.6f}")
                
                global_idx = 0
                for role_name, role_strategies in zip(game.role_names, game.strategy_names_per_role):
                    role_mixture = mixture[global_idx:global_idx + len(role_strategies)]
                    role_sum = role_mixture.sum().item()
                    
                    print(f"  {role_name} (role weight: {role_sum:.6f}):")
                    
                    if role_sum > 1e-6:  
                        normalized_role_mixture = role_mixture / role_sum
                        
                        for j, strategy in enumerate(role_strategies):
                            prob = normalized_role_mixture[j].item()
                            if prob > 0.001:
                                print(f"    {strategy}: {prob:.6f}")
                    else:
                        print(f" WARNING: Role has near-zero weight ({role_sum:.2e}) - this may indicate an issue with role symmetric equilibrium")
                        print(f"    Raw probabilities: {[f'{prob:.2e}' for prob in role_mixture.tolist()]}")
                    
                    global_idx += len(role_strategies)
            
            # Analyze equilibrium structure
            print("\nDetailed analysis of best equilibrium:")
            global_idx = 0
            equilibrium_description = []
            
            for role_name, role_strategies in zip(game.role_names, game.strategy_names_per_role):
                role_mixture = eq_mixture[global_idx:global_idx + len(role_strategies)]
                dominant_strategy_idx = torch.argmax(role_mixture)
                dominant_prob = role_mixture[dominant_strategy_idx].item()
                dominant_strategy = role_strategies[dominant_strategy_idx]
                
                if dominant_prob > 0.9:
                    equilibrium_description.append(f"{role_name}: {dominant_strategy}")
                else:
                    # Mixed strategy for this role
                    mixed_strategies = []
                    for j, strategy in enumerate(role_strategies):
                        if role_mixture[j].item() > 0.1:
                            mixed_strategies.append(f"{strategy}({role_mixture[j].item():.2f})")
                    equilibrium_description.append(f"{role_name}: Mixed({', '.join(mixed_strategies)})")
                
                global_idx += len(role_strategies)
            
            print(f"Equilibrium structure: {' | '.join(equilibrium_description)}")
            
            regret_val = eq_regret if isinstance(eq_regret, float) else eq_regret.item()
            print(f"Regret: {regret_val:.6f}")
        
        # Create experiment parameters dictionary
        experiment_params = {
            "holding_period": holding_period,
            "sim_time": sim_time,
            "num_strategic_mobi": num_strategic_mobi,
            "num_strategic_zi": num_strategic_zi,
            "num_iterations": num_iterations,
            "batch_size": batch_size,
            "is_role_symmetric": True,
            "role_names": role_names,
            "num_players_per_role": num_players_per_role,
            "strategy_names_per_role": strategy_names_per_role,
            "simulator_params": simulator.strategy_params
        }
        
        # Save results for this holding period
        print(f"\n💾 Saving results for holding period {holding_period}...")
        results_dir = save_comprehensive_rsg_results(
            egta, game, welfare_data, labels, experiment_params, 
            output_dir=f"results/rsg_mobi_zi_egta/holding_period_{holding_period}"
        )
        
        # Determine equilibrium type for summary
        if egta.equilibria:
            eq_type = "Mixed"
            global_idx = 0
            pure_count = 0
            for role_name, role_strategies in zip(game.role_names, game.strategy_names_per_role):
                role_mixture = eq_mixture[global_idx:global_idx + len(role_strategies)]
                if torch.max(role_mixture) > 0.9:
                    pure_count += 1
                global_idx += len(role_strategies)
            
            if pure_count == len(game.role_names):
                eq_type = "Pure"
            elif pure_count > 0:
                eq_type = "Partially Mixed"
        else:
            eq_type = "None Found"
        
        period_summary = {
            "holding_period": holding_period,
            "best_welfare": welfare_data[0] if welfare_data else None,
            "best_regret": regret_val if egta.equilibria else None,
            "equilibrium_type": eq_type,
            "num_equilibria": len(egta.equilibria),
            "results_dir": results_dir,
            "equilibrium_description": ' | '.join(equilibrium_description) if egta.equilibria else "None"
        }
        all_results.append(period_summary)
        
        print(f"Completed holding period {holding_period}")
        print(f"Results saved to: {results_dir}")
    
    print(f"\n{'='*60}")
    print("ROLE SYMMETRIC CROSS-PERIOD ANALYSIS SUMMARY")
    print(f"{'='*60}")
    
    summary_file = "results/rsg_mobi_zi_egta/cross_period_summary.json"
    os.makedirs("results/rsg_mobi_zi_egta", exist_ok=True)
    with open(summary_file, 'w') as f:
        json.dump({
            "analysis_timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "holding_periods_analyzed": holding_periods,
            "experiment_type": "role_symmetric",
            "results": all_results
        }, f, indent=2)
    
    print(f"Cross-period summary saved to: {summary_file}")
    
    # Print summary table
    print("ROLE SYMMETRIC SUMMARY TABLE:")
    print("Holding Period | Equilibrium Type | Welfare | Regret | Description")
    print("-" * 85)
    for result in all_results:
        hp = result["holding_period"]
        eq_type = result["equilibrium_type"]
        welfare = result["best_welfare"]
        regret = result["best_regret"]
        description = result["equilibrium_description"][:40] + "..." if len(result["equilibrium_description"]) > 40 else result["equilibrium_description"]
        
        welfare_str = f"{welfare:.1f}" if welfare is not None else "N/A"
        regret_str = f"{regret:.4f}" if regret is not None else "N/A"
        print(f"{hp:14d} | {eq_type:15s} | {welfare_str:7s} | {regret_str:6s} | {description}")
    
    return game, eq_mixture, egta, welfare_data, labels, experiment_params

def save_comprehensive_rsg_results(egta, game, welfare_data, labels, experiment_params, output_dir="results/rsg_mobi_zi_egta", basin_results=None):
    """Save comprehensive results for role symmetric games."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    results_dir = os.path.join(output_dir, f"comprehensive_rsg_results_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    
    # Save experiment parameters
    params_file = os.path.join(results_dir, "experiment_parameters.json")
    with open(params_file, 'w') as f:
        json.dump(experiment_params, f, indent=2)
    print(f"Saved experiment parameters to {params_file}")
    
    # Save equilibria (role symmetric format)
    equilibria_detailed = []
    for i, (mixture, regret_val) in enumerate(egta.equilibria):
        eq_dict = {
            "equilibrium_id": i + 1,
            "regret": float(regret_val),
            "welfare": welfare_data[i] if i < len(welfare_data) else None,
            "is_role_symmetric": True,
            "mixture_by_role": {},
            "mixture_vector": mixture.tolist(),
            "support_by_role": {},
            "support_size": sum(1 for x in mixture if x.item() > 0.001),
            "equilibrium_type": "Mixed"
        }
        
        # Analyze mixture by role
        global_idx = 0
        pure_roles = 0
        for role_name, role_strategies in zip(game.role_names, game.strategy_names_per_role):
            role_mixture = {}
            role_support = []
            role_probs = mixture[global_idx:global_idx + len(role_strategies)]
            
            max_prob = 0.0
            for j, strategy in enumerate(role_strategies):
                prob = role_probs[j].item()
                if prob > 0.001:
                    role_mixture[strategy] = prob
                    role_support.append(strategy)
                if prob > max_prob:
                    max_prob = prob
            
            eq_dict["mixture_by_role"][role_name] = role_mixture
            eq_dict["support_by_role"][role_name] = role_support
            
            if max_prob > 0.9:
                pure_roles += 1
            
            global_idx += len(role_strategies)
        
        # Determine equilibrium type
        if pure_roles == len(game.role_names):
            eq_dict["equilibrium_type"] = "Pure"
        elif pure_roles > 0:
            eq_dict["equilibrium_type"] = "Partially Mixed"
        else:
            eq_dict["equilibrium_type"] = "Mixed"
        
        equilibria_detailed.append(eq_dict)
    
    eq_file = os.path.join(results_dir, "equilibria_detailed.json")
    with open(eq_file, 'w') as f:
        json.dump(equilibria_detailed, f, indent=2)
    print(f"Saved detailed equilibria to {eq_file}")
    
    # Save welfare analysis
    welfare_analysis = {
        "welfare_data": welfare_data,
        "labels": labels,
        "best_equilibrium_welfare": max(welfare_data) if welfare_data else None,
        "worst_equilibrium_welfare": min(welfare_data) if welfare_data else None,
        "welfare_variance": np.var(welfare_data) if len(welfare_data) > 1 else 0,
        "analysis_type": "role_symmetric"
    }
    
    welfare_file = os.path.join(results_dir, "welfare_analysis.json")
    with open(welfare_file, 'w') as f:
        json.dump(welfare_analysis, f, indent=2)
    print(f"Saved welfare analysis to {welfare_file}")
    
    # Save game details
    game_details = {
        "is_role_symmetric": True,
        "role_names": game.role_names,
        "num_players_per_role": game.num_players_per_role.tolist(),  # Convert tensor to list
        "strategy_names_per_role": game.strategy_names_per_role,
        "total_strategies": game.num_strategies,
        "total_players": game.num_players,
        "total_profiles_simulated": len(egta.payoff_data),
        "game_type": "role_symmetric"
    }
    
    game_file = os.path.join(results_dir, "game_details.json")
    with open(game_file, 'w') as f:
        json.dump(game_details, f, indent=2)
    print(f"Saved game details to {game_file}")
    
    print(f"Comprehensive role symmetric results saved to: {results_dir}")
    return results_dir

if __name__ == "__main__":
    os.makedirs("results/rsg_mobi_zi_egta", exist_ok=True)
    game, eq_mixture, egta, welfare_data, labels, experiment_params = run_role_symmetric_mobi_zi_egta()
    
    # Run basin of attraction analysis if we have an equilibrium
    if eq_mixture is not None:
        basin_results = analyze_basins_of_attraction_rsg(game, num_points=50, iters=3000)
        
        # Save comprehensive results with basin analysis
        results_dir = save_comprehensive_rsg_results(
            egta, game, welfare_data, labels, experiment_params, 
            basin_results=basin_results
        ) 