import torch
import numpy as np
import os
import sys
import time
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
import seaborn as sns

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from marketsim.egta.simulators.melo_wrapper import MeloSimulator
from marketsim.egta.core.game import Game
from marketsim.egta.egta import EGTA
from marketsim.egta.schedulers.dpr import DPRScheduler
from marketsim.egta.schedulers.random import RandomScheduler
from marketsim.egta.solvers.equilibria import replicator_dynamics, regret
import math

def analyze_basins_of_attraction(game, num_points=100, iters=5000):
    """
    Analyze and visualize the basins of attraction for the game's equilibria.
    
    Args:
        game: The game to analyze
        num_points: Number of initial points to test
        iters: Number of iterations for replicator dynamics
        
    Returns:
        Tuple of (grid_mixtures, final_states, equilibria)
    """
    print("\nAnalyzing basins of attraction...")
    
    # We need at least 2 strategies to analyze basins of attraction
    if game.num_strategies < 2:
        print("Need at least 2 strategies to analyze basins of attraction")
        return None
    
    # For 2-strategy games, create a 1D grid from all MELO to all CDA
    if game.num_strategies == 2:
        # Generate grid of initial mixtures
        grid_mixtures = []
        for i in range(num_points + 1):
            p = i / num_points
            mix = torch.zeros(game.num_strategies, device=game.game.device)
            mix[0] = p  # MELO_0_100 probability
            mix[1] = 1 - p  # MELO_100_0 probability
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
        
        # Visualize the basins of attraction
        # Plot 1: Basin of Attraction
        plt.figure(figsize=(12, 6))
        x_vals = [mix[0].item() for mix in grid_mixtures]
        
        # Create colors for each equilibrium
        colors = {"All MELO": "blue", "All CDA": "red", "Mixed": "purple", "None": "gray"}
        point_colors = [colors[eq] for eq in converged_to_eq]
        
        plt.scatter(x_vals, [1] * len(x_vals), c=point_colors, s=50)
        plt.yticks([])
        plt.xlabel('Initial MELO Probability')
        plt.title('Basin of Attraction (Initial → Final State)')
        
        # Add a legend
        from matplotlib.lines import Line2D
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

        # Plot 3: Heatmap showing trajectories
        # Find the maximum trace length
        max_trace_len = max(len(trace) for trace in traces)
        
        # Create an array with the right dimensions
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
        # For games with more than 2 strategies, we'd need a different visualization
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
    num_zi_agents = 15
    holding_period = 290
    sim_time = 10000  # 1×10^4 timesteps as specified
    num_iterations = 10
    num_strategies = 2 
    batch_size = int(math.factorial(num_mobi_agents + num_strategies - 1) / 
                     (math.factorial(num_mobi_agents)  * math.factorial(num_strategies - 1)))
    print(batch_size)
    
    print(f"Running EGTA with {num_mobi_agents} MOBI and {num_zi_agents} ZI agents")
    print(f"Holding period: {holding_period}")
    print(f"Simulation time: {sim_time}")
    
    # Create simulator with paper-specified parameters
    simulator = MeloSimulator(
        num_strategic=num_mobi_agents, 
        sim_time=sim_time,
        lam=0.006,           # Background trader arrival rate: once every 167 timesteps
        mean=1000000,        # Fundamental mean: 1×10^6
        r=0.05,              # Mean-reversion parameter: 5×10^-2
        shock_var=100,       # Fundamental shock variance: 1×10^2
        q_max=10,            # Maximum position: 10
        pv_var=5000000,      # Private value variance: 5×10^6
        shade=[10, 30],      # Surplus offset bounds: [10, 30]
        holding_period=holding_period,  
        lam_melo=0.001,      # MOBI trader arrival rate: 1×10^-3 (once every 1000 timesteps)
        num_zi=num_zi_agents,  
        num_hbl=0, 
        reps=10000           # Number of simulations per strategy profile: 10,000
    )
    
    strategies = simulator.get_strategies()
    print(f"Available strategies: {strategies}")
    
    for strategy, params in simulator.strategy_params.items():
        print(f"{strategy}: CDA={params['cda_proportion']}, MELO={params['melo_proportion']}")
    
    scheduler = DPRScheduler(
        strategies=strategies,
        num_players=simulator.get_num_players(),
        batch_size=batch_size,
        reduction_size=4, #num_mobi - 1 / reduction_size - 1 needs to be an integer
        seed=42
    )
    
    # Alternative: Use RandomScheduler for more diverse profile sampling
    # scheduler = RandomScheduler(
    #     strategies=strategies,
    #     num_players=simulator.get_num_players(),
    #     batch_size=20,
    #     seed=42
    # )
    
    egta = EGTA(
        simulator=simulator,
        scheduler=scheduler,
        device=device,
        output_dir="results/mobi_zi_egta",
        max_profiles=batch_size,
        seed=42
    )
    
    # Run EGTA
    print("Running EGTA...")
    start_time = time.time()
    game = egta.run(
        max_iterations=num_iterations,
        profiles_per_iteration=20,
        save_frequency=1,
        verbose=True
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
    
    # Get equilibrium from EGTA
    if egta.equilibria:
        eq_mixture, eq_regret = egta.equilibria[0]  # Get the first (usually best) equilibrium
        
        print("\nEquilibria found by EGTA:")
        
        # Calculate welfare for each equilibrium
        print("\nWelfare Analysis of Equilibria:")
        welfare_data = []
        welfare_data_exp = []  # Exponentiated welfare
        labels = []
        
        strategy_to_idx = {name: i for i, name in enumerate(game.strategy_names)}
        
        profile_payoffs = {}
        for profile_data in egta.payoff_data:
            profile_str = str([(player_id, strat) for player_id, strat, _ in profile_data])
            if profile_str not in profile_payoffs:
                profile_payoffs[profile_str] = []
            profile_payoffs[profile_str].extend([payoff for _, _, payoff in profile_data])
        
        for i, (mixture, regret_val) in enumerate(egta.equilibria):
            #calculate expected payoff as weighted sum of profile payoffs
            expected_welfare = 0.0
            weight_sum = 0.0
            
            # For each profile in our payoff data
            for profile_str, payoffs in profile_payoffs.items():
                # Count strategy occurrences in profile
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
                
                # Calculate the average payoff for this profile
                avg_payoff = sum(payoffs) / len(payoffs) if payoffs else 0
                
                # Add to expected welfare
                expected_welfare += profile_prob * avg_payoff
                weight_sum += profile_prob
            
            if weight_sum > 0:
                expected_welfare /= weight_sum
            
            # Convert from log space to direct payoff
            expected_welfare_exp = np.exp(expected_welfare)
            
            welfare_data.append(expected_welfare)
            welfare_data_exp.append(expected_welfare_exp)
            labels.append(f"Eq {i+1}")
            
            print(f"\nEquilibrium {i+1} (regret: {regret_val:.6f}):")
            print(f"  Expected MOBI Agent Welfare (log space): {expected_welfare:.6f}")
            print(f"  Expected MOBI Agent Welfare (actual): {expected_welfare_exp:.2f}")
            for j, strategy in enumerate(game.strategy_names):
                print(f"  {strategy}: {mixture[j].item():.6f}")
        
        # Create two charts - one for log space, one for direct space
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        ax1.bar(labels, welfare_data)
        ax1.set_ylabel('Expected Agent Welfare (log space)')
        ax1.set_title('Welfare Analysis - Log Space')
        for i, v in enumerate(welfare_data):
            ax1.text(i, v + 0.02, f'{v:.2f}', ha='center')
        
        ax2.bar(labels, welfare_data_exp)
        ax2.set_ylabel('Expected Agent Welfare (actual)')
        ax2.set_title('Welfare Analysis - Actual Values')
        for i, v in enumerate(welfare_data_exp):
            ax2.text(i, v * 1.02, f'{v:.0f}', ha='center')
            
        plt.tight_layout()
        plt.savefig('mobi_zi_welfare.png')
        plt.show()
        
        print("\nDetailed analysis of best equilibrium:")
        for i, strategy in enumerate(game.strategy_names):
            print(f"{strategy}: {eq_mixture[i].item():.6f}")
        
        regret_val = eq_regret if isinstance(eq_regret, float) else eq_regret.item()
        print(f"Regret: {regret_val:.6f}")
        
        # Save equilibrium visualization
        plt.figure(figsize=(10, 6))
        plt.bar(game.strategy_names, eq_mixture.cpu().numpy())
        plt.ylabel('Probability')
        plt.title(f'Equilibrium Distribution (Regret: {regret_val:.6f})')
        plt.ylim(0, 1)
        
        for i, v in enumerate(eq_mixture.cpu().numpy()):
            plt.text(i, v + 0.02, f'{v:.4f}', ha='center')
        
        plt.savefig('mobi_zi_equilibrium.png')
        plt.show()
        
        cda_idx = game.strategy_names.index("MELO_100_0") if "MELO_100_0" in game.strategy_names else 0
        all_cda_exists = False
        all_cda_eq_idx = -1
        
        for i, (eq_mix, _) in enumerate(egta.equilibria):
            if eq_mix[cda_idx].item() > 0.99:  
                all_cda_exists = True
                all_cda_eq_idx = i
                break
        
        if all_cda_exists:
            print(f"\nAn all-CDA equilibrium exists (Equilibrium {all_cda_eq_idx+1}) as expected")
        else:
            print("\nWARNING: No all-CDA equilibrium found among the equilibria, which is unexpected")
        
        if eq_mixture[cda_idx].item() > 0.99:
            print("The best equilibrium is predominantly CDA")
        else:
            print("The best equilibrium is NOT predominantly CDA (but another equilibrium may be)")
        
        #basins_result = analyze_basins_of_attraction(game, num_points=100, iters=1000)
        
        return game, eq_mixture
   

if __name__ == "__main__":
    os.makedirs("results/mobi_zi_egta", exist_ok=True)
    game, eq_mixture = run_mobi_zi_egta() 