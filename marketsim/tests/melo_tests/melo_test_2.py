import sys
import os
import math

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from collections import defaultdict
from marketsim.simulator.melo_simulator import MELOSimulatorSampledArrival
from tqdm import tqdm
from marketsim.agent.agent import Agent
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# --- Simulation Parameters ---
REPS = 1  # Run multiple simulations for aggregation
SIM_TIME = 10000
NUM_ZI = 30
NUM_HBL = 0
NUM_STRATEGIC = 9 # These will be MeloAgents
TOTAL_BACKGROUND = NUM_ZI + NUM_HBL
HOLDING_PERIOD = 10 # Reduced holding period
SHOCK_VAR = 5e2    # Increased shock variance
LAM_MELO = 1e-3    # Arrival rate for Melo agents
LAM = 6e-3         # Arrival rate for ZI/HBL agents
MEAN = 1e6
R = 0.01
Q_MAX = 10
PV_VAR = 5e6
SHADE = [10, 30]
RECORD_INTERVAL = 100 # How often to record queue stats (for single run analysis)

# --- Strategy Definition (Always MELO) ---
strategy_name = "MELO_0_100"
strategies_list = [strategy_name]
strategy_params_dict = {
    strategy_name: {"cda_proportion": 0.0, "melo_proportion": 1.0}
}
strategy_counts_dict = {
    strategy_name: NUM_STRATEGIC
}

# --- Data Storage ---
cumulative_payoffs = defaultdict(float)
positions = defaultdict(float)
# Aggregated M-ELO stats
total_melo_matches_all_reps = []
avg_match_time_all_reps = []
total_melo_volume_all_reps = []
all_match_times = [] 

single_run_melo_trades = []
single_run_midpoints = []
single_run_best_bids = []
single_run_best_asks = []
single_run_melo_queue_stats = defaultdict(list)


print(f"Running {REPS} simulations...")
for rep in tqdm(range(REPS)):
    
    if REPS == 1:
        melo_queue_history = defaultdict(list) 
        midpoint_history = [] 
        best_bid_history = [] 
        best_ask_history = [] 

    sim = MELOSimulatorSampledArrival(num_background_agents=TOTAL_BACKGROUND,
                                      sim_time=SIM_TIME,
                                      lam=LAM,
                                      mean=MEAN,
                                      num_strategic=NUM_STRATEGIC,
                                      lam_melo=LAM_MELO,
                                      r=R,
                                      shock_var=SHOCK_VAR,
                                      q_max=Q_MAX,
                                      num_zi=NUM_ZI,
                                      num_hbl=NUM_HBL,
                                      pv_var=PV_VAR,
                                      shade=SHADE,
                                      holding_period=HOLDING_PERIOD,
                                      # Pass the explicit strategy definitions
                                      strategies=strategies_list, 
                                      strategy_counts=strategy_counts_dict,
                                      strategy_params=strategy_params_dict
                                      )
    for t in range(sim.sim_time):
        if REPS == 1 and t % RECORD_INTERVAL == 0:
            order_book = sim.meloMarket.order_book
            midpoint = order_book.midpoint
            best_bid = order_book.curr_best_bid
            best_ask = order_book.curr_best_ask

            melo_queue_history['buy_eligibility'].append((t, order_book.buy_eligibility_queue.count()))
            melo_queue_history['sell_eligibility'].append((t, order_book.sell_eligibility_queue.count()))
            melo_queue_history['buy_activation'].append((t, len(order_book.buy_activation_queue)))
            melo_queue_history['sell_activation'].append((t, len(order_book.sell_activation_queue)))
            melo_queue_history['buy_active'].append((t, len(order_book.buy_active_queue)))
            melo_queue_history['sell_active'].append((t, len(order_book.sell_active_queue)))
            midpoint_history.append((t, midpoint if not math.isnan(midpoint) else np.nan)) # Handle potential NaN
            best_bid_history.append((t, best_bid if best_bid != -math.inf else np.nan))
            best_ask_history.append((t, best_ask if best_ask != math.inf else np.nan))

        
        if sim.arrivals[t] or sim.arrivals_melo[t]:
            try:
                sim.step()
            except KeyError:
                print(f"KeyError at time {t} for arrivals: {sim.arrivals[t]}")
                break 
        else:
            # No arrivals, just update queues
            sim.meloMarket.event_queue.set_time(sim.time)
            sim.market.event_queue.set_time(sim.time)
            sim.meloMarket.update_queues(
                sim.market.order_book.get_best_bid(),
                sim.market.order_book.get_best_ask()
            )

        sim.time += 1
        if sim.time >= sim.sim_time:
             sim.step() # Call final step like in original run()
             break

    # --- End of Simulation Rep ---
    payoffs = sim.end_sim()

    # Collect M-ELO matches from the completed run
    melo_matches_this_run = sim.meloMarket.order_book.buy_matched_orders + sim.meloMarket.order_book.sell_matched_orders

    # Calculate stats for this run
    num_matches = len(melo_matches_this_run)
    total_melo_matches_all_reps.append(num_matches)

    if num_matches > 0:
        match_times = [m.time for m in melo_matches_this_run]
        total_volume = sum(m.order.quantity for m in melo_matches_this_run)
        avg_match_time = np.mean(match_times)
        
        avg_match_time_all_reps.append(avg_match_time)
        total_melo_volume_all_reps.append(total_volume)
        all_match_times.extend(match_times) # Collect all times for histogram
    else:
        avg_match_time_all_reps.append(np.nan) # Use NaN if no matches
        total_melo_volume_all_reps.append(0)


    # Accumulate payoffs
    for agent_id, payoff in payoffs.items():
        cumulative_payoffs[agent_id] += payoff
        if agent_id in sim.agents:
             positions[agent_id] += abs(sim.agents[agent_id].position)

    # Store detailed history only if REPS == 1
    if REPS == 1:
         single_run_melo_trades = [(m.time, m.price, m.order.quantity, m.order.order_type) for m in melo_matches_this_run]
         single_run_midpoints = midpoint_history
         single_run_best_bids = best_bid_history
         single_run_best_asks = best_ask_history
         single_run_melo_queue_stats = melo_queue_history

print("Simulations finished.")

# --- Post-Processing and Analysis ---

# Compute average payoffs
average_payoffs = {agent_id: cumulative_payoffs[agent_id] / REPS for agent_id in cumulative_payoffs}
avg_position = {agent_id: positions[agent_id] / REPS for agent_id in positions}

# Print average payoffs by agent group
print("\n--- Average Payoffs ---")
print(average_payoffs)

strategic_keys = [i for i in average_payoffs if i >= TOTAL_BACKGROUND]
non_strategic_keys = [i for i in average_payoffs if i < TOTAL_BACKGROUND]

if non_strategic_keys:
    avg_non_strategic = np.mean([average_payoffs[k] for k in non_strategic_keys])
    print(f"Average Payoff for Non-Strategic Agents (ZI/HBL): {avg_non_strategic:.2f}")
else:
     print("No non-strategic agents found in results.")

if strategic_keys:
    # Convert tensor payoffs to float before calculating mean
    strategic_payoffs = [average_payoffs[k].item() if hasattr(average_payoffs[k], 'item') else average_payoffs[k] for k in strategic_keys]
    avg_strategic = np.mean(strategic_payoffs)
    print(f"Average Payoff for Strategic Agents (M-ELO): {avg_strategic:.2f}")
else:
     print("No strategic (M-ELO) agents found in results.")

# --- Aggregated M-ELO Activity Analysis ---
print(f"\n--- Aggregated M-ELO Market Activity ({REPS} Runs) ---")
avg_total_matches = np.mean(total_melo_matches_all_reps)
std_total_matches = np.std(total_melo_matches_all_reps)
print(f"Average M-ELO Matches per Run: {avg_total_matches:.2f} (Std: {std_total_matches:.2f})")

avg_total_volume = np.mean(total_melo_volume_all_reps)
std_total_volume = np.std(total_melo_volume_all_reps)
print(f"Average M-ELO Volume per Run: {avg_total_volume:.2f} (Std: {std_total_volume:.2f})")

# Calculate average of the average match times (ignoring NaNs)
valid_avg_times = [t for t in avg_match_time_all_reps if not np.isnan(t)]
if valid_avg_times:
    overall_avg_match_time = np.mean(valid_avg_times)
    print(f"Overall Average M-ELO Match Time (across runs with matches): {overall_avg_match_time:.2f}")
else:
    print("No M-ELO matches occurred in any run.")


# --- Plotting ---

# Plot aggregated histogram of match times if there were matches
if all_match_times:
    print("\nGenerating aggregated M-ELO match time histogram...")
    plt.figure(figsize=(12, 6))
    plt.hist(all_match_times, bins=50, density=True, alpha=0.7, label=f'Match Times ({REPS} runs)')
    plt.xlabel("Simulation Timestep", fontsize=14)
    plt.ylabel("Density", fontsize=14)
    plt.title(f"Distribution of M-ELO Match Times (HP={HOLDING_PERIOD}, SV={SHOCK_VAR})", fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Plot detailed single-run analysis only if REPS == 1
if REPS == 1:
    print("\nGenerating plots for single run...")

    # 1. Plot M-ELO Queue Sizes Over Time
    plt.figure(figsize=(14, 8))
    for queue_name, data in single_run_melo_queue_stats.items():
        times = [item[0] for item in data]
        counts = [item[1] for item in data]
        plt.plot(times, counts, label=queue_name, marker='.', linestyle='-', markersize=4)

    plt.xlabel("Timestep", fontsize=14)
    plt.ylabel("Number of Orders", fontsize=14)
    plt.title(f"M-ELO Queue Sizes Over Time (Single Run, HP={HOLDING_PERIOD}, SV={SHOCK_VAR})", fontsize=16)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

    # 2. Plot CDA BBO, Midpoint, and M-ELO Trade Times
    plt.figure(figsize=(14, 8))

    # Plot BBO and Midpoint
    times_bbo = [t for t, _ in single_run_best_bids]
    bids = [p for _, p in single_run_best_bids]
    asks = [p for _, p in single_run_best_asks]
    midpoints = [p for _, p in single_run_midpoints]

    plt.plot(times_bbo, bids, label="CDA Best Bid", color="#D22B2B", alpha=0.7)
    plt.plot(times_bbo, asks, label="CDA Best Ask", color="#197bff", alpha=0.7)
    plt.plot(times_bbo, midpoints, label="CDA Midpoint", color="grey", linestyle=':', alpha=0.8)

    # Plot M-ELO trades as vertical lines and scatter points
    melo_buy_matches = [(t, p) for t, p, q, side in single_run_melo_trades if side == 1]
    melo_sell_matches = [(t, p) for t, p, q, side in single_run_melo_trades if side == -1]
    
    if melo_buy_matches or melo_sell_matches:
        # Vertical Lines (optional, keep for context or remove if cluttered)
        all_match_times = [t for t, p in melo_buy_matches] + [t for t, p in melo_sell_matches]
        min_trade_time = min(all_match_times) if all_match_times else 0
        max_trade_time = max(all_match_times) if all_match_times else SIM_TIME
        valid_bids = [b for b in bids if not np.isnan(b)]
        valid_asks = [a for a in asks if not np.isnan(a)]
        plot_min_y = min(valid_bids) if valid_bids else MEAN - 5*SHADE[1]
        plot_max_y = max(valid_asks) if valid_asks else MEAN + 5*SHADE[1]
        # plt.vlines([t for t, p in melo_buy_matches], plot_min_y, plot_max_y, color="#F28C28", linestyle="dotted", alpha=0.7, linewidth=1.5, label='_nolegend_') # Vertical lines less prominent
        # plt.vlines([t for t, p in melo_sell_matches], plot_min_y, plot_max_y, color="green", linestyle="dotted", alpha=0.7, linewidth=1.5, label='_nolegend_')

        # Scatter points at match time and price
        if melo_buy_matches:
            buy_times, buy_prices = zip(*melo_buy_matches)
            plt.scatter(buy_times, buy_prices, color="#F28C28", marker='^', s=80, label='MELO Buy Match', zorder=5, edgecolors='black') # Orange triangle
        if melo_sell_matches:
            sell_times, sell_prices = zip(*melo_sell_matches)
            plt.scatter(sell_times, sell_prices, color="green", marker='v', s=80, label='MELO Sell Match', zorder=5, edgecolors='black') # Green triangle

    else:
        print("No M-ELO trades occurred in this single run.")

    # Labels and styling
    plt.xlabel("Timestep", fontsize=14)
    plt.ylabel("Price", fontsize=14)
    plt.title(f"CDA BBO, Midpoint, and M-ELO Matches (Single Run, HP={HOLDING_PERIOD}, SV={SHOCK_VAR})", fontsize=16)

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), fontsize=10)

    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

else:
    print(f"\nDetailed single-run plots skipped (REPS = {REPS}). Run with REPS=1 for detailed plots.")


print("\nAnalysis Complete.") 