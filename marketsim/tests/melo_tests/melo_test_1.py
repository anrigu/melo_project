from collections import defaultdict
from marketsim.simulator.melo_simulator import MELOSimulatorSampledArrival
from tqdm import tqdm
from agent.agent import Agent
surpluses = []

cumulative_payoffs = defaultdict(float)

for _ in tqdm(range(250)):
    sim = MELOSimulatorSampledArrival(num_background_agents=25, 
                                      sim_time=10000, 
                                      lam=6e-3, 
                                      mean=1e6, 
                                      num_strategic=10,
                                      lam_melo=1e-3,
                                      r=0.05, 
                                      shock_var=1e2, 
                                      q_max=10,
                                      num_zi=14,
                                      num_hbl=1,
                                      pv_var=5e6,
                                      shade=[10,30])
    sim.run()
    payoffs = sim.end_sim()  # Dictionary {agent_id: payoff}

    # Accumulate payoffs for each agent
    for agent_id, payoff in payoffs.items():
        cumulative_payoffs[agent_id] += payoff

# Compute the average payoff for each agent
average_payoffs = {agent_id: cumulative_payoffs[agent_id] / 100 for agent_id in cumulative_payoffs}

# Print or use the results as needed
print(average_payoffs)

# positions = []
# values = []
# melo_profits = []
# fundamental_val = sim.market.get_final_fundamental()

# for agent_id in sim.agents:
#     agent:Agent = sim.agents[agent_id]
#     value = agent.get_pos_value() + agent.position * fundamental_val + agent.cash
#     print(agent.cash, agent.position, agent.get_pos_value(), value)
#     positions.append(agent.position)
#     values.append(value)
#     melo_profits.append(agent.melo_profit)

# import matplotlib.pyplot as plt

# # print(fundamental_val)

# plt.scatter(positions, values)
# plt.xlabel('Position')
# plt.ylabel('Value')
# plt.title(f'Agent Position vs Value at fundamental {fundamental_val}')
# plt.show()
# sum(positions)



 