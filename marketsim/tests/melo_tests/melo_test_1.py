from marketsim.simulator.melo_simulator import MELOSimulatorSampledArrival
from tqdm import tqdm
from agent.agent import Agent
surpluses = []

for _ in tqdm(range(10000)):
    sim = MELOSimulatorSampledArrival(num_background_agents=25, 
                                  sim_time=12000, 
                                  lam=5e-2, 
                                  mean=1e5, 
                                  r=0.05, 
                                  shock_var=5e6, 
                                  q_max=10,
                                  pv_var=5e6,
                                  shade=[500,1000])
    sim.run()
    fundamental_val = sim.market.get_final_fundamental()
    print(f"Fundamental value: {fundamental_val}")
    values = []
    for agent_id in sim.agents:
        agent = sim.agents[agent_id]
        value = agent.get_pos_value() + agent.position * fundamental_val + agent.cash
        # print(agent.cash, agent.position, agent.get_pos_value(), value)
        values.append(value)
    surpluses.append(sum(values)/len(values))
print(sum(surpluses)/len(surpluses)*25)

positions = []
values = []
melo_profits = []
fundamental_val = sim.market.get_final_fundamental()

for agent_id in sim.agents:
    agent:Agent = sim.agents[agent_id]
    value = agent.get_pos_value() + agent.position * fundamental_val + agent.cash
    print(agent.cash, agent.position, agent.get_pos_value(), value)
    positions.append(agent.position)
    values.append(value)
    melo_profits.append(agent.meloProfit)

import matplotlib.pyplot as plt

print(fundamental_val)

plt.scatter(positions, values)
plt.xlabel('Position')
plt.ylabel('Value')
plt.title(f'Agent Position vs Value at fundamental {fundamental_val}')
plt.show()
sum(positions)



 