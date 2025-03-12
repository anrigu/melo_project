from marketsim.simulator.melo_simulator import MELOSimulatorSampledArrival
from tqdm import tqdm
from agent.agent import Agent
import numpy as np

'''
[
['agent_1', 'ROCK', 0],
['agent_2', 'ROCK', 0],
['agent_3', 'ROCK', 0],
['agent_4', 'ROCK', 0],
]
'''
agent_types = []

NUM_AGENTS = 25
NUM_ZI = 23
NUM_HBL = 0
NUM_MELO = NUM_AGENTS - NUM_ZI - NUM_HBL

NUM_SIMULATIONS = 10000

assert NUM_ZI + NUM_HBL + NUM_MELO == NUM_AGENTS

surpluses = []
for agent in NUM_AGENTS:
    agent_arr = ["agent_{}".format(agent)]
    if agent < NUM_ZI:
        agent_arr.append("ZI")
    elif agent < NUM_ZI + NUM_HBL:
        agent_arr.append("HBL")
    else:
        agent_arr.append("MELO")
    agent_arr.append(0)
    agent_types.append(agent_arr)

for _ in tqdm(range(NUM_SIMULATIONS)):
    sim = MELOSimulatorSampledArrival(num_background_agents=NUM_AGENTS, 
                                  sim_time=12000, 
                                  lam=5e-2, 
                                  mean=1e5, 
                                  r=0.05, 
                                  shock_var=5e6, 
                                  q_max=10,
                                  pv_var=5e6,
                                  shade=[500,1000],
                                  num_zi = NUM_ZI,
                                  num_hbl = NUM_HBL,
                                  num_melo = NUM_MELO)
    sim.run()
    fundamental_val = sim.market.get_final_fundamental()
    print(f"Fundamental value: {fundamental_val}")
    values = []
    for agent_id in sim.agents:
        agent = sim.agents[agent_id]
        value = agent.get_pos_value() + agent.position * fundamental_val + agent.cash
        if agent_id >= NUM_ZI + NUM_HBL:
            # Agent is MOBI
            value += agent.melo_profit
        
        values.append(value)
    surpluses.append(values)

avg_payoffs = np.sum(surpluses, axis=0) / NUM_SIMULATIONS
assert len(avg_payoffs) == NUM_AGENTS

from egta.symmetric_game import *
from egta.game import *
from egta.utils.eq_computation import *
from egta.utils.log_multimodal import *
from egta.utils.random_functions import *
from egta.utils.simplex_operations import *
from egta.process_data import *
from egta.reductions.dpr import DPRGAME

device = "cuda" if torch.cuda.is_available() else "cpu"
game = create_symmetric_game_from_data(avg_payoffs, device=device)
dpr_game = DPRGAME(game, 4) 

#solve dpr_game 
#game, mix, iters=1000, offset=0
best_mixture, eq_candidates, regrets = find_equilibria(dpr_game, logging=True)
print(eq_candidates)
print("\nNash Equilibrium:")

for i, strat in enumerate(game.strategy_names):
    print(f"{strat}: {best_mixture[i].item():.4f}")
print(best_mixture)
#print(f"\nMaximum deviation from uniform: {torch.max(torch.abs(eq_mixture - 1/rps_game.num_actions)).item():.6f}")

regret = game.regret(best_mixture)
print(best_mixture)
print(f"Regret at equilibrium: {regret.item():.6f}")

 