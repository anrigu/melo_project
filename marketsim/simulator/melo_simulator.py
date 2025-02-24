import random
from agent.agent import Agent 
from agent.hbl_agent import HBLAgent
from fourheap.constants import BUY, SELL, MELO, CDA
from market.market import Market
from market.melo_market import MeloMarket
from fundamental.lazy_mean_reverting import LazyGaussianMeanReverting
from agent.zero_intelligence_agent import ZIAgent
import torch.distributions as dist
import torch
import math
from collections import defaultdict


class MELOSimulatorSampledArrival:
    def __init__(self,
                 num_background_agents: int,
                 sim_time: int,
                 num_assets: int = 1,
                 lam: float = 0.1,
                 mean: float = 100,
                 r: float = .05,
                 shock_var: float = 10,
                 q_max: int = 10,
                 pv_var: float = 5e6,
                 shade=None,
                 eta: float = 0.2,
                 hbl_agent: bool = False,
                 lam_r: float = None,
                 holding_period = 50
                 ):

        if shade is None:
            shade = [10, 30]
        if lam_r is None:
            lam_r = lam

        self.num_agents = num_background_agents
        self.num_assets = num_assets
        self.sim_time = sim_time
        self.lam = lam
        self.lam_r = lam_r
        self.time = 0
        self.hbl_agent = hbl_agent
        self.holding_period = holding_period

        self.arrivals = defaultdict(list)
        self.arrivals_sampled = 10000
        self.initial_arrivals = sample_arrivals(lam, self.num_agents)
        self.arrival_times = sample_arrivals(lam_r, self.arrivals_sampled)
        self.arrival_index = 0
        
        fundamental = LazyGaussianMeanReverting(mean=mean, final_time=sim_time, r=r, shock_var=shock_var)
        self.market = Market(fundamental=fundamental, time_steps=sim_time)
        self.meloMarket = MeloMarket(fundamental, sim_time, self.holding_period)

        self.agents = {}
        #Market is only passed in for access to fundamental. Melo doesn't care about that.
        for agent_id in range(num_background_agents + 1):
            self.arrivals[self.arrival_times[self.arrival_index].item()].append(agent_id)
            self.arrival_index += 1
            self.agents[agent_id] = (
                ZIAgent(
                    agent_id=agent_id,
                    market=self.market,
                    q_max=q_max,
                    shade=shade,
                    pv_var=pv_var,
                    eta=eta
                ))

    def step(self):
        agents = self.arrivals[self.time]
        if self.time < self.sim_time:
            self.market.event_queue.set_time(self.time)
            self.meloMarket.event_queue.set_time(self.time)
            for agent_id in agents:
                agent: Agent = self.agents[agent_id]
                self.meloMarket.withdraw_all(agent_id)
                self.market.withdraw_all(agent_id)
                #TODO: CHANGE BACK TO RANDOM.
                # marketSelection = random.choice(MELO, CDA)
                marketSelection = MELO
                
                side = random.choice([BUY, SELL])
                if marketSelection == MELO:
                    #PLACE MELO
                    #TODO: Probably should be sampled froma distribution or something.
                    quantity = 100
                    orders = agent.melo_take_action(side, quantity) 
                    self.meloMarket.add_orders(orders)
                    new_orders = self.meloMarket.step(self.market.order_book.get_best_bid(), self.market.order_book.get_best_ask())
                    midpoint = (self.market.order_book.get_best_bid() + self.market.order_book.get_best_ask()) / 2
                    if math.isnan(midpoint):
                        midpoint = 1e5
                    for side_orders in new_orders:
                        for matched_order in side_orders:
                            #TODO: change to update the MELO position not the CDA position becasue PVs are different.
                            agent_id = matched_order.order.agent_id
                            quantity = matched_order.order.order_type*matched_order.order.quantity
                            cash = -midpoint*matched_order.order.quantity*matched_order.order.order_type
                            self.agents[agent_id].melo_update_position(quantity, cash)

                else:
                    #PLACE CDA ORDER
                    orders = agent.take_action(side)
                    self.market.add_orders(orders)
                    new_orders = self.market.step()
                    for matched_order in new_orders:
                        agent_id = matched_order.order.agent_id
                        quantity = matched_order.order.order_type*matched_order.order.quantity
                        cash = -matched_order.price*matched_order.order.quantity*matched_order.order.order_type
                        self.agents[agent_id].update_position(quantity, cash)

                if self.arrival_index == self.arrivals_sampled:
                    self.arrival_times = sample_arrivals(self.lam_r, self.arrivals_sampled)
                    self.arrival_index = 0
                self.arrivals[self.arrival_times[self.arrival_index].item() + 1 + self.time].append(agent_id)
                self.arrival_index += 1
                
            #Default return type if no matches = [[], []]
            melo_matched_orders = self.meloMarket.update_queues(self.market.order_book.get_best_bid(), self.market.order_book.get_best_ask())
            if len(melo_matched_orders[0]) > 0:
                for matched_order in melo_matched_orders:
                    agent_id = matched_order.order.agent_id
                    quantity = matched_order.order.order_type*matched_order.order.quantity
                    cash = -matched_order.price*matched_order.order.quantity*matched_order.order.order_type
                    self.agents[agent_id].update_position(quantity, cash)
        else:
            self.end_sim()

    def end_sim(self):
        fundamental_val = self.market.get_final_fundamental()
        values = {}
        for agent_id in self.agents:
            agent = self.agents[agent_id]
            values[agent_id] = agent.get_pos_value() + agent.position*fundamental_val + agent.cash
        print(f'At the end of the simulation we get {values}')
        input()
        return values

    def run(self):
        for t in range(self.sim_time):
            if t >= 2500:
                a = 1
            if self.arrivals[t]:
                try:
                    self.step()
                except KeyError:
                    print(self.arrivals[self.time])
                    return self.market, self.meloMarket
            else:
                melo_matched_orders = self.meloMarket.update_queues()
                if len(melo_matched_orders[0]) > 0:
                    for matched_order in melo_matched_orders:
                        agent_id = matched_order.order.agent_id
                        quantity = matched_order.order.order_type*matched_order.order.quantity
                        cash = -matched_order.price*matched_order.order.quantity*matched_order.order.order_type
                        self.agents[agent_id].update_position(quantity, cash)
            self.time += 1
        self.step()


def sample_arrivals(p, num_samples):
    geometric_dist = dist.Geometric(torch.tensor([p]))
    return geometric_dist.sample((num_samples,)).squeeze()  # Returns a tensor of 1000 sampled time steps