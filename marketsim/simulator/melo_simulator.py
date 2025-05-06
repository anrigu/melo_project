import random
from marketsim.agent.agent import Agent 
from marketsim.agent.hbl_agent import HBLAgent
from marketsim.fourheap.constants import BUY, SELL, MELO, CDA
from marketsim.market.market import Market
from marketsim.market.melo_market import MeloMarket
from marketsim.fundamental.lazy_mean_reverting import LazyGaussianMeanReverting
from marketsim.agent.zero_intelligence_agent import ZIAgent
from marketsim.agent.melo_agent import MeloAgent
import torch.distributions as dist
import torch
import math
from collections import defaultdict, Counter

def sample_arrivals(p, num_samples):
    # Ensure p is a float tensor - this fixes the torch.finfo error
    p_tensor = torch.tensor([float(p)], dtype=torch.float32)
    geometric_dist = dist.Geometric(p_tensor)
    return geometric_dist.sample((num_samples,)).squeeze()  # Returns a tensor of 10000 sampled time steps


class MELOSimulatorSampledArrival:
    def __init__(self,
                 num_background_agents: int,
                 sim_time: int,
                 num_zi: int,
                 num_hbl: int,
                 num_strategic: int,
                 strategies = None,
                 strategy_counts = None,
                 strategy_params = None,
                 num_assets: int = 1,
                 lam: float = 0.1,
                 mean: float = 100,
                 r: float = .05,
                 shock_var: float = 10,
                 q_max: int = 10,
                 pv_var: float = 5e6,
                 shade=None,
                 eta: float = 1.0,
                 hbl_agent: bool = False,
                 lam_r: float = None,
                 holding_period = 50,
                 lam_melo = 0.1,
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
        self.lam_melo = lam_melo
        self.time = 0
        self.hbl_agent = hbl_agent
        self.holding_period = holding_period
        self.strategies = strategies
        self.strategy_counts = strategy_counts
        self.strategy_params = strategy_params

        self.arrivals = defaultdict(list)
        self.arrivals_melo = defaultdict(list)
        self.arrivals_sampled = 10000
        self.initial_arrivals = sample_arrivals(lam, self.num_agents)
        self.arrival_times = sample_arrivals(lam_r, self.arrivals_sampled)
        self.arrival_times_melo = sample_arrivals(lam_melo, self.arrivals_sampled)
        self.arrival_index_melo = 0
        self.arrival_index = 0
        
        self.best_buys = []
        self.best_asks = []
        self.timesteps_melo_trade = []
        self.timesteps_hbl_orders = []
        self.num_orders = 0

        #TODO: DATA TO DELETE LATER
        self.melo_orders = []
        self.fundamental_estimates = []

        fundamental = LazyGaussianMeanReverting(mean=mean, final_time=sim_time, r=r, shock_var=shock_var)
        self.market = Market(fundamental=fundamental, time_steps=sim_time)
        self.meloMarket = MeloMarket(fundamental, sim_time, self.holding_period)

        self.agents = {}

        self.order_tracker = {}

        self.fundamentals = []


        # print("R", r, "SHOCK", shock_var)

        #Market is only passed in for access to fundamental. Melo doesn't care about that
        for agent_id in range(num_zi):
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

        for agent_id in range(num_zi, num_zi + num_hbl):
            self.arrivals[self.arrival_times[self.arrival_index].item()].append(agent_id)
            self.arrival_index += 1
            self.agents[agent_id] = (
                HBLAgent(
                    agent_id=agent_id,
                    market=self.market,
                    q_max=q_max,
                    shade=shade,
                    pv_var=pv_var,
                    L=4,
                    arrival_rate=self.lam
                ))

        if not self.strategies:
            strategic_agent_id = num_zi + num_hbl
            for agent_id in range(num_strategic):
                self.arrivals_melo[self.arrival_times_melo[self.arrival_index_melo].item()].append(strategic_agent_id)
                self.arrival_index_melo += 1
                if agent_id <= 1:
                    cda_proportion = 0
                    melo_proportion = 1
                else:
                    cda_proportion = 0
                    melo_proportion = 1
                self.agents[strategic_agent_id] = (
                    MeloAgent(
                        agent_id=strategic_agent_id,
                        #Not important which market
                        market=self.market,
                        q_max=q_max,
                        pv_var=pv_var,
                        cda_proportion=cda_proportion,
                        melo_proportion=melo_proportion
                    ))
                strategic_agent_id += 1
        else:
            strategic_agent_id = num_zi + num_hbl
            for strategy in self.strategies:
                count = strategy_counts[strategy]
                for _ in range(count):
                    self.arrivals_melo[self.arrival_times_melo[self.arrival_index_melo].item()].append(strategic_agent_id)
                    self.arrival_index_melo += 1
                    params = self.strategy_params[strategy]

                    self.agents[strategic_agent_id] = (
                        MeloAgent(
                            agent_id=strategic_agent_id,
                            #Not important which market
                            market=self.market,
                            meloMarket=self.meloMarket,
                            q_max=q_max,
                            pv_var=pv_var,
                            cda_proportion=params["cda_proportion"],
                            melo_proportion=params["melo_proportion"],
                        ))
                    strategic_agent_id += 1            

    def step(self):
        agents = self.arrivals[self.time]
        melo_agents = self.arrivals_melo[self.time]
        if self.time < self.sim_time:
            melo_placed = False
            self.meloMarket.event_queue.set_time(self.time)
            self.market.event_queue.set_time(self.time)
            
            if agents:
                for agent_id in agents:

                    #Normal orderbook traders
                    agent = self.agents[agent_id]
                    self.market.withdraw_all(agent_id)
                    side = random.choice([BUY, SELL])
                    orders = agent.take_action(side)
                    self.market.add_orders(orders)
                    if isinstance(agent, HBLAgent):
                        self.timesteps_hbl_orders.append((self.time, side, orders[0].price))
                    if self.arrival_index == self.arrivals_sampled:
                        self.arrival_times = sample_arrivals(self.lam_r, self.arrivals_sampled)
                        self.arrival_index = 0
                    self.arrivals[self.arrival_times[self.arrival_index].item() + 1 + self.time].append(agent_id)
                    self.arrival_index += 1
                    
            if melo_agents:
                melo_placed = True
                for agent_id in melo_agents:
                    agent = self.agents[agent_id]
                    self.meloMarket.withdraw_all(agent_id, self.order_tracker)
                    self.market.withdraw_all(agent_id)
                    self.num_orders += 1
                    # Check if the agent is a MeloAgent with strategy parameters
                    assert isinstance(agent, MeloAgent) and hasattr(agent, 'cda_proportion') and hasattr(agent, 'melo_proportion')
                    # Use the strategy parameters to determine market selection
                    if random.random() < agent.melo_proportion:
                        marketSelection = MELO
                    else:
                        marketSelection = CDA
                        
                    side = random.choice([BUY, SELL])
                    if marketSelection == MELO:

                        #PLACE MELO
                        orders = agent.take_action(side, marketSelection) 
                        self.meloMarket.add_orders(orders)
                        
                        assert len(orders) == 1

                        self.melo_orders.append((self.time, orders[0].price, orders[0].order_type))
                        self.order_tracker[orders[0].order_id] = ""

                    else:
                        #PLACE CDA ORDER
                        orders = agent.take_action(side, marketSelection)
                        self.market.add_orders(orders)
                            
                    if self.arrival_index_melo == self.arrivals_sampled:
                        self.arrival_times_melo = sample_arrivals(self.lam_melo, self.arrivals_sampled)
                        self.arrival_index_melo = 0
                    self.arrivals_melo[self.arrival_times_melo[self.arrival_index_melo].item() + 1 + self.time].append(agent_id)
                    self.arrival_index_melo += 1

                new_orders = self.meloMarket.step(self.market.order_book.get_best_bid(), self.market.order_book.get_best_ask(), self.order_tracker)
                if len(new_orders[0]) > 0:
                    for side_orders in new_orders:
                        for matched_order in side_orders:
                            agent_id = matched_order.order.agent_id
                            current_agent: MeloAgent = self.agents[agent_id]
                            current_agent.record_trade(matched_order.order.order_type, matched_order.order.quantity)
                            quantity = matched_order.order.order_type*matched_order.order.quantity
                            cash = -matched_order.price*matched_order.order.quantity*matched_order.order.order_type
                            
                            # Update position and cash
                            current_agent.update_position(quantity, cash)
                            
                            # Update MELO profit based on trade direction
                            ##print(f"MELO Trade: Agent {agent_id}, Side {matched_order.order.order_type}, Quantity {matched_order.order.quantity}, Price {matched_order.price}")
                            ##print(f"Agent {agent_id} MELO Profit before trade: {current_agent.melo_profit}")
                            
                            # Update the MELO profit: BUY (1) decreases profit, SELL (-1) increases profit
                            if matched_order.order.order_type == 1:  # BUY
                                current_agent.melo_profit -= float(matched_order.price) * matched_order.order.quantity
                            else:  # SELL
                                current_agent.melo_profit += float(matched_order.price) * matched_order.order.quantity
                            
                            ##print(f"Agent {agent_id} MELO Profit after trade: {current_agent.melo_profit}")
                        
            new_orders = self.market.step()
            for matched_order in new_orders:
                agent_id = matched_order.order.agent_id
                if isinstance(self.agents[agent_id], MeloAgent):
                    current_agent: MeloAgent = self.agents[agent_id]
                    current_agent.record_trade(matched_order.order.order_type, matched_order.order.quantity)
                    self.timesteps_melo_trade.append((self.time, matched_order.order.quantity, matched_order.order.order_type))
                quantity = matched_order.order.order_type*matched_order.order.quantity
                cash = -matched_order.price*matched_order.order.quantity*matched_order.order.order_type
                self.agents[agent_id].update_position(quantity, cash)

            
            if not melo_placed:
                # This only matters for Melo Market trades. No need to check for CDA trades
                #Default return type if no matches = [[], []]
                melo_matched_orders = self.meloMarket.update_queues(self.market.order_book.get_best_bid(), self.market.order_book.get_best_ask(), self.order_tracker)
                if len(melo_matched_orders[0]) > 0:
                    for side_orders in melo_matched_orders:
                        for matched_order in side_orders:
                            #TODO: change to update the MELO position not the CDA position becasue PVs are different.
                            agent_id = matched_order.order.agent_id
                            current_agent: MeloAgent = self.agents[agent_id]
                            current_agent.record_trade(matched_order.order.order_type, matched_order.order.quantity)
                            quantity = matched_order.order.order_type*matched_order.order.quantity
                            cash = -matched_order.price*matched_order.order.quantity*matched_order.order.order_type
                            current_agent.update_position(quantity, cash)
        else:
            self.end_sim()

    def end_sim(self):
        fundamental_val = self.market.get_final_fundamental()
        values = {}
        positions = {}
        # melo_profits = {}
        for agent_id in self.agents:
            agent = self.agents[agent_id]
            if isinstance(agent, MeloAgent):
                a = agent.position*fundamental_val + agent.cash + sum(quantity * value for quantity, value in agent.melo_pv_history)
                values[agent_id] = a
            else:
                values[agent_id] = agent.get_pos_value() + agent.position*fundamental_val + agent.cash
            positions[agent_id] = agent.position
        counts = Counter(self.order_tracker.values())
            
        # print(f'At the end of the simulation we get {values}')
        # print(f'MELO_ At the end of the simulation we get {melo_profits}')
        # input()
        activation_queue = len(self.meloMarket.order_book.buy_activation_queue) + len(self.meloMarket.order_book.sell_activation_queue)
        active_queue = len(self.meloMarket.order_book.buy_active_queue) + len(self.meloMarket.order_book.sell_active_queue)
        elig_queue = self.meloMarket.order_book.buy_eligibility_queue.count() + self.meloMarket.order_book.sell_eligibility_queue.count()
        return values, positions, self.best_buys, self.best_asks, self.timesteps_melo_trade, self.timesteps_hbl_orders, self.meloMarket.order_book.buy_cancelled, self.meloMarket.order_book.sell_cancelled, self.meloMarket.order_book.removed_eligibility, self.meloMarket.order_book.removed_activation, self.meloMarket.order_book.removed_active, self.melo_orders, self.fundamental_estimates, self.num_orders, elig_queue, activation_queue, active_queue # Return both CDA and MELO profits

    def run(self):
        for t in range(self.sim_time):
            if self.arrivals[t] or self.arrivals_melo[t]:
                # print("ENTERED TIMESTEP", t)
                try:
                    # print(f'CALLING STEP at time {self.time}')
                    self.step()
                except KeyError:
                    print(self.arrivals[self.time])
                    return self.market, self.meloMarket
            else:
                # print("SKIPPEDA T TIMESTEP", t)
                self.meloMarket.event_queue.set_time(self.time)
                self.market.event_queue.set_time(self.time)
                melo_matched_orders = self.meloMarket.update_queues(self.market.order_book.get_best_bid(), self.market.order_book.get_best_ask(), self.order_tracker)
                if len(melo_matched_orders[0]) > 0:
                    for side_orders in melo_matched_orders:
                        for matched_order in side_orders:
                            agent_id = matched_order.order.agent_id
                            current_agent: MeloAgent = self.agents[agent_id]
                            current_agent.record_trade(matched_order.order.order_type, matched_order.order.quantity)
                            quantity = matched_order.order.order_type*matched_order.order.quantity
                            cash = -matched_order.price*matched_order.order.quantity*matched_order.order.order_type
                            self.agents[agent_id].update_position(quantity, cash)

            self.fundamentals.append(self.market.get_fundamental_value())
            self.best_buys.append(self.market.order_book.get_best_bid())
            self.best_asks.append(self.market.order_book.get_best_ask())
            self.fundamental_estimates.append(self.agents[0].estimate_fundamental())
            self.time += 1
        self.step()


def sample_arrivals(p, num_samples):
    # Ensure p is a float tensor - this fixes the torch.finfo error 
    p_tensor = torch.tensor([float(p)], dtype=torch.float32)
    geometric_dist = dist.Geometric(p_tensor)
    return geometric_dist.sample((num_samples,)).squeeze()  # Returns a tensor of 1000 sampled time steps