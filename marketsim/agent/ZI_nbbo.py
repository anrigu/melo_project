import random
from marketsim.agent.agent import Agent
from marketsim.market.market import Market
from marketsim.fourheap.order import Order
from marketsim.private_values.private_values import PrivateValues
from marketsim.fourheap.constants import BUY, SELL
from typing import List


class ZIAgent(Agent):
    def __init__(self, agent_id: int, market: Market, q_max: int, eta: float, shade: List, offset: float = 0, pv_var: float = 5e6):
        self.agent_id = agent_id
        self.market = market
        self.pv = PrivateValues(q_max, pv_var)
        self.position = 0
        self.offset = offset
        self.eta = eta
        self.shade = shade
        self.cash = 0
        self.nbbo_price = 0
        self.bbo_weight = .5

    
    def get_id(self) -> int:
        return self.agent_id

    #estimate fundamental value takes in market BBO 
    def estimate_fundamental(self):
        mean, r, T = self.market.get_info()
        t = self.market.get_time()
        val = self.market.get_fundamental_value()
        midpoints = self.market.get_midprices()
        rho = (1-r)**(T-t)
        estimate_fundemental = (1-rho)*mean + rho*val
        if len(midpoints) == 0:
            estimate = estimate_fundemental
        else:
            midpoint = midpoints[-1]
            estimate = (1-self.bbo_weight)*estimate_fundemental + self.bbo_weight*midpoint
        return estimate

    def take_action(self, side):
        t = self.market.get_time()
        estimate = self.estimate_fundamental()
        spread = self.shade[1] - self.shade[0]
        price = estimate + side*spread*random.random() + self.shade[0]


        return Order(
            price=price,
            quantity=1,
            agent_id=self.get_id(),
            time=t,
            order_type=side,
            order_id=random.randint(1, 10000000)
                     )

    def update_position(self, q, p):
        self.position += q
        self.cash += p

    def __str__(self):
        return f'ZI{self.agent_id}'

    def get_pos_value(self) -> float:
        return self.pv.value_at_position(self.position)

    def reset(self):
        self.position = 0
        self.cash = 0


