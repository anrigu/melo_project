import random
import numpy as np
from agent.agent import Agent
from market.market import Market
from fourheap.order import Order
from private_values.private_values import PrivateValues
from fourheap.constants import BUY, SELL
from typing import List
import torch
import math


class ZIAgent(Agent):
    def __init__(self, agent_id: int, market: Market, q_max: int, shade: List, pv_var: float, eta: float = 1.0):
        self.agent_id = agent_id
        self.market = market
        self.q_max = q_max
        self.pv_var = pv_var
        self.pv = PrivateValues(q_max, pv_var)
        #Rand generated on entry
        self.meloPV = 0
        self.position = 0
        self.meloPosition = 0
        self.shade = shade
        self.cash = 0
        self.meloProfit = 0
        self.eta = eta
        self.melo_trades = []
        self.q_max = q_max
        self.pv_var = pv_var

    def generate_pv(self):
        #Generate new private values
        self.pv = PrivateValues(self.q_max, self.pv_var)

    def generate_melo_pv(self):
        self.meloPV = random.uniform(10, 100)

    def get_id(self) -> int:
        return self.agent_id

    # def noisy_obs(self):
    #     mean, r, T = self.market.get_info()
    #     t = self.market.get_time()
    #     val = self.market.get_fundamental_value()
    #     ot = val + np.random.normal(0,np.sqrt(self.obs_noise))

    #     rho_noisy = (1-r)**(t-self.prev_arrival_time)
    #     rho_var = rho_noisy ** 2

    #     prev_estimate = (1-rho_noisy)*mean + rho_noisy*self.prev_obs_mean
    #     prev_var =  rho_var * self.prev_obs_var + (1 - rho_var) / (1 - (1-r)**2) * int(self.market.fundamental.shock_std ** 2)

    #     curr_estimate = self.obs_noise / (self.obs_noise + prev_var) * prev_estimate + prev_var / (self.obs_noise + prev_var) * ot
    #     curr_var = self.obs_noise * prev_var / (self.obs_noise + prev_var)

    #     rho = (1-r)**(T-self.prev_arrival_time)

    #     self.prev_arrival_time = T
    #     self.prev_obs_mean = curr_estimate
    #     self.prev_obs_var = curr_var

        # return (1 - rho) * mean + rho * curr_estimate

    def estimate_fundamental(self):
        mean, r, T = self.market.get_info()
        t = self.market.get_time()
        val = self.market.get_fundamental_value()

        rho = (1-r)**(T-t)

        estimate = (1-rho)*mean + rho*val
        # print(f'It is time {t} with final time {T} and I observed {val} and my estimate is {rho, estimate}')
        return estimate
        # return estimate + np.random.normal(0, np.sqrt(3e5))

    def melo_take_action(self, side: bool, quantity:int, seed: int = None) -> List[Order]:
        t = self.market.get_time()
        self.generate_melo_pv()
        if side == BUY:
            price = self.estimate_fundamental() + self.meloPV
            # price = midpoint + self.meloPv[0]
        else:
            price = self.estimate_fundamental() - self.meloPV
            # price = midpoint - self.meloPv[0]
        
        order = Order(
            price=price,
            quantity=quantity,
            agent_id=self.get_id(),
            time=t,
            order_type=side,
            order_id=random.randint(1, 10000000)
        )
        return [order]

    def take_action(self, side, seed = 0):
        t = self.market.get_time()
        # random.seed(t + seed)
        estimate = self.estimate_fundamental() 
        # estimate = self.estimate_fundamental() + np.random.normal(0, np.sqrt(1e6))
        spread = self.shade[1] - self.shade[0]
        valuation_offset = spread*random.random() + self.shade[0]
        if side == BUY:
            price = estimate + self.pv.value_for_exchange(self.position, BUY) - valuation_offset
        else:
            price = estimate + self.pv.value_for_exchange(self.position, SELL) + valuation_offset
        # if 1000 < t < 1500:
        #     print(f'It is time {t} and I am on {side} as a ZI. My estimate is {estimate}, my position is {self.position}, and my marginal pv is '
        #         f'{self.pv.value_for_exchange(self.position, side)} with offset {valuation_offset}. '
        #         f'Therefore I offer price {price}')

        if self.eta != 1.0:
            if side == BUY:
                surplus = price - estimate
                best_price = self.market.order_book.get_best_ask()
                if (price - best_price) > self.eta*surplus and best_price != np.inf:
                    price = best_price
            else:
                surplus = estimate - price
                best_price = self.market.order_book.get_best_bid()
                if (best_price - price) > self.eta*surplus and best_price != np.inf:
                    price = best_price

        order = Order(
            price=price,
            quantity=1,
            agent_id=self.get_id(),
            time=t,
            order_type=side,
            order_id=random.randint(1, 10000000)
        )
        # print(f'It is timestep {t} and I am assigned {side}. I am making an order with price {price} since my estimate is {estimate} ')
        # print(f'My current position is {self.position} and my private values are {self.pv.values}')

        return [order]

    def update_position(self, q, p):
        self.position += q
        self.cash += p

    def melo_record_trade(self, side, quantity, matched_order) -> None:
        if side == BUY:
            #Reserve price - executed price
            one_unit_profit = matched_order.order.price - matched_order.price
        else:
            #executed price - reserve price
            one_unit_profit = matched_order.price - matched_order.order.price
        total_profit = one_unit_profit * quantity
        self.meloProfit += total_profit

    def __str__(self):
        return f'ZI{self.agent_id}'

    def get_pos_value(self) -> float:
        return self.pv.value_at_position(self.position)

    def reset(self):
        self.position = 0
        self.cash = 0
        self.pv = PrivateValues(self.q_max, self.pv_var)

