import random
import numpy as np
import math
from agent.agent import Agent
from market.market import Market
from fourheap.order import Order
from private_values.private_values import PrivateValues
from fourheap.constants import BUY, SELL


class SpoofingAgent(Agent):
    def __init__(self, agent_id: int, market: Market, q_max: int, order_size:int, spoofing_size: int, normalizers: dict, learning:bool):
        self.agent_id = agent_id
        self.market = market
        # if pv != -1:
        #     self.pv=pv
        # else:
        #     self.pv = PrivateValues(q_max, pv_var)
        self.position = 0
        self.spoofing_size = spoofing_size
        self.order_size = order_size
        self.cash = 0
        self.last_value = 0 # value at last time step (liquidate all inventory)
        self.normalizers = normalizers # A dictionary {"fundamental": float, "invt": float, "cash": float}
        self.learning = learning

        # Regular was chosen as a bit more than limit of PV evaluation.
        self.action_normalization = {"regular": 150, "spoofing": 10}

        # self.obs_noise = obs_noise
        # self.prev_arrival_time = 0
        # self.prev_obs_mean = 0
        # self.prev_obs_var = 0
        self.q_max = q_max
        # self.pv_var = pv_var

    # def generate_pv(self):
    #     #Generate new private values
    #     self.pv = PrivateValues(self.q_max, self.pv_var)
        
    def get_id(self) -> int:
        return self.agent_id

    # def noisy_obs(self):
    #     mean, r, T = self.market.get_info()
    #     t = self.market.get_time()
    #     val = self.market.get_fundamental_value()
    #     ot = val + random..normal(0,np.sqrt(self.obs_noise))

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

    #     return (1 - rho) * mean + rho * curr_estimate

    def estimate_fundamental(self):
        mean, r, T = self.market.get_info()
        t = self.market.get_time()
        val = self.market.get_fundamental_value()

        rho = (1-r)**(T-t)

        estimate = (1-rho) * mean + rho*val
        # print(f'It is time {t} with final time {T} and I observed {val} and my estimate is {rho, estimate}')
        return estimate

    def take_action(self, action = (0,0), seed = None):
        '''
            action: tuple (offset from price quote, offset from valuation)
        '''
        t = self.market.get_time()
        random.seed(t + seed)
        placeholder = random.random()
        orderId1 = random.randint(1, 10000000)
        orderId2 = random.randint(1, 10000000)
        # if 1000 < t < 1050:
        #     print(placeholder, orderId1, orderId2)
        #     input()

        regular_order_offset, spoofing_order_offset = action
        # regular_order_offset = action
        # Normalization constants need to be tuned
        if self.learning:
            unnormalized_reg_offset = regular_order_offset * self.action_normalization["regular"]
            unnormalized_spoof_offset = spoofing_order_offset * self.action_normalization["spoofing"] 
            # unnormalized_spoof_offset = 1 
        else:
            #TODO: TUNE THE REG_OFFSET
            unnormalized_reg_offset = 50
            unnormalized_spoof_offset = 1
        
        orders = []
        if math.isinf(self.market.order_book.buy_unmatched.peek()):
            # Should rarely happen since the spoofer enters after t = 1000
            # If it does, just submit a bid that won't lose the spoofer money
            spoofing_price = self.estimate_fundamental()
        else:
            spoofing_price = self.market.order_book.buy_unmatched.peek() - unnormalized_spoof_offset
        
        best_sell = self.market.order_book.sell_unmatched.peek()
        best_buy = self.market.order_book.buy_unmatched.peek()
        if not math.isinf(self.market.order_book.sell_unmatched.peek()) and not math.isinf(self.market.order_book.buy_unmatched.peek()):
            midprice = (best_buy + best_sell) / 2
            regular_order_price = midprice + unnormalized_reg_offset
            base = 1
        else:
            regular_order_price = self.estimate_fundamental() + unnormalized_reg_offset
            base = 0
        
        # Regular order.
        regular_order = Order(
            price=regular_order_price,    
            quantity=self.order_size,
            agent_id=self.get_id(),
            time=t,
            order_type=SELL,
            order_id=orderId1
        )
        orders.append(regular_order)

        # Spoofing Order
        spoofing_order = Order(
            price=spoofing_price,
            quantity=self.spoofing_size,
            agent_id=self.get_id(),
            time=t,
            order_type=BUY,
            order_id=orderId2
        )
        orders.append(spoofing_order)
        
        return orders, base

    def update_position(self, q, p):
        self.position += q
        self.cash += p

    def __str__(self):
        return f'SPF{self.agent_id}'

    # def get_pos_value(self) -> float:
    #     return self.pv.value_at_position(self.position)

    def reset(self):
        self.position = 0
        self.cash = 0
        self.last_value = 0


