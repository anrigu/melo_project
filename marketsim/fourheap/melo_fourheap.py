from collections import deque, defaultdict
from typing import List
from marketsim.fourheap import constants
from marketsim.fourheap.fourheap import FourHeap
from marketsim.fourheap.order import Order, CancelledOrder, MatchedOrder
from marketsim.fourheap.order_queue import OrderQueue
import math
import copy
import numpy as np

class MELOFourHeap(FourHeap):
    """
    A four-heap data structure implementation for M-ELO (Midpoint Extended Limit Order) orders.
    
    M-ELO orders require a mandatory holding period before becoming eligible for matching.
    The structure maintains separate queues for orders in different states:
    - Eligibility queues: Orders waiting for midpoint cross
    - Activation queues: Orders that crossed midpoint but are in holding period
    - Active queues: Orders that completed holding period and are ready for matching
    - Matched/Cancelled queues: Orders that have been matched or cancelled
    
    Attributes:
        curr_best_bid (float): Current best bid price in the market
        curr_best_ask (float): Current best ask price in the market
        midpoint (float): Current midpoint price ((best_bid + best_ask) / 2)
        order_timestamps (dict): Mapping of order IDs to their timestamp
        time (int): Current time step
        holding_period (int): Required holding period before order activation
    """

    def __init__(self, holding_period=50, plus_one=False):
        """
        Initialize the M-ELO four-heap structure.

        Args:
            holding_period (int): Mandatory waiting period before orders become eligible
        """
        super().__init__(plus_one)
        self.curr_best_bid = -math.inf
        self.curr_best_ask = math.inf 
        self.midpoint = 1e5
        self.order_timestamps = {}
        self.holding_period = holding_period

        #cancelled_queues due to midpoint rise during holding
        self.buy_cancelled: List[CancelledOrder] = []
        self.sell_cancelled: List[CancelledOrder] = []

        #matched_queues
        self.buy_matched_orders: List[MatchedOrder] = []
        self.sell_matched_orders: List[MatchedOrder] = []

        #active queues
        self.buy_active_queue: deque[Order] = deque([])
        self.sell_active_queue: deque[Order] = deque([])

        #Pending wait time after activation
        self.buy_activation_queue: deque[(Order, int)] = deque([])
        self.sell_activation_queue: deque[(Order, int)] = deque([])

        #Waiting for midpoint cross for activation (sorted by price)
        self.buy_eligibility_queue = OrderQueue(is_max_heap=True)
        self.sell_eligibility_queue = OrderQueue(is_max_heap=False)

    def insert(self, order: Order):
        """
        Insert a new order into the appropriate queue based on its price relative to midpoint.
        
        Args:
            order (Order): The order to be inserted
        """
        # print(f"Inserting order {order.order_id} at time {order.time}")  # Add this debug line
        #TODO: REMOVE OLD ORDER -> Maybe this is done in market?
        self.agent_id_map[order.agent_id].append(order.order_id)

        #If crosses midpoint on placement
        if order.order_type == constants.BUY:
            if not math.isnan(self.midpoint) and order.price >= self.midpoint:
                self.buy_activation_queue.append((order, order.time))
            else:
                self.buy_eligibility_queue.add_order(order)
        else:
            if not math.isnan(self.midpoint) and order.price <= self.midpoint:
                self.sell_activation_queue.append((order, order.time))
            else:
                self.sell_eligibility_queue.add_order(order)
    
    def update_active_queue(self, curr_time):
        """
        Move orders from activation queues to active queues if they've completed their holding period.
        Orders are processed in FIFO order within each queue.
        """
        # print("CALLED AT THIS TIMESTEP", curr_time)
        while self.buy_activation_queue and curr_time - self.buy_activation_queue[0][1] >= self.holding_period:
            self.buy_active_queue.append(self.buy_activation_queue.popleft()[0])
        
        while self.sell_activation_queue and curr_time - self.sell_activation_queue[0][1] >= self.holding_period:
            self.sell_active_queue.append(self.sell_activation_queue.popleft()[0])

    def withdraw_all(self, agent_id: int):
        for order_id in self.agent_id_map[agent_id]:
            self.remove(order_id)
        self.agent_id_map[agent_id] = []

    def update_eligiblity_queue(self, curr_time):
        """
        Check orders in eligibility queues and move them to activation queues if they cross midpoint.
        """
        if not math.isnan(self.midpoint):
            while True:
                peek_value = self.buy_eligibility_queue.peek()
                if peek_value == float('inf') or peek_value == float('-inf'):
                    break
                if peek_value >= self.midpoint:
                    self.buy_activation_queue.append((self.buy_eligibility_queue.peek_order(), curr_time))
                    self.buy_eligibility_queue.remove(self.buy_eligibility_queue.peek_order_id())
                else:
                    break

            while True:
                peek_value = self.sell_eligibility_queue.peek()
                if peek_value == float('inf') or peek_value == float('-inf'):
                    break
                if peek_value <= self.midpoint:
                    self.sell_activation_queue.append((self.sell_eligibility_queue.peek_order(), curr_time))
                    self.sell_eligibility_queue.remove(self.sell_eligibility_queue.peek_order_id())
                else:
                    break

        
    def matching_orders(self, curr_time):
        """
        Match compatible orders from buy and sell active queues.
        
        Orders are matched at the midpoint price. Orders that would result in losses
        (buy orders above midpoint or sell orders below midpoint) are cancelled.
        Partial matches are supported, with remaining quantities staying in the queue.
        """
        #NOTE: The matched orders get split up by quantities so that the orders in the same indices of the matched_queue are matched.
        new_matched = [[], []]
        if not math.isnan(self.midpoint):
            while self.buy_active_queue and self.sell_active_queue: 
                # Revalidate order queues before processing matches
                while self.buy_active_queue and self.midpoint > self.buy_active_queue[0].price:
                    self.buy_cancelled.append(self.buy_active_queue.popleft())

                while self.sell_active_queue and self.midpoint < self.sell_active_queue[0].price:
                    self.sell_cancelled.append(self.sell_active_queue.popleft())

                # If after cancellations there are no matching orders, exit
                if not self.buy_active_queue or not self.sell_active_queue:
                    break

                buy_match_order = copy.deepcopy(self.buy_active_queue[0])
                sell_match_order = copy.deepcopy(self.sell_active_queue[0])
                matched_quantity = min(self.buy_active_queue[0].quantity, self.sell_active_queue[0].quantity)
                self.buy_active_queue[0].quantity -= matched_quantity
                self.sell_active_queue[0].quantity -= matched_quantity

                # Handle fully matched buy order
                if self.buy_active_queue[0].quantity == 0:
                    new_matched[0].append(MatchedOrder(self.midpoint, curr_time, buy_match_order))
                    self.buy_active_queue.popleft()
                else:
                    buy_match_order.quantity = matched_quantity
                    new_matched[0].append(MatchedOrder(self.midpoint, curr_time, buy_match_order))

                # Handle fully matched sell order
                if self.sell_active_queue[0].quantity == 0:
                    new_matched[1].append(MatchedOrder(self.midpoint, curr_time, sell_match_order))
                    self.sell_active_queue.popleft()
                else:
                    sell_match_order.quantity = matched_quantity
                    new_matched[1].append(MatchedOrder(self.midpoint, curr_time, sell_match_order))

            self.buy_matched_orders.extend(new_matched[0])
            self.sell_matched_orders.extend(new_matched[1])

        return new_matched

    def remove(self, order_id: int):
        for order in self.buy_active_queue:
            if order.order_id == order_id:
                self.buy_active_queue.remove(order)
                return
        
        for order in self.sell_active_queue:
            if order.order_id == order_id:
                self.sell_active_queue.remove(order)
                return
        
        for order_tuple in self.buy_activation_queue:
            if order_tuple[0].order_id == order_id:
                self.buy_activation_queue.remove(order_tuple)
                return
        
        for order_tuple in self.sell_activation_queue:
            if order_tuple[0].order_id == order_id:
                self.sell_activation_queue.remove(order_tuple)
                return
        
        if self.buy_eligibility_queue.contains(order_id):
            self.buy_eligibility_queue.remove(order_id)
            return
        
        if self.sell_eligibility_queue.contains(order_id):
            self.sell_eligibility_queue.remove(order_id)
            return

    def market_clear(self):
        matched_orders = self.buy_matched_orders + self.sell_matched_orders
        return matched_orders
    
    
    def update_best_bid(self, new_best_bid):
        """
        Update the best bid price and recalculate midpoint.
        
        Args:
            new_best_bid (float): New best bid price
        """
        self.curr_best_bid = new_best_bid
    
    def update_best_ask(self, new_best_ask):
        """
        Update the best ask price and recalculate midpoint.
        
        Args:
            new_best_ask (float): New best ask price
        """
        self.curr_best_ask = new_best_ask

    def _update_midpoint(self):
        """Calculate and update the midpoint price based on current best bid and ask."""
        self.midpoint = (self.curr_best_bid + self.curr_best_ask) / 2
        if self.midpoint == float('inf') or self.midpoint == float('-inf'):
            self.midpoint = math.nan
        

    def observe(self) -> str:
        """
        Generate a string representation of the current heap state.
        
        Returns:
            str: Formatted string showing the state of all queues
        """
        s = '--------------\n'
        names = ['buy_matched', 'buy_unmatched', 'sell_matched', 'sell_unmatched']
        for i, heap in enumerate(self.heaps):
            s += names[i]
            s += '\n'
            # s += f'Top order_id: {heap.peek_order().order_id}\n'
            s += f'Top price: {abs(heap.peek())}\n'
            s += f'Number of orders: {heap.count()}\n\n\n'

        return s
