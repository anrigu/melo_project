from marketsim.event.event_queue import EventQueue
from marketsim.fourheap.fourheap import FourHeap
from marketsim.fundamental.fundamental_abc import Fundamental
from marketsim.market.market import Market
from marketsim.fourheap.melo_fourheap import MELOFourHeap

class MeloMarket(Market):
    def __init__(self, fundamental: Fundamental, time_steps, holding_period):
        super().__init__(fundamental, time_steps)
        self.order_book = MELOFourHeap(holding_period)
        self.orders_rec = []

    def clear_market(self):
        new_orders = self.order_book.market_clear()
        self.matched_orders += new_orders
        return new_orders

    def withdraw_all(self, agent_id: int, order_tracker):
        self.order_book.withdraw_all(agent_id, order_tracker)

    def step(self, best_bid, best_ask, order_tracker):
        # TODO Need to figure out how to handle ties for price and time
        orders = self.event_queue.step()
        for order in orders:
            if order.quantity <= 0:
                continue
            # print(f"EventQueue INSERTING orders at time {self.get_time()}: {[o.order_id for o in orders]}")
            self.order_book.insert(order, order_tracker)
            self.order_book.update_best_bid(best_bid)
            self.order_book.update_best_ask(best_ask)
            self.order_book._update_midpoint()
        self.order_book.update_eligiblity_queue(self.get_time() - 1, order_tracker)
        self.order_book.update_active_queue(self.get_time() - 1, order_tracker)
        new_orders = self.order_book.matching_orders(self.get_time() - 1, order_tracker)

        return new_orders

    def update_queues(self, best_bid=None, best_ask=None, order_tracker={}):
        if best_bid or best_ask:
            self.order_book.update_best_bid(best_bid)
            self.order_book.update_best_ask(best_ask)
            self.order_book._update_midpoint()
        self.order_book.update_eligiblity_queue(self.get_time(), order_tracker)
        self.order_book.update_active_queue(self.get_time(), order_tracker)
        return self.order_book.matching_orders(self.get_time(), order_tracker)
        
    def get_midprices(self):
        return self.order_book.midprices

    def reset(self, fundamental=None):
        self.order_book = FourHeap()
        self.matched_orders = []
        self.event_queue = EventQueue()
        if self.fundamental:
            self.fundamental = fundamental
