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
        self.placed_order_count = 0

    def clear_market(self):
        new_orders = self.order_book.market_clear()
        self.matched_orders += new_orders
       # print(f"[DEBUG M-ELO] Market Clear: Found {len(new_orders)} matched order events.") 
        return new_orders

    def withdraw_all(self, agent_id: int):
      #  print(f"[DEBUG M-ELO] MeloMarket withdrawing all for Agent ID: {agent_id}") 
        self.order_book.withdraw_all(agent_id)

    def step(self, best_bid, best_ask):
        current_time = self.get_time()
       # print(f"[DEBUG M-ELO] --- MeloMarket Step Start: Time={current_time} ---")
        orders = self.event_queue.step()
       # print(f"[DEBUG M-ELO] Event Queue Step returned {len(orders)} orders for Time={current_time}")
        for order in orders:
            if order.quantity <= 0:
               # print(f"[DEBUG M-ELO] Skipping order ID={order.order_id} due to non-positive quantity ({order.quantity})")
                continue
            self.placed_order_count += 1
            self.order_book.insert(order)
            self.order_book.update_best_bid(best_bid)
            self.order_book.update_best_ask(best_ask)
            self.order_book._update_midpoint()
        self.order_book.update_eligiblity_queue(current_time - 1)
        self.order_book.update_active_queue(current_time - 1)
       # print(f"[DEBUG M-ELO] Updating queues for Time={current_time - 1}")
        new_orders = self.order_book.matching_orders(current_time - 1)
        #print(f"[DEBUG M-ELO] Matching orders for Time={current_time - 1} resulted in {len(new_orders[0])} buy matches, {len(new_orders[1])} sell matches.")
        #print(f"[DEBUG M-ELO] --- MeloMarket Step End: Time={current_time} ---")
        return new_orders

    def update_queues(self, best_bid=None, best_ask=None):
        current_time = self.get_time()
       # print(f"[DEBUG M-ELO] --- MeloMarket Update Queues Start: Time={current_time} ---")
        if best_bid or best_ask:
        #    print(f"[DEBUG M-ELO] Updating BB/BA: Bid={best_bid}, Ask={best_ask}")
            self.order_book.update_best_bid(best_bid)
            self.order_book.update_best_ask(best_ask)
            self.order_book._update_midpoint()
        self.order_book.update_eligiblity_queue(current_time)
        self.order_book.update_active_queue(current_time)
       # print(f"[DEBUG M-ELO] Updating queues for Time={current_time}")
        matching_result = self.order_book.matching_orders(current_time)
      #  print(f"[DEBUG M-ELO] Matching orders for Time={current_time} resulted in {len(matching_result[0])} buy matches, {len(matching_result[1])} sell matches.")
      #  print(f"[DEBUG M-ELO] --- MeloMarket Update Queues End: Time={current_time} ---")
        return matching_result
        
    def get_midprices(self):
        return self.order_book.midprices

    def reset(self, fundamental=None):
        self.order_book = FourHeap()
        self.matched_orders = []
        self.event_queue = EventQueue()
        if self.fundamental:
            self.fundamental = fundamental

    def get_placed_count(self) -> int:
        """Return the total count of orders placed (attempted insertions)."""
        return self.placed_order_count
