from abc import ABC, abstractmethod
from marketsim.fourheap.order import Order, MatchedOrder
from typing import List


class Agent(ABC):
    @abstractmethod
    def get_id(self) -> int:
        pass

    @abstractmethod
    def take_action(self, side: bool, market: int, seed: int = None) -> List[Order]:
        pass
    
    @abstractmethod
    def reset(self):
        pass

    def get_pos_value_melo(self) -> float:
        pass

    def get_pos_value(self) -> float:
        pass