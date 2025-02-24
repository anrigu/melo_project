from abc import ABC, abstractmethod
from fourheap.order import Order
from typing import List


class Agent(ABC):
    @abstractmethod
    def get_id(self) -> int:
        pass

    @abstractmethod
    def take_action(self, side: bool, seed: int = None) -> List[Order]:
        pass

    @abstractmethod
    def melo_take_action(self, side: bool, quantity:int, seed: int = None) -> List[Order]:
        pass

    @abstractmethod
    def reset(self):
        pass

    def get_pos_value_melo(self) -> float:
        pass

    def get_pos_value(self) -> float:
        pass

