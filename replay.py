from typing import List, Any
from pydantic import BaseModel, Field

import torch


class Replay(BaseModel):
    state: torch.Tensor = Field()
    goal: torch.Tensor = Field()
    action: torch.Tensor = Field()
    reward: float = Field()
    next_state: torch.Tensor = Field()
    is_finished: bool = Field()

    class Config:
        arbitrary_types_allowed = True


class CircularBuffer:
    def __init__(self, size: int) -> None:
        self._size = size
        self._queue = []


    def enqueue(self, element: Any):
        self._queue.append(element)
        self._queue = self._queue[-self._size:]


    def enqueue_multiple(self, elements: List[Any]):
        self._queue += elements
        self._queue = self._queue[-self._size:]
        

    def to_list(self):
        return self._queue


    def __rep__(self):
        return self.to_list()
