#from pydantic import Model

import torch


class Replay: # TODO: inherit from pydantic Model
    state: torch.Tensor
    goal: torch.Tensor
    action: torch.Tensor
    reward: float
    next_state: torch.Tensor
    is_finished: bool

    def __init__(self, state, goal, action, reward, next_state, is_finished) -> None:
        self.state = state
        self.goal = goal
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.is_finished = is_finished

    def __str__(self) -> str:
        return (
            "Replay("
                f"state={self.state.tolist()}, "
                f"goal={self.goal.tolist()}, "
                f"action={self.action.tolist()}, "
                f"reward={self.reward}, "
                f"next_state={self.next_state.tolist()}, "
                f"is_finished={self.is_finished}"
            ")"
        )
    

    def __repr__(self) -> str:
        return str(self)


class CircularBuffer:

    def __init__(self, size: int) -> None:
        self._size = size
        self._queue = []


    def enqueue(self, element):
        self._queue.append(element)
        self._queue = self._queue[-self._size:]
        

    def to_list(self):
        return self._queue
