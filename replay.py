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


if __name__ == "__main__":
    replay = Replay(
        state=torch.tensor([1, 2, 3]),
        goal=torch.tensor([3, 2, 1]),
        action=torch.tensor([1, 0, 0]),
        reward=1.0,
        next_state=torch.tensor([0, 1, 0]),
        is_finished=False,
    )
