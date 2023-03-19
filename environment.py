import torch
from termcolor import colored


class BitStringEnvironment:
    def __init__(self, string_length: int, device: str) -> None:
        self.state = torch.randint(0, 2, size=(string_length, ), dtype=torch.int8, device=device)
        self.goal = torch.randint(0, 2, size=(string_length, ), dtype=torch.int8, device=device)


    def get_state_and_goal(self):
        return self.state.clone(), self.goal.clone()


    def get_reward(self):
        return 1 if torch.all(self.state == self.goal) else -1


    def perform_action(self, action: torch.Tensor) -> float:
        action_index = torch.argmax(action)
        next_state = self.state.clone()
        next_state[action_index] = 1 - next_state[action_index]
        self.state = next_state

        return next_state, self.get_reward()


    def is_finished(self):
        return torch.all(self.state == self.goal)
    

    def __str__(self) -> str:
        return "".join([
            (
                colored(str(int(state_bit)), "green")
                if state_bit == goal_bit else
                colored(str(int(state_bit)), "red")
            )
            for state_bit, goal_bit in zip(self.state, self.goal)
        ])
