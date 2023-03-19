from typing import Optional

import torch
from termcolor import colored


class BitStringEnvironment:
    def __init__(self, string_length: int, device: str) -> None:
        self.state = torch.randint(0, 2, size=(string_length, ), dtype=torch.int8, device=device)
        self.goal = torch.randint(0, 2, size=(string_length, ), dtype=torch.int8, device=device)


    @staticmethod
    def reward_function(state: torch.Tensor, goal: torch.Tensor):
        return 1 if torch.all(state == goal) else -1
    
    @staticmethod
    def is_finished_function(state: torch.Tensor, goal: torch.Tensor) -> bool:
        return bool(torch.all(state == goal))


    def get_state_and_goal(self):
        return self.state.clone(), self.goal.clone()


    def get_reward(self):
        return self.reward_function(self.state, self.goal)
    
    
    def is_finished(self) -> bool:
        return self.is_finished_function(self.state, self.goal)
        

    def perform_action(self, action: torch.Tensor) -> float:
        action_index = torch.argmax(action)
        next_state = self.state.clone()
        next_state[action_index] = 1 - next_state[action_index]
        self.state = next_state

        return next_state, self.get_reward()
    

    def __str__(self) -> str:
        return "".join([
            (
                colored(str(int(state_bit)), "green")
                if state_bit == goal_bit else
                colored(str(int(state_bit)), "red")
            )
            for state_bit, goal_bit in zip(self.state, self.goal)
        ])
