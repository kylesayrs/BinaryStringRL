import torch
import numpy


class EGreedyPolicy:
    def __init__(self, epsilon: float) -> None:
        self.epsilon = epsilon


    def get_action(self, dqn, state: torch.Tensor, goal: torch.Tensor):
        action_qualities = dqn.infer_single(state, goal)
        max_quality_action_index = torch.argmax(action_qualities)
        print(f"action_qualities: {action_qualities}: {max_quality_action_index} : {self.epsilon}")

        # epsilon chance of picking the greedy choice
        if numpy.random.choice([True, False], p=[self.epsilon, 1 - self.epsilon]):
            selected_action_index = max_quality_action_index

        else:
            other_action_indices = list(range(len(state)))
            other_action_indices.remove(max_quality_action_index)

            selected_action_index = numpy.random.choice(other_action_indices)

        action = torch.zeros(len(action_qualities))
        action[selected_action_index] = 1.0

        return action
    

    def step(self):
        pass


class EGreedyPolicyWithDelta(EGreedyPolicy):
    def __init__(self, epsilon: float, delta_epsilon: float) -> None:
        super().__init__(epsilon)
        self.delta_epsilon = delta_epsilon

    
    def step(self):
        self.epsilon += self.delta_epsilon
        self.epsilon = min(self.epsilon, 1.0)
