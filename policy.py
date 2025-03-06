from abc import ABC
import torch
import numpy

from config import Config

from dqn import DQN


class Policy(ABC):
    def get_action(self, dqn: DQN, state: torch.Tensor, goal: torch.Tensor, network: str = "query") -> torch.Tensor:
        raise NotImplementedError()
    

    def step(self):
        pass


class EGreedyPolicy(Policy):
    def __init__(self, epsilon: float) -> None:
        self.epsilon = epsilon


    def get_action(self, dqn: DQN, state: torch.Tensor, goal: torch.Tensor, network: str = "query") -> torch.Tensor:
        action_qualities = dqn.infer_single(state, goal, network=network)
        max_quality_action_index = torch.argmax(action_qualities)

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


class EGreedyPolicyWithDelta(EGreedyPolicy):
    def __init__(self, epsilon: float, delta_epsilon: float) -> None:
        super().__init__(epsilon)
        self.delta_epsilon = delta_epsilon

    
    def step(self):
        self.epsilon += self.delta_epsilon
        self.epsilon = min(self.epsilon, 1.0)


class StrictlyGreedyPolicy():
    def get_action(self, dqn, state: torch.Tensor, goal: torch.Tensor, network: str = "query") -> torch.Tensor:
        action_qualities = dqn.infer_single(state, goal, network=network)
        max_quality_action_index = torch.argmax(action_qualities)
        #print(f"state: {state}")
        #print(f"action_qualities: {action_qualities}: {max_quality_action_index} : {self.epsilon}")

        action = torch.zeros(len(action_qualities))
        action[max_quality_action_index] = 1.0

        return action


class EGreedyPolicyWithNoise(Policy):
    def __init__(self, epsilon: float, noise_std: float) -> None:
        self.epsilon = epsilon
        self.noise_std = noise_std


    def get_action(self, dqn: DQN, state: torch.Tensor, goal: torch.Tensor, network: str = "query") -> torch.Tensor:
        action_qualities = dqn.infer_single(state, goal, network=network)
        max_quality_action_index = torch.argmax(action_qualities)

        # epsilon chance of picking the greedy choice
        if numpy.random.choice([True, False], p=[self.epsilon, 1 - self.epsilon]):
            selected_action_index = max_quality_action_index

            action = torch.zeros(len(action_qualities))
            action[selected_action_index] = 1.0
            noise = numpy.random.normal(0, self.noise_std, len(action[selected_action_index]))
            action[selected_action_index] += action[selected_action_index] * noise

        else:
            other_action_indices = list(range(len(state)))
            other_action_indices.remove(max_quality_action_index)

            selected_action_index = numpy.random.choice(other_action_indices)

            action = torch.zeros(len(action_qualities))
            action[selected_action_index] = 1.0

        return action


def create_policy_from_config(config: Config):
    if config.POLICY_TYPE == "EGreedyPolicy":
        return EGreedyPolicy(config.POLICY_EPSILON)
    
    if config.POLICY_TYPE == "EGreedyPolicyWithDelta":
        return EGreedyPolicyWithDelta(
            config.POLICY_EPSILON,
            (
                (config.POLICY_EPSILON_MAX - config.POLICY_EPSILON)
                / config.NUM_EPISODES
                / config.CYCLE_LENGTH
            ),
        )

    if config.POLICY_TYPE == "EGreedyPolicyWithNoise":
        return EGreedyPolicyWithNoise(
            config.POLICY_EPSILON,
            config.POLICY_NOISE_STD,
        )
    
    raise ValueError(f"Unknown policy type {config.POLICY_TYPE}")
