from typing import List

import torch

from replay import Replay


class BinaryStringModel(torch.nn.Module):
    def __init__(self, string_length: int) -> None:
        super().__init__()
        
        self.string_length = string_length

        self.linear_0 = torch.nn.Linear(2 * self.string_length, 2 * self.string_length)
        self.linear_1 = torch.nn.Linear(2 * self.string_length, 2 * self.string_length)
        self.linear_2 = torch.nn.Linear(2 * self.string_length, self.string_length)
        self.linear_3 = torch.nn.Linear(self.string_length, self.string_length)
        self.linear_4 = torch.nn.Linear(self.string_length, self.string_length)
        self.linear_5 = torch.nn.Linear(self.string_length, self.string_length)

        self.relu = torch.nn.ReLU()


    def forward(self, state: torch.Tensor, goal: torch.Tensor):
        assert len(state.shape) == 2, "BinaryStringModel forward must receive batch"
        assert len(goal.shape) == 2, "BinaryStringModel forward must receive batch"
        assert state.shape[1] == goal.shape[1], "Number of states != Number of goals"

        # preprocessing
        x = torch.concat([state, goal], dim=1)
        x = x.to(torch.float32)

        # network
        x = self.linear_0(x)
        x = self.relu(x)
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.linear_2(x)
        x = self.relu(x)
        x = self.linear_3(x)
        x = self.relu(x)
        x = self.linear_4(x)
        x = self.relu(x)
        x = self.linear_5(x)

        return x


class DQN:
    def __init__(
        self,
        string_length: int,
        discount: float,
        momentum: float,
        learning_rate: float,
        device: str,
    ) -> None:
        self.string_length = string_length
        self.discount = discount
        self.momentum = momentum
        self.device = device

        self.query_network = BinaryStringModel(string_length=string_length).to(self.device)
        self.target_network = BinaryStringModel(string_length=string_length).to(self.device)
        for param in self.target_network.parameters():
            param.requires_grad = False
        self.update_target_network(1.0)

        #self.optimizer = torch.optim.Adam(self.query_network.parameters(), lr=learning_rate)
        self.optimizer = torch.optim.SGD(self.query_network.parameters(), lr=learning_rate)
        self.criterion = torch.nn.MSELoss()


    def infer_single(self, state: torch.Tensor, goal: torch.Tensor):
        with torch.no_grad():
            input = (state.unsqueeze(0), goal.unsqueeze(0))
            return self.query_network(*input)[0]
    

    def update_target_network(self, momentum=None):
        momentum = self.momentum if momentum is None else momentum

        with torch.no_grad():
            for query_param, target_param in zip(
                self.query_network.parameters(), self.target_network.parameters()
            ):
                target_param.data = (
                    self.momentum * query_param.data +
                    (1 - self.momentum) * target_param.data
                )


    def step_batch(self, batch: List[Replay]):
        # unpack batch
        states = torch.vstack([replay.state for replay in batch])
        goals = torch.vstack([replay.goal for replay in batch])
        actions = torch.vstack([replay.action for replay in batch])
        rewards = torch.tensor([replay.reward for replay in batch], device=self.device)
        next_states = torch.vstack([replay.next_state for replay in batch])
        is_finisheds = torch.tensor([replay.is_finished for replay in batch], device=self.device)

        # compute target values
        with torch.no_grad():
            future_action_qualities = self.target_network(next_states, goals)

            is_not_finished = 1 - is_finisheds.to(torch.float32)
            is_not_finished_transposed = torch.transpose(is_not_finished.unsqueeze(0), 0, 1)
            future_action_qualities *= is_not_finished_transposed  # zero if finished

            target_values = rewards + self.discount * torch.max(future_action_qualities, dim=1).values
            action_indices = (actions == 1)
        
        # optimize
        self.optimizer.zero_grad()
        outputs = self.query_network(states, goals)
        
        # create targets
        with torch.no_grad():
            targets = outputs.clone().detach()
            targets[action_indices] = target_values.to(torch.float32)

        # backwards
        loss = self.criterion(targets, outputs)
        loss.backward()
        self.optimizer.step()

        return loss.item()
