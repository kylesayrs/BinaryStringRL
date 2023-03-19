from typing import Callable

import torch
import numpy

from replay import Replay

def _random_flip_bits(string: torch.Tensor, num_flips: int):
    assert num_flips <= len(string), "Cannot flip more bits than string length"

    string = string.clone()
    flip_indices = numpy.random.randint(0, len(string), size=(num_flips, ))
    string[flip_indices] = 1 - string[flip_indices]

    return string


def create_proximal_goal_replays(
    replay: Replay,
    reward_function: Callable,
    is_finished_function: Callable,
    max_distance: int
):
    replays = []
    for distance in range(max_distance + 1):
        new_goal = _random_flip_bits(replay.next_state, distance)

        new_replay = replay.copy(update={
            "goal": new_goal,
            "reward": reward_function(replay.next_state, new_goal),
            "is_finished": is_finished_function(replay.next_state, new_goal),
        })

        replays.append(new_replay)

    return replays
