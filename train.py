import torch
import numpy

from config import Config
from environment import BitStringEnvironment
from dqn import DQN
from policy import Policy, StrictlyGreedyPolicy
from replay import Replay, CircularBuffer
from her import create_proximal_goal_replays


def train(dqn: DQN, policy: Policy, config: Config):
    replay_buffer = CircularBuffer(config.REPLAY_BUFFER_SIZE)
    losses = []
    num_steps_needed = []

    for episode_i in range(config.NUM_EPISODES):
        environment = BitStringEnvironment(config.STRING_LENGTH, config.DEVICE)

        # sample replays
        num_environment_steps = 0
        while (
            not environment.is_finished()
            and num_environment_steps < config.MAX_EPISODE_STEPS
        ):
            # do action in environment
            state, goal = environment.get_state_and_goal()
            action = policy.get_action(dqn, state, goal, network="query")
            next_state, reward = environment.perform_action(action)

            # save replay of action
            replay = Replay(
                state=state,
                goal=goal,
                action=action,
                reward=reward,
                next_state=next_state,
                is_finished=environment.is_finished()
            )
            replay_buffer.enqueue(replay)

            # HER: add replays with virtual goals
            if config.HER_MAX_DISTANCE > 0:
                additional_replays = create_proximal_goal_replays(
                    replay,
                    reward_function=BitStringEnvironment.reward_function,
                    is_finished_function=BitStringEnvironment.is_finished_function,
                    max_distance=config.HER_MAX_DISTANCE
                )
                replay_buffer.enqueue_multiple(additional_replays)

            # increment
            num_environment_steps += 1

        # cycle: perform optimization
        if episode_i % config.EPISODES_PER_CYCLE == 0:

            cumulative_loss = 0.0
            for _ in range(config.BATCHES_PER_CYCLE):
                batch = numpy.random.choice(replay_buffer.to_list(), config.BATCH_SIZE)
                dqn_loss = dqn.step_batch(batch)

                policy.step()

                cumulative_loss += dqn_loss.item()

            dqn.update_target_network(config.DQN_MOMENTUM)
            losses.append(cumulative_loss)
            #losses.append(cumulative_loss / config.BATCH_SIZE / config.BATCHES_PER_CYCLE)
        
        # logging
        if episode_i % config.LOGGING_RATE == 0:
            eval_metrics = evaluate(dqn, config)

            if config.VERBOSITY >= 1:
                print(eval_metrics["sample_environment"], end="")
                print(
                    f" | episode: {episode_i:3d} / {config.NUM_EPISODES:3d}"
                    f" | eval acc: {eval_metrics['num_solved']:3d} / {config.NUM_EVAL_EPISODES:3d}"
                    # f" | avg len: {int(sum(eval_metrics['num_steps']) / config.NUM_EVAL_EPISODES):3d} / {config.MAX_EPISODE_STEPS:3d}"
                    f" | train loss: {losses[-1] / config.BATCH_SIZE / config.BATCHES_PER_CYCLE:.3e}"
                )

        num_steps_needed.append(num_environment_steps)

    metrics = {
        "loss": losses,
        "num_steps": num_steps_needed,
    }

    return dqn, policy, metrics


@torch.no_grad()
def evaluate(dqn: DQN, config: Config):
    policy = StrictlyGreedyPolicy.get_instance()
    num_steps_needed = []
    num_solved = 0

    for episode_i in range(config.NUM_EVAL_EPISODES):
        environment = BitStringEnvironment(config.STRING_LENGTH, config.DEVICE)

        num_environment_steps = 0
        while (
            not environment.is_finished()
            and num_environment_steps < config.MAX_EPISODE_STEPS
        ):
            # do action in environment
            state, goal = environment.get_state_and_goal()
            action = policy.get_action(dqn, state, goal, network="target")
            _, _ = environment.perform_action(action)

            # increment
            num_environment_steps += 1

        num_steps_needed.append(num_environment_steps)
        if environment.is_finished():
            num_solved += 1

    return {
        "sample_environment": environment,
        "num_steps": num_steps_needed,
        "num_solved": num_solved,
    }
