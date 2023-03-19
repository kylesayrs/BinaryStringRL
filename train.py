import numpy

from config import Config
from environment import BitStringEnvironment
from dqn import DQN
from policy import Policy
from replay import Replay, CircularBuffer
from her import create_proximal_goal_replays

def train(dqn: DQN, policy: Policy, config: Config):
    replay_buffer = CircularBuffer(config.REPLAY_BUFFER_SIZE)
    batch_counter = 0
    cumulative_loss = 0.0
    losses = [0.0]  # dumb solution
    num_steps_needed = []

    for episode_i in range(config.NUM_EPISODES):
        environment = BitStringEnvironment(config.STRING_LENGTH, config.DEVICE)
        if config.VERBOSITY >= 2: print(environment, end="")

        num_environment_steps = 0
        while (
            not environment.is_finished()
            and num_environment_steps < config.MAX_EPISODE_STEPS
        ):
            # do action in environment
            state, goal = environment.get_state_and_goal()
            action = policy.get_action(dqn, state, goal)
            next_state, reward = environment.perform_action(action)
            if config.VERBOSITY >= 3: print("\r" + str(environment), end="")

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
            if config.HER_ENABLED:
                additional_replays = create_proximal_goal_replays(
                    replay,
                    reward_function=BitStringEnvironment.reward_function,
                    is_finished_function=BitStringEnvironment.is_finished_function,
                    max_distance=config.HER_MAX_DISTANCE
                )
                replay_buffer.enqueue_multiple(additional_replays)

            # optimize dqn
            batch = numpy.random.choice(replay_buffer.to_list(), config.BATCH_SIZE)
            loss_value = dqn.step_batch(batch)
            cumulative_loss += loss_value

            num_environment_steps += 1
            batch_counter += 1

            # logging
            if batch_counter >= config.LOGGING_RATE:
                losses.append(cumulative_loss / config.LOGGING_RATE)
                cumulative_loss = 0.0
                batch_counter = 0

        # per episode updates
        policy.step()
        dqn.update_target_network()
        
        # logging
        if config.VERBOSITY >= 1:
            print("\r" + str(environment), end="")
            print(
                f" | {num_environment_steps:3d} / {config.MAX_EPISODE_STEPS:3d}"
                f" | {episode_i} / {config.NUM_EPISODES}"
            , end="")

        if config.VERBOSITY >= 2:
            print(
                f" | loss: {losses[-1]:.3f}"
                f" | epsilon: {policy.epsilon:.2f}"
            , end="")

        if config.VERBOSITY > 0:
            print()

        num_steps_needed.append(num_environment_steps)

    metrics = {
        "loss": losses,
        "num_steps": num_steps_needed,
    }

    return dqn, policy, metrics


def evaluate(dqn: DQN, policy: Policy, config: Config):
    num_steps_needed = []
    num_solved = 0

    for episode_i in range(config.NUM_EVAL_EPISODES):
        environment = BitStringEnvironment(config.STRING_LENGTH, config.DEVICE)
        if config.VERBOSITY >= 2: print(environment, end="")

        num_environment_steps = 0
        while (
            not environment.is_finished()
            and num_environment_steps < config.STRING_LENGTH
        ):
            # do action in environment
            state, goal = environment.get_state_and_goal()
            action = policy.get_action(dqn, state, goal)
            _, _ = environment.perform_action(action)
            if config.VERBOSITY >= 2: print("\r" + str(environment), end="")

            # increment
            num_environment_steps += 1

        num_steps_needed.append(num_environment_steps)
        if environment.is_finished():
            num_solved += 1

        if config.VERBOSITY >= 1:
            print("\r" + str(environment), end="")
            print(
                f" | {num_environment_steps:3d} / {config.STRING_LENGTH:3d}"
                f" | {episode_i} / {config.NUM_EVAL_EPISODES}"
            )

    metrics = {
        "num_steps": num_steps_needed,
        "num_solved": num_solved,
    }

    return metrics
