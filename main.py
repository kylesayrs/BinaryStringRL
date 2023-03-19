import numpy
import matplotlib.pyplot as plt

from config import Config
from environment import BitStringEnvironment
from policy import Policy, EGreedyPolicy, StrictlyGreedyPolicy
from dqn import DQN
from replay import Replay, CircularBuffer


def train(dqn: DQN, policy: Policy, config: Config):
    replay_buffer = CircularBuffer(config.REPLAY_BUFFER_SIZE)
    cumulative_loss = 0.0
    batch_counter = 0
    losses = []
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
            if config.VERBOSITY >= 2: print("\r" + str(environment), end="")

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

            # optimize dqn and policy
            batch = numpy.random.choice(replay_buffer.to_list(), config.BATCH_SIZE)
            loss_value = dqn.step_batch(batch)
            cumulative_loss += loss_value

            batch_counter += 1
            if batch_counter >= config.LOGGING_RATE:
                losses.append(cumulative_loss)
                cumulative_loss = 0.0
                batch_counter = 0

            policy.step()

            # increment
            num_environment_steps += 1

        num_steps_needed.append(num_environment_steps)
        
        if config.VERBOSITY >= 1:
            print("\r" + str(environment), end="")
            print(
                f" | {num_environment_steps:3d} / {config.MAX_EPISODE_STEPS:3d}"
                f" | {episode_i} / {config.NUM_EPISODES}"
            )

    metrics = {
        "loss": losses,
        "num_steps": num_steps_needed,
    }

    return dqn, policy, metrics


def evaluate(dqn: DQN, policy: Policy, config: Config):
    num_steps_needed = []

    for episode_i in range(config.NUM_EVAL_EPISODES):
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
            _, _ = environment.perform_action(action)
            if config.VERBOSITY >= 2: print("\r" + str(environment), end="")

            # increment
            num_environment_steps += 1

        num_steps_needed.append(num_environment_steps)

        if config.VERBOSITY >= 1:
            print("\r" + str(environment), end="")
            print(
                f" | {num_environment_steps:3d} / {config.MAX_EPISODE_STEPS:3d}"
                f" | {episode_i} / {config.NUM_EVAL_EPISODES}"
            )

    metrics = {
        "num_steps": num_steps_needed,
    }

    return metrics


if __name__ == "__main__":
    config = Config(device="mps")

    dqn = DQN(
        config.STRING_LENGTH,
        config.DISCOUNT,
        config.DQN_MOMENTUM,
        config.LEARNING_RATE,
        config.DEVICE,
    )
    policy = EGreedyPolicy(config.POLICY_EPSILON)

    # train
    dqn, policy, train_metrics = train(dqn, policy, config)

    plt.plot(train_metrics["loss"], label="loss")
    plt.legend()
    plt.show()

    # evaluate (TODO: evaluate with other policies such as weighted probability)
    eval_metrics = evaluate(dqn, StrictlyGreedyPolicy(), config)

    plt.hist(eval_metrics["num_steps"], bins=range(max(eval_metrics["num_steps"]) + 1))
    plt.show()
