import tqdm
import numpy
import matplotlib.pyplot as plt

from config import Config
from environment import BitStringEnvironment
from policy import EGreedyPolicy, EGreedyPolicyWithDelta
from dqn import DQN
from replay import Replay, CircularBuffer
from utils import moving_average


if __name__ == "__main__":
    config = Config()

    dqn = DQN(
        config.STRING_LENGTH,
        config.DISCOUNT,
        config.DQN_MOMENTUM,
        config.LEARNING_RATE,
    )
    policy = EGreedyPolicyWithDelta(config.POLICY_EPSILON, config.POLICY_EPSILON_DELTA)

    replay_buffer = CircularBuffer(config.REPLAY_BUFFER_SIZE)
    cumulative_loss = 0.0
    batch_counter = 0
    losses = []
    num_steps_needed = []

    # simulate and optimize
    for episode_i in range(config.NUM_EPISODES):
        progress = tqdm.tqdm(total=config.MAX_EPISODE_STEPS)
        environment = BitStringEnvironment(config.STRING_LENGTH)

        num_environment_steps = 0
        while True:
            if environment.is_finished():
                break

            print(environment)
            state, goal = environment.get_state_and_goal()
            action = policy.get_action(dqn, state, goal)
            next_state, reward = environment.perform_action(action)
            is_finished = environment.is_finished()
            print(environment)

            replay = Replay(
                state=state,
                goal=goal,
                action=action,
                reward=reward,
                next_state=next_state,
                is_finished=is_finished
            )
            replay_buffer.enqueue(replay)

            # optimize dqn and policy
            batch = numpy.random.choice(replay_buffer.to_list(), config.BATCH_SIZE, replace=True)
            loss_value = dqn.step_batch(batch)
            cumulative_loss += loss_value

            batch_counter += 1
            if batch_counter >= config.LOGGING_RATE:
                print(f"cumulative_loss: {cumulative_loss}")

                losses.append(cumulative_loss)

                cumulative_loss = 0.0
                batch_counter = 0

            policy.step()

            # increment and check for finished
            num_environment_steps += 1
            progress.update(1)
            if is_finished or num_environment_steps >= config.MAX_EPISODE_STEPS:
                num_steps_needed.append(num_environment_steps)
                break


    plt.plot(losses, label="losses")
    plt.legend()
    plt.show()

    plt.plot(moving_average(num_steps_needed, n=10), label="num_steps_needed")
    plt.legend()
    plt.show()
