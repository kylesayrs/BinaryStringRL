import matplotlib.pyplot as plt

from config import Config
from policy import create_policy_from_config, StrictlyGreedyPolicy
from dqn import DQN
from train import train, evaluate


if __name__ == "__main__":
    config = Config(DEVICE="cpu")

    dqn = DQN(
        config.STRING_LENGTH,
        config.GAMMA,
        config.LEARNING_RATE,
        config.DEVICE,
    )
    policy = create_policy_from_config(config)

    # train
    dqn, policy, train_metrics = train(dqn, policy, config)

    plt.plot(train_metrics["loss"], label="loss")
    plt.legend()
    plt.show()

    # evaluate
    eval_metrics = evaluate(dqn, StrictlyGreedyPolicy(), config)

    print(f"{100 * eval_metrics['num_solved'] / len(eval_metrics['num_steps']):.2f}% solved")

    plt.hist(eval_metrics["num_steps"], bins=range(config.STRING_LENGTH + 1))
    plt.show()
