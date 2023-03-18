class Config:
    # environment
    STRING_LENGTH: int = 2

    # policy
    POLICY_EPSILON: float = 0.7
    POLICY_EPSILON_DELTA: float = 0.00

    # simulation
    NUM_EPISODES: int = 5000
    MAX_EPISODE_STEPS: int = 10
    REPLAY_BUFFER_SIZE: int = 50

    # optimization
    LEARNING_RATE: float = 0.08
    BATCH_SIZE: int = 25
    DISCOUNT: float = 0.01  # lower means more discount
    DQN_MOMENTUM: float = 0.55

    # logging
    LOGGING_RATE: int = 10
    NUM_EVAL_EPISODES: int = 500


    def __init__(self) -> None:
        assert self.BATCH_SIZE <= self.REPLAY_BUFFER_SIZE
