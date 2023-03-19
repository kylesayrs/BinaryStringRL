from pydantic import BaseModel, Field, validator


class Config(BaseModel):
    # environment
    STRING_LENGTH: int = Field(default=3)

    # policy
    POLICY_EPSILON: float = Field(default=0.7)
    POLICY_EPSILON_MAX: float = Field(default=0.7)

    # simulation
    NUM_EPISODES: int = Field(default=10000)
    MAX_EPISODE_STEPS: int = Field(default=5)  # spend longer on harder goals
    REPLAY_BUFFER_SIZE: int = Field(default=256)

    # HER
    HER_ENABLED: bool = Field(default=False)
    HER_MAX_DISTANCE: int = Field(default=0)

    # optimization
    LEARNING_RATE: float = Field(default=0.03)
    BATCH_SIZE: int = Field(default=32)
    GAMMA: float = Field(default=0.5)  # coefficient on future q values
    DQN_MOMENTUM: float = Field(default=0.55)  # per episode
    NUM_EVAL_EPISODES: int = Field(default=500)

    # logging
    VERBOSITY: int = Field(
        description=(
            "0: No logging\n"
            "1: Print the final state of each episode\n"
            "2: Print loss and policy epsilon\n"
            "3: Print the intermediate states in a buffer\n"
        ),
        default=2
    )
    LOGGING_RATE: int = Field(description="per batch", default=100)

    # hardware
    DEVICE = Field(default="cpu")


    @validator("BATCH_SIZE")
    def batch_size_less_than_replay_buffer(cls, value):
        assert value <= cls.REPLAY_BUFFER_SIZE
