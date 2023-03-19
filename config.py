from pydantic import BaseModel, Field, validator


class Config(BaseModel):
    # environment
    STRING_LENGTH: int = Field(default=7)

    # policy
    POLICY_EPSILON: float = Field(default=0.8)
    POLICY_EPSILON_DELTA: float = Field(default=0.00)

    # simulation
    NUM_EPISODES: int = Field(default=5000)
    MAX_EPISODE_STEPS: int = Field(default=15)
    REPLAY_BUFFER_SIZE: int = Field(default=256)

    # HER
    HER_ENABLED: bool = Field(default=True)
    HER_MAX_DISTANCE: int = Field(default=1)

    # optimization
    LEARNING_RATE: float = Field(default=0.03)
    BATCH_SIZE: int = Field(default=32)
    DISCOUNT: float = Field(default=0.8)  # coefficient on future q values
    DQN_MOMENTUM: float = Field(default=0.55)

    # logging
    VERBOSITY: int = Field(
        description=(
            "0: No logging\n"
            "1: Print the final state of each episode\n"
            "2: Print the intermediate states in a buffer\n"
        ),
        default=1
    )
    LOGGING_RATE: int = Field(default=10)
    NUM_EVAL_EPISODES: int = Field(default=500)

    # hardware
    DEVICE = Field(default="cpu")


    @validator("BATCH_SIZE")
    def batch_size_less_than_replay_buffer(cls, value):
        assert value <= cls.REPLAY_BUFFER_SIZE
