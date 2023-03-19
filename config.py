from pydantic import BaseModel, Field, validator


class Config(BaseModel):
    # environment
    STRING_LENGTH: int = Field(default=3)

    # policy
    POLICY_EPSILON: float = Field(default=0.7)
    POLICY_EPSILON_DELTA: float = Field(default=0.00)

    # simulation
    NUM_EPISODES: int = Field(default=5000)
    MAX_EPISODE_STEPS: int = Field(default=500)
    REPLAY_BUFFER_SIZE: int = Field(default=128)

    # optimization
    LEARNING_RATE: float = Field(default=0.03)
    BATCH_SIZE: int = Field(default=16)
    DISCOUNT: float = Field(default=0.8)  # coefficient on future q values
    DQN_MOMENTUM: float = Field(default=0.7)

    # logging
    VERBOSITY: int = Field(default=1)
    LOGGING_RATE: int = Field(default=10)
    NUM_EVAL_EPISODES: int = Field(default=500)

    # hardware
    DEVICE = Field(default="cpu")


    @validator("BATCH_SIZE")
    def batch_size_less_than_replay_buffer(cls, value):
        assert value <= cls.REPLAY_BUFFER_SIZE
