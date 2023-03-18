from pydantic import BaseModel, Field, validator


class Config(BaseModel):
    # environment
    STRING_LENGTH: int = Field(default=2)

    # policy
    POLICY_EPSILON: float = Field(default=0.7)
    POLICY_EPSILON_DELTA: float = Field(default=0.00)

    # simulation
    NUM_EPISODES: int = Field(default=5000)
    MAX_EPISODE_STEPS: int = Field(default=10)
    REPLAY_BUFFER_SIZE: int = Field(default=50)

    # optimization
    LEARNING_RATE: float = Field(default=0.08)
    BATCH_SIZE: int = Field(default=25)
    DISCOUNT: float = Field(default=0.01)  # lower means more discount
    DQN_MOMENTUM: float = Field(default=0.55)

    # logging
    LOGGING_RATE: int = Field(default=10)
    NUM_EVAL_EPISODES: int = Field(default=500)

    # hardware
    DEVICE = Field(default="cpu")


    @validator("BATCH_SIZE")
    def batch_size_less_than_replay_buffer(cls, value):
        assert value <= cls.REPLAY_BUFFER_SIZE

