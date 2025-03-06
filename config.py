from pydantic import BaseModel, Field, validator


class Config(BaseModel):
    # environment
    STRING_LENGTH: int = Field(default=30)

    # policy
    POLICY_TYPE: str = Field(default="EGreedyPolicy")  # HER: EGreedyPolicyWithNoise
    POLICY_EPSILON: float = Field(default=0.8)  # HER: 0.8
    POLICY_EPSILON_MAX: float = Field(default=None)  # HER: None
    POLICY_NOISE_STD: float = Field(default=0.05)  # HER: 0.05

    # simulation
    NUM_EPISODES: int = Field(default=160_000)  # HER: 160_000 = 16 episodes * 50 cycles * 200 epochs
    MAX_EPISODE_STEPS: int = Field(default=30)  # HER: STRING_LENGTH
    REPLAY_BUFFER_SIZE: int = Field(default=1024)  # HER: 1_000_000

    # HER
    HER_MAX_DISTANCE: int = Field(default=1)

    # optimization
    LEARNING_RATE: float = Field(default=0.001)  # HER: 0.001
    EPISODES_PER_CYCLE: int = Field(default=16)  # HER: 16
    BATCHES_PER_CYCLE: int = Field(default=40)  # HER: 40
    BATCH_SIZE: int = Field(default=128)  # HER: 128
    GAMMA: float = Field(default=0.98)  # coefficient on future q values HER: 0.98
    DQN_MOMENTUM: float = Field(default=0.05)  # per cycle HER: 0.05
    NUM_EVAL_EPISODES: int = Field(default=500)

    # logging
    VERBOSITY: int = Field(
        description=(
            "0: No logging\n"
            "1: Print the final state of each episode\n"
            "2: Print loss\n"
            "3: Print the intermediate states in a buffer\n"
        ),
        default=2
    )
    LOGGING_RATE: int = Field(description="episodes per log", default=1000)

    # hardware
    DEVICE: str = Field(default="cpu")


    @validator("BATCH_SIZE")
    def batch_size_less_than_replay_buffer(cls, value):
        assert value <= cls.REPLAY_BUFFER_SIZE
    

    @validator("GAMMA")
    def large_gamma(cls, value):
        assert value <= 1.0


    @validator("MAX_EPISODE_STEPS")
    def episode_steps_greater_than_string_length(cls, value):
        assert value >= cls.STRING_LENGTH
