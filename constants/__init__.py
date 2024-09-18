import datetime as dt
import math
from pathlib import Path
from typing import Dict, List, Tuple
import torch

from taoverse.model.competition.data import ( Competition, ModelConstraints )
from taoverse.model.competition.epsilon import FixedEpsilon
from competitions.data import CompetitionId

# ---------------------------------
# Project Constants.
# ---------------------------------

__version__ = "1.0.0"
version_split = __version__.split(".")
__spec_version__ = (
    (1000 * int(version_split[0]))
    + (10 * int(version_split[1]))
    + (1 * int(version_split[2]))
)

# The version of the validator state. When incremented, causes validators
# to start from a fresh state.
VALIDATOR_STATE_VERSION = 2

# The validator WANDB project.
WANDB_PROJECT = "genki-dama"
WANDB_ENTITY = "crynux"
# The uid for this subnet.
SUBNET_UID = 1
# Minimum stake to consider a validator when checking for miners with weights.
WEIGHT_SYNC_VALI_MIN_STAKE = 100_000
# Minimum percent of weight on a vali for a miner to be considered a top miner.
# Since there can be multiple competitions at different reward percentages we can't just check biggest.
WEIGHT_SYNC_MINER_MIN_PERCENT = 0.10
# The root directory of this project.
ROOT_DIR = Path(__file__).parent.parent
# The maximum bytes for the hugging face repo.
MAX_HUGGING_FACE_BYTES: int = 15 * 1024 * 1024 * 1024

MODEL_CONSTRAINTS_BY_COMPETITION_ID: Dict[CompetitionId, ModelConstraints] = {
    CompetitionId.ChIPTUNE_MUSIC_MODEL: ModelConstraints(
                    max_model_parameter_size=6_900_000_000,
                    sequence_length=4096,
                    allowed_architectures=[],
                    kwargs={
                        "torch_dtype": torch.float16,
                    },
                    eval_block_delay=1200,  # ~4 hours.
                    tokenizer="",
                    max_bytes=1024 * 1024 * 1024 * 20,
                    epsilon_func=FixedEpsilon(1.0)
                ),
}

# Schedule of competitions by block.
COMPETITION_SCHEDULE_BY_BLOCK: List[Tuple[int, List[Competition]]] = [
    (
        0,
        [
            Competition(
                CompetitionId.ChIPTUNE_MUSIC_MODEL,
                MODEL_CONSTRAINTS_BY_COMPETITION_ID[CompetitionId.ChIPTUNE_MUSIC_MODEL],
                1.0,
            )
        ],
    )
]

for block_and_competitions in COMPETITION_SCHEDULE_BY_BLOCK:
    assert math.isclose(
        sum(competition.reward_percentage for competition in block_and_competitions[1]),
        1.0,
    )

# ---------------------------------
# Miner/Validator Model parameters.
# ---------------------------------

weights_version_key = __spec_version__

# validator weight moving average term
alpha = 0.5
# validator scoring exponential temperature
# 0.01 gives ~96% to best model with only ~3 receiving any weights.
temperature = 0.01
# validator score boosting for earlier models.
timestamp_epsilon = 0.005
# time required between updates to the chain.
chain_update_cadence = dt.timedelta(minutes=20)
# time required between retrying evaluation of a stale model. (First retry will be immediate).
model_retry_cadence = dt.timedelta(hours=4)
# validator eval batch min to keep for next loop.
sample_min = 5
# We allow the sample_min per competition + 10 additional models to be held at any one time.
updated_models_limit = sample_min + 10
