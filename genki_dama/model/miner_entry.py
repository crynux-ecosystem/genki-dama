from typing import Any, Dict, Optional

from pydantic import BaseModel, Field
import sys

from genki_dama.model.creative_model import CreativeModel


class MinerEntry(BaseModel):
    block: int = Field(default=sys.maxsize, description="The block number")
    hotkey: Optional[str] = Field(default_factory=None, description="The hotkey of the miner")
    invalid: bool = Field(default=False, description="invalidity of determining score")
    creative_model: Optional[CreativeModel] = Field(default_factory=None, description="The model of the miner")
    safetensors_model_size: int = Field(default=0, description="The safetensors model size according to huggingface")
    total_score: float = 0
