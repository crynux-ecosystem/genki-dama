import dataclasses
import os
from typing import Any, Dict

import torch
from audiocraft.models.loaders import (load_compression_model_ckpt,
                                       load_lm_model_ckpt)
from huggingface_hub import (snapshot_download, update_repo_visibility,
                             upload_folder)
from taoverse.model.data import ModelId
from taoverse.model.storage.disk import utils



class AudioCraftModel(object):
    def __init__(
        self,
        lm_model_pkg: Dict[str, Any] | None = None,
        compression_model_pkg: Dict[str, Any] | None = None,
        local_dir: str | None = None
    ):
        self.lm_model_pkg = lm_model_pkg
        self.compression_model_pkg = compression_model_pkg

        self.local_dir = local_dir

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        **kwargs,
    ):
        if os.path.isdir(pretrained_model_name_or_path):
            lm_model_pkg = load_lm_model_ckpt(pretrained_model_name_or_path)
            compression_model_pkg = load_compression_model_ckpt(
                pretrained_model_name_or_path
            )

            local_dir = pretrained_model_name_or_path

        elif os.path.isfile(pretrained_model_name_or_path):
            raise ValueError(
                "pretrained_model_name_or_path can only be a local directory or a remote Huggingface model ID"
            )

        else:
            cache_dir = kwargs.get("cache_dir", None)
            revision = kwargs.get("revision", None)
            token = kwargs.get("token", None)

            local_dir = snapshot_download(
                pretrained_model_name_or_path,
                revision=revision,
                cache_dir=cache_dir,
                token=token,
            )

            utils.realize_symlinks_in_directory(local_dir)

            lm_model_pkg = load_lm_model_ckpt(local_dir)
            compression_model_pkg = load_compression_model_ckpt(local_dir)
        return cls(lm_model_pkg, compression_model_pkg, local_dir)

    def save_pretrained(self, save_directory: str):
        assert self.lm_model_pkg is not None
        assert self.compression_model_pkg is not None

        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        torch.save(self.lm_model_pkg, os.path.join(save_directory, "state_dict.bin"))
        torch.save(self.compression_model_pkg, os.path.join(save_directory, "compression_state_dict.bin"))


    def push_to_hub(
        self,
        repo_id: str,
        commit_message: str | None = None,
        private: bool | None = None,
        token: str | None = None,
        create_pr: bool = False,
        safe_serialization: bool = True,
        variant: str | None = None,
    ):
        assert self.lm_model_pkg is not None
        assert self.compression_model_pkg is not None
        assert self.local_dir is not None

        if variant == "fp16":
            lm_state_dict = self.lm_model_pkg["best_state"]
            for k in lm_state_dict:
                lm_state_dict[k] = lm_state_dict[k].to(dtype=torch.float16)
            torch.save(
                self.lm_model_pkg, os.path.join(self.local_dir, "state_dict.bin")
            )
        elif variant == "bf16":
            lm_state_dict = self.lm_model_pkg["best_state"]
            for k in lm_state_dict:
                lm_state_dict[k] = lm_state_dict[k].to(dtype=torch.bfloat16)
            torch.save(
                self.lm_model_pkg, os.path.join(self.local_dir, "state_dict.bin")
            )

        info = upload_folder(
            repo_id=repo_id,
            folder_path=self.local_dir,
            commit_message=commit_message,
            token=token,
            create_pr=create_pr,
            allow_patterns=["state_dict.bin", "compression_state_dict.bin"],
        )

        if private:
            update_repo_visibility(repo_id=repo_id, private=private, token=token)
        return info


@dataclasses.dataclass
class AudioModel:
    class Config:
        arbitrary_types_allowed = True

    # Identifier for this model.
    id: ModelId

    # PreTrainedModel.base_model returns torch.nn.Module if needed.
    model: AudioCraftModel
