from typing import List, Tuple

import bittensor as bt

from taoverse.model.competition import utils as competition_utils
from taoverse.model.competition.data import Competition, ModelConstraints
from taoverse.model.model_updater import MinerMisconfiguredError, ModelUpdater
from taoverse.model.storage.model_metadata_store import ModelMetadataStore
from taoverse.model.model_tracker import ModelTracker
from taoverse.model.utils import get_hash_of_two_strings

from models import DiskAudioModelStore, RemoteAudioModelStore, AudioModel


class AudioModelUpdater(ModelUpdater):
    def __init__(
        self,
        metadata_store: ModelMetadataStore,
        remote_store: RemoteAudioModelStore,
        local_store: DiskAudioModelStore,
        model_tracker: ModelTracker,
    ):
        self.metadata_store = metadata_store
        self.remote_store = remote_store
        self.local_store = local_store
        self.model_tracker = model_tracker

    async def sync_model(
        self,
        hotkey: str,
        curr_block: int,
        schedule_by_block: List[Tuple[int, List[Competition]]],
        force: bool = False,
    ) -> bool:
        """Updates local model for a hotkey if out of sync and returns if it was updated."

        Args:
           hotkey (str): The hotkey of the model to sync.
           curr_block (int): The current block.
           force (bool): Whether to force a sync for this model, even if it's chain metadata hasn't changed.
           schedule_by_block (List[Tuple[int, List[Competition]]]): Which competitions are being run at a given block.
        """
        # Get the metadata for the miner.
        metadata = await self._get_metadata(hotkey)

        if not metadata:
            raise MinerMisconfiguredError(
                hotkey, f"No valid metadata found on the chain"
            )

        # Check that the metadata indicates a competition available at time of upload.
        competition = competition_utils.get_competition_for_block(
            comp_id=metadata.id.competition_id,
            block=metadata.block,
            schedule_by_block=schedule_by_block,
        )
        if not competition:
            raise MinerMisconfiguredError(
                hotkey,
                f"No competition found for {metadata.id.competition_id} at block {metadata.block}",
            )

        # Check that the metadata is old enough to meet the eval_block_delay for the competition.
        # If not we return false and will check again next time we go through the update loop.
        if curr_block - metadata.block < competition.constraints.eval_block_delay:
            bt.logging.debug(
                f"""Sync for hotkey {hotkey} delayed as the current block: {curr_block} is not at least
                {competition.constraints.eval_block_delay} blocks after the upload block: {metadata.block}.
                Will automatically retry later."""
            )
            return False

        # Check what model id the model tracker currently has for this hotkey.
        tracker_model_metadata = self.model_tracker.get_model_metadata_for_miner_hotkey(
            hotkey
        )
        # If we are not forcing a sync due to retrying a top model we can short-circuit if no change.
        if not force and metadata == tracker_model_metadata:
            return False

        # Get the local path based on the local store to download to (top level hotkey path)
        path = self.local_store.get_path(hotkey)

        # Otherwise we need to download the new model based on the metadata.
        model = await self.remote_store.download_model(
            metadata.id, path, competition.constraints
        )

        # Update the tracker even if the model fails the following checks to avoid redownloading without new metadata.
        self.model_tracker.on_miner_model_updated(hotkey, metadata)

        # Check that the hash of the downloaded content matches.
        # This is only useful for SN9's legacy competition before multi-competition support
        # was introduced. Securing hashes was optional. In modern competitions, `hash` is
        # always None, and only `secure_hash` is used.
        if model.id.hash != metadata.id.hash:
            # Check that the hash of the downloaded content matches.
            secure_hash = get_hash_of_two_strings(model.id.hash, hotkey)
            if secure_hash != metadata.id.secure_hash:
                raise MinerMisconfiguredError(
                    hotkey,
                    f"Hash of content downloaded from hugging face does not match chain metadata. {metadata}",
                )

        if not self.verify_model_satisfies_parameters(model, competition.constraints):
            raise MinerMisconfiguredError(
                hotkey,
                f"Model does not satisfy parameters for competition {competition.id}",
            )

        return True

    @staticmethod
    def verify_model_satisfies_parameters(
        model: AudioModel, model_constraints: ModelConstraints
    ) -> bool:
        if not model_constraints:
            bt.logging.trace(f"No competition found for {model.id.competition_id}")
            return False

        # Check that the parameter count of the model is within allowed bounds.
        assert model.model.lm_model_pkg is not None
        assert model.model.compression_model_pkg is not None
        parameter_size = sum(
            v.numel() for v in model.model.lm_model_pkg["best_state"].values()
        ) + sum(
            v.numel() for v in model.model.compression_model_pkg["best_state"].values()
        )

        bt.logging.debug(
            f"Model {model.id.name} parameter size: {parameter_size} / ({model_constraints.min_model_parameter_size}, {model_constraints.max_model_parameter_size})"
        )

        if (
            parameter_size > model_constraints.max_model_parameter_size
            or parameter_size < model_constraints.min_model_parameter_size
        ):
            bt.logging.debug(
                f"Model {model.id.name} does not satisfy constraints for competition {model.id.competition_id}"
            )
            bt.logging.debug(f"Number of model parameters is {parameter_size}")
            bt.logging.debug(
                f"Max parameters allowed is {model_constraints.max_model_parameter_size}"
            )
            bt.logging.debug(
                f"Min parameters allowed is {model_constraints.min_model_parameter_size}"
            )
            return False

        return True
