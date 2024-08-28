# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# TODO(developer): Set your name
# Copyright © 2023 <your name>

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import time
from typing import Optional
import bittensor as bt

from genki_dama.model.creative_model import CreativeModel
from genki_dama.model.miner_entry import MinerEntry
from genki_dama.protocol import Dummy
from genki_dama.validator.reward import get_rewards
from genki_dama.utils.uids import get_random_uids

def get_miner_entry(self, hotkey: str) -> Optional[MinerEntry]:
    try:
        metadata = bt.extrinsics.serving.get_metadata(self.subtensor, self.config.netuid, hotkey)
        if metadata is None:
            return None
        commitment = metadata["info"]["fields"][0]
        hex_data = commitment[list(commitment.keys())[0]][2:]
        chain_str = bytes.fromhex(hex_data).decode()
        creative_model = CreativeModel.from_compressed_str(chain_str)
        block = metadata["block"]
        entry = MinerEntry(block=block, hotkey=hotkey, creative_model=creative_model)
        return entry
    except Exception as e:
        bt.logging.error(f"could not fetch data for {hotkey} : {e}")
        return None


async def forward(self):
    # TODO(developer): Define how the validator selects a miner to query, how often, etc.
    # get_random_uids is an example method, but you can replace it with your own.
    all_uids = self.metagraph.uids

    stake = self.metagraph.S
    weights = self.metagraph.W
    dividends = self.metagraph.D

    bt.logging.info(f"stake: {stake}, weights: {weights}, dividends: {dividends}")

    # Log the results for monitoring purposes.
    bt.logging.info(f"All miners: {all_uids}")

    # Get all the submitted models from the miners.
    for uid in all_uids:
        miner_entry = get_miner_entry(self, self.metagraph.hotkeys[uid])
        if miner_entry is not None:
            bt.logging.debug(f"Miner hotkey: {miner_entry.hotkey}")
            bt.logging.debug(f"Miner model: {miner_entry.creative_model.namespace} / {miner_entry.creative_model.name}")
            bt.logging.debug(f"Submitted block: {miner_entry.block}")
        else:
            bt.logging.debug(f"No model found for hotkey: {self.metagraph.hotkeys[uid]}")

    # bt.logging.info(bt.metagraph.axons)

    # for uid in all_uids:
    #     hot_key = bt.metagraph.hotkeys[uid]
    #     bt.logging.info(f"{uid}: {hot_key}")

    # TODO(developer): Define how the validator scores responses.
    # Adjust the scores based on responses from miners.
    rewards = [10 if miner_uid == 2 else 1 for miner_uid in all_uids]
    bt.logging.info(f"Scored responses: {rewards}")

    # Update the scores based on the rewards. You may want to define your own update_scores function for custom behavior.
    self.update_scores(rewards, all_uids)
    # self.set_weights()
    time.sleep(30)
