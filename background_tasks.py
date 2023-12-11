#!/usr/bin/env python3

# Tasks that are intended to run alongside the API server to keep it up to date.
import time
import json
import requests
import sseclient
import multiprocessing
from apiconfig import APIConfig
import atexit

from api_client import post_block_rewards, get_sync_gaps
from load_blocks import download_block_rewards

cfg = APIConfig()

BN_URL = cfg.get_beacon_node_url()
BLOCKPRINT_URL = "http://localhost:7000"

EVENT_URL_PATH = "eth/v1/events?topics=block_reward"
HEADERS = {"Accept": "text/event-stream",
           "x-api-key": cfg.beacon_node_api_key()}

BACKFILL_WAIT_SECONDS = 60
FAIL_WAIT_SECONDS = 5


class BlockRewardListener:
    def __init__(self, bn_url, blockprint_url):
        self.bn_url = bn_url
        self.blockprint_url = blockprint_url

    def run(self):
        while True:
            try:
                event_url = f"{self.bn_url}/{EVENT_URL_PATH}"
                res = requests.get(event_url, stream=True, headers=HEADERS)
                res.raise_for_status()

                client = sseclient.SSEClient(res)

                for event in client.events():
                    block_reward = json.loads(event.data)
                    slot = int(block_reward["meta"]["slot"])
                    print(f"Classifying block {slot}")
                    post_block_rewards(self.blockprint_url, [block_reward])

            except Exception as e:
                print(f"Block listener failed with: {e}")
                time.sleep(FAIL_WAIT_SECONDS)


def explode_gap(start_slot, end_slot, sprp):
    next_boundary = (start_slot // sprp + 1) * sprp

    if end_slot > next_boundary:
        return [(start_slot, next_boundary)] + explode_gap(
            next_boundary + 1, end_slot, sprp
        )
    else:
        return [(start_slot, end_slot)]


def explode_gaps(gaps, sprp=300):
    """
    Divide sync gaps into manageable chunks aligned to Lighthouse's restore points
    Also for memory efficiency, we don't want to load the entire gap into memory
    """

    exploded = []

    new_gaps = []
    # For each gap, add their start and end slots to the list of gaps
    for gap in gaps:
        for slot in range(int(gap["start"]), int(gap["end"]) + 1, sprp):
            new_gaps.append({"start": slot, "end": slot + sprp - 1})

    gaps = new_gaps

    for gap in gaps:
        start_slot = int(gap["start"])
        end_slot = int(gap["end"])

        exploded.extend(explode_gap(start_slot, end_slot, sprp))

    return exploded


class Backfiller:
    def __init__(self, bn_url, blockprint_url):
        self.bn_url = bn_url
        self.blockprint_url = blockprint_url

    def run(self):
        while True:
            try:
                sync_gaps = get_sync_gaps(self.blockprint_url)

                chunks = explode_gaps(sync_gaps)

                # reverse the chunks so we fill recent gaps first
                chunks.reverse()

                for (start_slot, end_slot) in chunks:
                    print(f"Downloading backfill blocks {start_slot}..={end_slot}")
                    try:
                        block_rewards = download_block_rewards(
                            start_slot, end_slot, beacon_node=self.bn_url
                        )

                        block_rewards.reverse()


                        print(f"Classifying backfill blocks {start_slot}..={end_slot}")
                        post_block_rewards(self.blockprint_url, block_rewards)
                    except Exception as e:
                        print(f"Failed to download blocks: {e}")

                        for slot in range(end_slot, start_slot - 1, -1):
                            try:
                                block_rewards = download_block_rewards(
                                    slot, slot, beacon_node=self.bn_url
                                )
                                print(f"Classifying backfill block {slot}")
                                post_block_rewards(self.blockprint_url, block_rewards)
                            except Exception as e:
                                print(f"Failed to download block: {e}")
                                continue

                        continue

                if len(chunks) == 0:
                    print("Blockprint is synced")
                    time.sleep(BACKFILL_WAIT_SECONDS)

            except Exception as e:
                print(f"Backfiller failed with: {e}")
                time.sleep(FAIL_WAIT_SECONDS)


if __name__ == "__main__":
    listener_task = lambda: BlockRewardListener(BN_URL, BLOCKPRINT_URL).run()
    listener = multiprocessing.Process(target=listener_task, name="block_listener")
    listener.start()

    backfill_task = lambda: Backfiller(BN_URL, BLOCKPRINT_URL).run()
    backfiller = multiprocessing.Process(target=backfill_task, name="backfiller")
    backfiller.start()

    atexit.register(lambda: listener.terminate())
    atexit.register(lambda: backfiller.terminate())

    listener.join()
    backfiller.join()

    print("Exiting")
