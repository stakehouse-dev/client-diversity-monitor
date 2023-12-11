#!/usr/bin/env python3

# This is a sample client for Lighthouse's events endpoint that gathers
# block rewards, sends them to an instance of the classifier API and prints the results
# in real time.

import json
import requests
import sseclient
from apiconfig import APIConfig

cfg = APIConfig()

EVENT_URL = cfg.get_beacon_node_url() + "/eth/v1/events?topics=block_reward"
HEADERS = {"Accept": "text/event-stream",
           "x-api-key": cfg.beacon_node_api_key()}

CLASSIFIER_URL = "http://localhost:7000"


def main():
    res = requests.get(EVENT_URL, stream=True, headers=HEADERS)
    res.raise_for_status()

    client = sseclient.SSEClient(res)

    for event in client.events():
        data = json.loads(event.data)
        slot = data["meta"]["slot"]
        graffiti = data["meta"]["graffiti"]
        att_reward = data["attestation_rewards"]["total"]
        sync_reward = int(data["sync_committee_rewards"])
        total_reward = att_reward + sync_reward
        print(
            f"block at slot {slot} [[{graffiti}]]: {total_reward} gwei ({att_reward} + {sync_reward})"  # noqa: E501
        )
        res = requests.post(
            f"{CLASSIFIER_URL}/classify",
            data=json.dumps(data, ensure_ascii=False).encode("utf-8"),
        )
        res.raise_for_status()

        classification = res.json()
        print(classification)


if __name__ == "__main__":
    main()
