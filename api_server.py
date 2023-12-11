import os
import json
import falcon
from falcon.http_status import HTTPStatus
from apiconfig import APIConfig

from multi_classifier import MultiClassifier
from build_db import (
    open_block_db,
    get_blocks_per_client,
    get_sync_status,
    get_sync_gaps,
    update_block_db,
    get_validator_blocks,
    get_all_validators_latest_blocks,
    get_blocks,
    get_lsdblocks,
    get_client_diversity_blocks,
    get_client_diversity_blocks_lsd,
    get_client_diversity_validators,
    get_client_diversity_validators_lsd,
    get_client_diversity_blocks_for_lsd,
    get_client_diversity_validators_for_lsd,
    get_client_diversity_validators_lsd_overview,
    count_true_positives,
    count_true_negatives,
    count_false_positives,
    count_false_negatives,
)

cfg = APIConfig()

DATA_DIR = "./data"
BLOCK_DB = os.environ.get("BLOCK_DB") or "./block_db.sqlite"
BN_URL = cfg.get_beacon_node_url()
SELF_URL = "http://localhost:7000"
DISABLE_CLASSIFIER = "DISABLE_CLASSIFIER" in os.environ


class Classify:
    def __init__(self, classifier, block_db):
        self.classifier = classifier
        self.block_db = block_db

    def on_post(self, req, resp):
        try:
            block_rewards = json.load(req.bounded_stream)
        except json.decoder.JSONDecodeError as e:
            resp.text = json.dumps({"error": f"invalid JSON: {e}"})
            resp.code = falcon.HTTP_400
            return

        if not check_block_rewards_ok(block_rewards, resp):
            return

        update_block_db(self.block_db, self.classifier, block_rewards)
        print(
            f"Processed {len(block_rewards)} block{'' if block_rewards == [] else 's'}"
        )
        resp.text = "OK"


class ClassifyNoStore:
    def __init__(self, classifier):
        self.classifier = classifier

    def on_post(self, req, resp):
        try:
            block_rewards = json.load(req.bounded_stream)
        except json.decoder.JSONDecodeError as e:
            resp.text = json.dumps({"error": f"invalid JSON: {e}"})
            resp.code = falcon.HTTP_400
            return

        if not check_block_rewards_ok(block_rewards, resp):
            return

        classifications = []
        for block_reward in block_rewards:
            label, _, _, _ = classifier.classify(block_reward)
            classifications.append(
                {
                    "best_guess_single": label,
                }
            )
        resp.text = json.dumps(classifications, ensure_ascii=False)


def check_block_rewards_ok(block_rewards, resp):
    # Check required fields
    for block_reward in block_rewards:
        if (
            "block_root" not in block_reward
            or "attestation_rewards" not in block_reward
            or "per_attestation_rewards" not in block_reward["attestation_rewards"]
        ):
            resp.text = json.dumps({"error": "input JSON is not a block reward"})
            resp.code = falcon.HTTP_400
            return False
    return True


class BlocksPerClient:
    def __init__(self, block_db):
        self.block_db = block_db

    def on_get(self, req, resp, start_epoch, end_epoch=None):
        end_epoch = end_epoch or (start_epoch + 1)

        start_slot = 32 * start_epoch
        end_slot = 32 * end_epoch
        blocks_per_client = get_blocks_per_client(self.block_db, start_slot, end_slot)
        resp.text = json.dumps(blocks_per_client, ensure_ascii=False)


class SyncStatus:
    def __init__(self, block_db):
        self.block_db = block_db

    def on_get(self, req, resp):
        sync_status = get_sync_status(self.block_db)
        resp.text = json.dumps(sync_status, ensure_ascii=False)


class SyncGaps:
    def __init__(self, block_db):
        self.block_db = block_db

    def on_get(self, req, resp):
        gaps = get_sync_gaps(self.block_db)
        resp.text = json.dumps(gaps, ensure_ascii=False)


class ValidatorBlocks:
    def __init__(self, block_db):
        self.block_db = block_db

    def on_get(self, req, resp, validator_index, since_slot=None):
        validator_blocks = get_validator_blocks(
            self.block_db, validator_index, since_slot
        )
        resp.text = json.dumps(validator_blocks, ensure_ascii=False)


class MultipleValidatorsBlocks:
    def __init__(self, block_db):
        self.block_db = block_db

    def on_post(self, req, resp, since_slot=None):
        # Validate request.
        try:
            validator_indices = json.load(req.bounded_stream)
        except json.decoder.JSONDecodeError as e:
            resp.text = json.dumps({"error": f"invalid JSON: {e}"})
            resp.code = falcon.HTTP_400
            return

        if type(validator_indices) != list or any(
            type(x) != int for x in validator_indices
        ):
            resp.text = json.dumps({"error": "request must be a list of integers"})
            resp.code = falcon.HTTP_400
            return

        all_blocks = {}
        for validator_index in validator_indices:
            validator_blocks = get_validator_blocks(
                self.block_db, validator_index, since_slot
            )
            all_blocks[validator_index] = validator_blocks

        resp.text = json.dumps(all_blocks, ensure_ascii=False)


class AllValidatorsLatestBlocks:
    def __init__(self, block_db):
        self.block_db = block_db

    def on_get(self, req, resp):
        result = get_all_validators_latest_blocks(self.block_db)
        resp.text = json.dumps(result, ensure_ascii=False)


class Blocks:
    def __init__(self, block_db):
        self.block_db = block_db

    def on_get(self, req, resp, start_slot=0, end_slot=None):
        # Would get all blocks monitored
        blocks = get_blocks(self.block_db, start_slot, end_slot)
        resp.text = json.dumps(blocks, ensure_ascii=False)


class LSDBlocks:
    def __init__(self, block_db):
        self.block_db = block_db

    def on_get(self, req, resp, start_slot=0, end_slot=None):
        # Would get all blocks that were proposed by a validator in an LSD network
        blocks = get_lsdblocks(self.block_db, start_slot, end_slot)
        resp.text = json.dumps(blocks, ensure_ascii=False)


class BlockDiversity:
    def __init__(self, block_db):
        self.block_db = block_db

    def on_get(self, req, resp, start_slot=0, end_slot=None):
        # Would get the client diversity of all blocks monitored
        # i.e the percentage of blocks proposed by each client
        blocks = get_client_diversity_blocks(self.block_db, start_slot, end_slot)
        resp.text = json.dumps(blocks, ensure_ascii=False)


class ValidatorDiversity:
    def __init__(self, block_db):
        self.block_db = block_db

    def on_get(self, req, resp, start_slot=0, end_slot=None):
        # Would get the client diversity of all known validators
        # i.e the percentage of validators in the network (monitored)
        # that run each type of client
        blocks = get_client_diversity_validators(self.block_db, start_slot, end_slot)
        resp.text = json.dumps(blocks, ensure_ascii=False)


class AllLSDBlockDiversity:
    def __init__(self, block_db):
        self.block_db = block_db

    def on_get(self, req, resp, start_slot=0, end_slot=None):
        # Would get the client diversity of all lsd blocks
        # i.e the percentage of lsd blocks proposed by each client
        # This is grouped by each lsd network
        blocks = get_client_diversity_blocks_lsd(self.block_db, start_slot, end_slot)
        resp.text = json.dumps(blocks, ensure_ascii=False)


class AllLSDValidatorDiversity:
    def __init__(self, block_db):
        self.block_db = block_db

    def on_get(self, req, resp, start_slot=0, end_slot=None):
        # Would get the client diversity of all known validators in lsd networks
        # i.e the percentage of validators in each lsd network (monitored)
        # that run each type of client
        blocks = get_client_diversity_validators_lsd(self.block_db, start_slot, end_slot)
        resp.text = json.dumps(blocks, ensure_ascii=False)


class LSDValidatorDiversityOverview:
    def __init__(self, block_db):
        self.block_db = block_db

    def on_get(self, req, resp, start_slot=0, end_slot=None):
        # Would get the client diversity of all known validators in lsd networks
        # i.e the percentage of validators in each lsd network (monitored)
        # that run each type of client
        blocks = get_client_diversity_validators_lsd_overview(self.block_db, start_slot, end_slot)
        resp.text = json.dumps(blocks, ensure_ascii=False)


class LSDBlockDiversity:
    def __init__(self, block_db):
        self.block_db = block_db

    def on_get(self, req, resp, lsd_id, start_slot=0, end_slot=None):
        # Would get the client diversity of all blocks monitored in a given lsd network
        # i.e the percentage of blocks proposed by each client in that specific lsd network
        blocks = get_client_diversity_blocks_for_lsd(self.block_db, lsd_id, start_slot, end_slot)
        resp.text = json.dumps(blocks, ensure_ascii=False)


class LSDValidatorDiversity:
    def __init__(self, block_db):
        self.block_db = block_db

    def on_get(self, req, resp, lsd_id, start_slot=0, end_slot=None):
        # Would get the client diversity of all known validators in a given lsd network
        # i.e the percentage of validators in the given lsd network (monitored)
        # that run each type of client
        blocks = get_client_diversity_validators_for_lsd(
            self.block_db, lsd_id, start_slot, end_slot)
        resp.text = json.dumps(blocks, ensure_ascii=False)


class ConfusionMatrix:
    def __init__(self, block_db):
        self.block_db = block_db

    def on_get(self, req, resp, client, start_slot, end_slot=None):
        true_pos = count_true_positives(self.block_db, client, start_slot, end_slot)
        true_neg = count_true_negatives(self.block_db, client, start_slot, end_slot)
        false_pos = count_false_positives(self.block_db, client, start_slot, end_slot)
        false_neg = count_false_negatives(self.block_db, client, start_slot, end_slot)
        resp.text = json.dumps(
            {
                "true_pos": true_pos,
                "true_neg": true_neg,
                "false_pos": false_pos,
                "false_neg": false_neg,
            }
        )

app = application = falcon.App(middleware=
    falcon.CORSMiddleware(allow_origins='*', allow_credentials='*')
)

classifier = None
if not DISABLE_CLASSIFIER:
    print("Initialising classifier, this could take a moment...")
    classifier = MultiClassifier(DATA_DIR) if not DISABLE_CLASSIFIER else None
    print("Done")

block_db = open_block_db(BLOCK_DB)

app.add_route("/classify/no_store", ClassifyNoStore(classifier))
app.add_route("/classify", Classify(classifier, block_db))
app.add_route("/sync/status", SyncStatus(block_db))
app.add_route("/sync/gaps", SyncGaps(block_db))

app.add_route(
    "/blocks_per_client/{start_epoch:int}/{end_epoch:int}", BlocksPerClient(block_db)
)

app.add_route("/blocks_per_client/{start_epoch:int}", BlocksPerClient(block_db))

app.add_route("/validator/{validator_index:int}/blocks", ValidatorBlocks(block_db))

app.add_route(
    "/validator/{validator_index:int}/blocks/{since_slot:int}",
    ValidatorBlocks(block_db),
)
app.add_route("/validator/blocks", MultipleValidatorsBlocks(block_db))
app.add_route("/validator/blocks/{since_slot:int}", MultipleValidatorsBlocks(block_db))
app.add_route("/validator/blocks/latest", AllValidatorsLatestBlocks(block_db))

app.add_route(
    "/confusion/{client}/{start_slot:int}/{end_slot:int}", ConfusionMatrix(block_db)
)

app.add_route("/blocks", Blocks(block_db))
app.add_route("/blocks/{start_slot:int}", Blocks(block_db))
app.add_route("/blocks/{start_slot:int}/{end_slot:int}", Blocks(block_db))

app.add_route("/blocks/lsd", LSDBlocks(block_db))
app.add_route("/blocks/lsd/{start_slot:int}", LSDBlocks(block_db))
app.add_route("/blocks/lsd/{start_slot:int}/{end_slot:int}", LSDBlocks(block_db))

app.add_route("/client_diversity/blocks", BlockDiversity(block_db))
app.add_route("/client_diversity/blocks/{start_slot:int}", BlockDiversity(block_db))
app.add_route("/client_diversity/blocks/{start_slot:int}/{end_slot:int}", BlockDiversity(block_db))

app.add_route("/client_diversity/validators", ValidatorDiversity(block_db))
app.add_route("/client_diversity/validators/{start_slot:int}", ValidatorDiversity(block_db))
app.add_route(
    "/client_diversity/validators/{start_slot:int}/{end_slot:int}", ValidatorDiversity(block_db))


app.add_route("/client_diversity/lsd/blocks", AllLSDBlockDiversity(block_db))
app.add_route("/client_diversity/lsd/blocks/{start_slot:int}", AllLSDBlockDiversity(block_db))
app.add_route(
    "/client_diversity/lsd/blocks/{start_slot:int}/{end_slot:int}", AllLSDBlockDiversity(block_db))

app.add_route("/client_diversity/lsd/validators", AllLSDValidatorDiversity(block_db))
app.add_route(
    "/client_diversity/lsd/validators/{start_slot:int}", AllLSDValidatorDiversity(block_db))
app.add_route(
    "/client_diversity/lsd/validators/{start_slot:int}/{end_slot:int}", AllLSDValidatorDiversity(block_db))

app.add_route("/client_diversity/lsd/validators/overview", LSDValidatorDiversityOverview(block_db))
app.add_route(
    "/client_diversity/lsd/validators/overview/{start_slot:int}", LSDValidatorDiversityOverview(block_db))
app.add_route(
    "/client_diversity/lsd/validators/overview/{start_slot:int}/{end_slot:int}", LSDValidatorDiversityOverview(block_db))

app.add_route("/client_diversity/lsd/{lsd_id}/blocks", LSDBlockDiversity(block_db))
app.add_route(
    "/client_diversity/lsd/{lsd_id}/blocks/{start_slot:int}", LSDBlockDiversity(block_db))
app.add_route(
    "/client_diversity/lsd/{lsd_id}/blocks/{start_slot:int}/{end_slot:int}", LSDBlockDiversity(block_db))

app.add_route("/client_diversity/lsd/{lsd_id}/validators", LSDValidatorDiversity(block_db))
app.add_route(
    "/client_diversity/lsd/{lsd_id}/validators/{start_slot:int}", LSDValidatorDiversity(block_db))
app.add_route(
    "/client_diversity/lsd/{lsd_id}/validators/{start_slot:int}/{end_slot:int}", LSDValidatorDiversity(block_db))

print("Up")
