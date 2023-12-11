#!/usr/bin/env python3

import os
import json
import sqlite3
import argparse
from knn_classifier import Classifier
from multi_classifier import MultiClassifier
from lsd import LSD
from prepare_training_data import CLIENTS

DB_CLIENTS = [client for client in CLIENTS if client != "Other"]


def list_all_files(classify_dir):
    for root, _, files in os.walk(classify_dir):
        for filename in files:
            yield os.path.join(root, filename)


def create_block_db(db_path):
    if os.path.exists(db_path):
        print("deleting existing database")
        os.remove(db_path)

    conn = sqlite3.connect(db_path)

    conn.execute(
        """CREATE TABLE blocks (
            slot INT,
            parent_slot INTEGER,
            proposer_index INT,
            best_guess_single TEXT,
            best_guess_multi TEXT,
            pr_lighthouse FLOAT,
            pr_lodestar FLOAT,
            pr_nimbus FLOAT,
            pr_prysm FLOAT,
            pr_teku FLOAT,
            graffiti_guess TEXT,
            lsd_id TEXT,
            lsd_index INT,
            lsd_ticker TEXT,
            UNIQUE(slot, proposer_index)
        )
        """
    )

    conn.execute("CREATE INDEX block_proposers ON blocks (proposer_index)")
    conn.execute("CREATE INDEX block_slots ON blocks (slot)")

    return conn


def open_block_db(db_path):
    if not os.path.exists(db_path):
        raise Exception(f"no database found at {db_path}")

    return sqlite3.connect(db_path)


def open_or_create_db(db_path, force_create=False):
    if os.path.exists(db_path) and not force_create:
        return open_block_db(db_path)
    else:
        return create_block_db(db_path)


def slot_range_from_filename(filename) -> (int, int):
    parts = os.path.splitext(os.path.basename(filename))[0].split("_")
    start_slot = int(parts[1])
    end_slot = int(parts[3])
    return (start_slot, end_slot)


def build_block_db(db_path, classifier, classify_dir, force_rebuild=False):
    conn = open_or_create_db(db_path, force_create=force_rebuild)

    for input_file in list_all_files(classify_dir):
        start_slot, end_slot = slot_range_from_filename(input_file)

        if slot_range_known_to_db(conn, start_slot, end_slot):
            print(f"skipping {input_file} (assumed known)")
            continue

        print(f"classifying rewards from file {input_file}")
        with open(input_file, "r") as f:
            block_rewards = json.load(f)

        update_block_db(conn, classifier, block_rewards)

    return conn


def update_block_db(conn, classifier, block_rewards):
    for block_reward in block_rewards:
        label, multilabel, prob_by_client, graffiti_guess = classifier.classify(
            block_reward
        )

        proposer_index = block_reward["meta"]["proposer_index"]
        slot = int(block_reward["meta"]["slot"])
        parent_slot = int(block_reward["meta"]["parent_slot"])

        lsd = LSD()
        lsd_id = None
        lsd_index = None
        lsd_ticker = None
        try:
            validator_info = lsd.beacon_get_validator(slot, proposer_index)
            print(
                f"\nGetting validator info for slot {slot} (epoch {slot//32}) proposer {proposer_index}")
            validator_lsd = lsd.get_validator_lsd_at_epoch(
                validator_info['pubkey'], slot // 32)
            if validator_lsd is not None:
                lsd_id = validator_lsd['lsd_id']
                lsd_index = validator_lsd['lsd_index']
                lsd_ticker = validator_lsd['lsd_ticker']
                print(
                    f"Found LSD info for slot {slot} proposer {proposer_index}: {lsd_id}, {lsd_index}, {lsd_ticker}")
        except Exception as e:
            print(
                f"Failed to get validator info for slot {slot} proposer {proposer_index}: {e}")

        print("\nClassified block reward:")
        print(f"  slot: {slot}")
        print(f"  parent_slot: {parent_slot}")
        print(f"  proposer_index: {proposer_index}")
        print(f"  label: {label}")
        print(f"  multilabel: {multilabel}")
        print(f"  prob_by_client: {prob_by_client}")
        print(f"  graffiti_guess: {graffiti_guess}")
        print(f"  lsd_id: {lsd_id}")
        print(f"  lsd_index: {lsd_index}")
        print(f"  lsd_ticker: {lsd_ticker}")

        insert_block(
            conn,
            slot,
            parent_slot,
            proposer_index,
            label,
            multilabel,
            prob_by_client,
            graffiti_guess,
            lsd_id,
            lsd_index,
            lsd_ticker,
        )

    conn.commit()


def insert_block(
    conn,
    slot,
    parent_slot,
    proposer_index,
    label,
    multilabel,
    prob_by_client,
    graffiti_guess,
    lsd_id=None,
    lsd_index=None,
    lsd_ticker=None,
):
    pr_clients = [prob_by_client.get(client) or 0.0 for client in DB_CLIENTS]

    try:
        conn.execute(
            """INSERT INTO blocks (slot, parent_slot, proposer_index, best_guess_single,
                                best_guess_multi, pr_lighthouse, pr_lodestar, pr_nimbus,
                                pr_prysm, pr_teku, graffiti_guess, lsd_id, lsd_index, lsd_ticker)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                slot,
                parent_slot,
                proposer_index,
                label,
                multilabel,
                *pr_clients,
                graffiti_guess,
                lsd_id,
                lsd_index,
                lsd_ticker,
            ),
        )
    except sqlite3.IntegrityError:
        # This is fine, as we may be reprocessing a file that we've already processed
        # before.
        pass
    except sqlite3.OperationalError:
        # This may present an sqlite database is locked error if multiple processes are
        # writing to the database at the same time. This is fine, as there is a backfill
        # process that will fill in the missing data after the backfill wait period (60s).
        pass


def get_greatest_block_slot(block_db):
    res = list(block_db.execute("SELECT MAX(slot) FROM blocks"))
    assert len(res) == 1

    slot = res[0][0]
    if slot is None:
        return 0
    else:
        return int(slot)


def get_missing_parent_blocks(block_db):
    res = block_db.execute(
        """SELECT slot, parent_slot FROM blocks b1
           WHERE
             (SELECT slot FROM blocks WHERE slot = b1.parent_slot) IS NULL
             AND slot > 3900000"""
    )
    result = []
    return [(int(x[0]), int(x[1])) for x in res]


def get_greatest_prior_block_slot(block_db, slot):
    res = list(block_db.execute("SELECT MAX(slot) FROM blocks WHERE slot < ?", (slot,)))
    assert len(res) == 1

    slot = res[0][0]
    if slot is None:
        return None
    else:
        return int(slot)


def get_sync_gaps(block_db):
    missing_parent_slots = get_missing_parent_blocks(block_db)
    gaps = []

    for (block_slot, parent_slot) in missing_parent_slots:
        prior_slot = get_greatest_prior_block_slot(block_db, parent_slot)

        if prior_slot is None:
            start_slot = 3900000
        else:
            start_slot = prior_slot + 1
        end_slot = block_slot - 1

        assert end_slot >= start_slot

        gaps.append({"start": start_slot, "end": end_slot})
    return gaps


def slot_range_known_to_db(block_db, start_slot, end_slot):
    res = list(
        block_db.execute(
            "SELECT COUNT(*) FROM blocks WHERE slot >= ? AND slot <= ?",
            (start_slot, end_slot),
        )
    )
    assert len(res) == 1
    count = int(res[0][0])
    return count > 0


def get_sync_status(block_db):
    greatest_block_slot = get_greatest_block_slot(block_db)
    synced = len(get_missing_parent_blocks(block_db)) == 0
    return {"greatest_block_slot": greatest_block_slot, "synced": synced}


def get_blocks_per_client(block_db, start_slot, end_slot):
    blocks_per_client = {client: 0 for client in ["Uncertain", *CLIENTS]}

    client_counts = block_db.execute(
        """SELECT best_guess_single, COUNT(proposer_index)
           FROM blocks
           WHERE slot >= ? AND slot < ?
           GROUP BY best_guess_single""",
        (start_slot, end_slot),
    )

    for (client, count) in client_counts:
        blocks_per_client[client] = int(count)

    return blocks_per_client


def get_validator_blocks(block_db, validator_index, since_slot=None):
    since_slot = since_slot or 0
    rows = block_db.execute(
        """SELECT slot, best_guess_single, best_guess_multi, pr_lighthouse, pr_lodestar,
                  pr_nimbus, pr_prysm, pr_teku, lsd_id, lsd_index, lsd_ticker
           FROM blocks WHERE proposer_index = ? AND slot >= ?""",
        (validator_index, since_slot),
    )

    def row_to_json(row):
        slot = row[0]
        best_guess_single = row[1]
        best_guess_multi = row[2]
        probability_map = {client: row[3 + i] for i, client in enumerate(DB_CLIENTS)}
        lsd_id = row[9]
        lsd_index = row[10]
        lsd_ticker = row[11]

        return {
            "slot": slot,
            "best_guess_single": best_guess_single,
            "best_guess_multi": best_guess_multi,
            "probability_map": probability_map,
            "lsd_id": lsd_id,
            "lsd_index": lsd_index,
            "lsd_ticker": lsd_ticker,
        }

    return [row_to_json(row) for row in rows]


def get_all_validators_latest_blocks(block_db):
    rows = block_db.execute(
        """SELECT b1.proposer_index, b1.slot, b1.best_guess_single, b1.lsd_id, b1.lsd_index, b1.lsd_ticker
           FROM blocks b1
           JOIN (SELECT MAX(slot) AS slot, proposer_index FROM blocks GROUP BY proposer_index)
           AS b2 ON b1.slot = b2.slot AND b1.proposer_index = b2.proposer_index;"""
    )

    def row_to_json(row):
        proposer_index = int(row[0])
        slot = row[1]
        best_guess_single = row[2]
        lsd_id = row[3]
        lsd_index = row[4]
        lsd_ticker = row[5]

        return {
            "proposer_index": proposer_index,
            "slot": slot,
            "best_guess_single": best_guess_single,
            "lsd_id": lsd_id,
            "lsd_index": lsd_index,
            "lsd_ticker": lsd_ticker,
        }

    return [row_to_json(row) for row in rows]


def get_blocks(block_db, start_slot, end_slot=None):
    end_slot = end_slot or (1 << 62)

    rows = block_db.execute(
        """SELECT slot, proposer_index, best_guess_single, best_guess_multi, pr_lighthouse,
           pr_lodestar, pr_nimbus, pr_prysm, pr_teku, lsd_id, lsd_index, lsd_ticker
           FROM blocks WHERE slot >= ? AND slot < ?""",
        (start_slot, end_slot),
    )

    def row_to_json(row):
        slot = row[0]
        proposer_index = int(row[1])
        best_guess_single = row[2]
        best_guess_multi = row[3]
        probability_map = {client: row[4 + i] for i, client in enumerate(DB_CLIENTS)}
        lsd_id = row[9]
        lsd_index = row[10]
        lsd_ticker = row[11]

        return {
            "slot": slot,
            "proposer_index": proposer_index,
            "best_guess_single": best_guess_single,
            "best_guess_multi": best_guess_multi,
            "probability_map": probability_map,
            "lsd_id": lsd_id,
            "lsd_index": lsd_index,
            "lsd_ticker": lsd_ticker,
        }

    return [row_to_json(row) for row in rows]


def get_lsdblocks(block_db, start_slot, end_slot=None):
    end_slot = end_slot or (1 << 62)

    # Get blocks where lsd_id is not null

    rows = block_db.execute(
        """SELECT slot, proposer_index, best_guess_single, best_guess_multi, pr_lighthouse,
           pr_lodestar, pr_nimbus, pr_prysm, pr_teku, lsd_id, lsd_index, lsd_ticker
           FROM blocks WHERE slot >= ? AND slot < ? AND lsd_id IS NOT NULL""",
        (start_slot, end_slot),
    )

    def row_to_json(row):
        slot = row[0]
        proposer_index = int(row[1])
        best_guess_single = row[2]
        best_guess_multi = row[3]
        probability_map = {client: row[4 + i] for i, client in enumerate(DB_CLIENTS)}
        lsd_id = row[9]
        lsd_index = row[10]
        lsd_ticker = row[11]

        return {
            "slot": slot,
            "proposer_index": proposer_index,
            "best_guess_single": best_guess_single,
            "best_guess_multi": best_guess_multi,
            "probability_map": probability_map,
            "lsd_id": lsd_id,
            "lsd_index": lsd_index,
            "lsd_ticker": lsd_ticker,
        }

    return [row_to_json(row) for row in rows]


def get_client_diversity_blocks(block_db, start_slot, end_slot=None):
    # Gets the percentage of blocks that are by each client
    # Data needs to be grouped by client best guess
    # Returns a list of dictionaries with the client name and percentage
    # The percentage is over the total number of blocks in the range

    end_slot = end_slot or (1 << 62)

    rows = block_db.execute(
        """SELECT best_guess_single, COUNT(*) FROM blocks
              WHERE slot >= ? AND slot < ?
              GROUP BY best_guess_single""",
        (start_slot, end_slot),
    )

    total_blocks = block_db.execute(
        """SELECT COUNT(*) FROM blocks
                WHERE slot >= ? AND slot < ?""",
        (start_slot, end_slot),
    )

    total_blocks = int(list(total_blocks)[0][0])

    def row_to_json(row):
        client = row[0]
        count = row[1]

        if total_blocks == 0:
            percentage = 0
        else:
            percentage = count / total_blocks

        return {
            "client": client,
            "percentage": percentage,
        }

    return [row_to_json(row) for row in rows]


def get_client_diversity_blocks_lsd(block_db, start_slot, end_slot=None):
    # Gets the percentage of blocks that are by each client for each lsd
    # Data grouped first by lsd_id and then for each lsd,
    # grouped by client best guess
    # Returns a list of dictionaries with the client name and percentage
    # The percentage is over the total number of blocks in the range

    end_slot = end_slot or (1 << 62)

    rows = block_db.execute(
        """SELECT lsd_id, lsd_ticker, lsd_index, best_guess_single, COUNT(*) FROM blocks
                WHERE slot >= ? AND slot < ?
                GROUP BY lsd_id, best_guess_single""",
        (start_slot, end_slot),
    )

    def row_to_json(row):
        lsd_id = row[0]
        lsd_ticker = row[1]
        lsd_index = row[2]
        client = row[3]
        count = row[4]

        if lsd_id is None:
            total_blocks = block_db.execute(
                """SELECT COUNT(*) FROM blocks
                    WHERE slot >= ? AND slot < ? AND lsd_id IS NULL""",
                (start_slot, end_slot),
            )
        else:
            total_blocks = block_db.execute(
                """SELECT COUNT(*) FROM blocks
                    WHERE slot >= ? AND slot < ? AND lsd_id = ?""",
                (start_slot, end_slot, lsd_id),
            )

        total_blocks = int(list(total_blocks)[0][0])

        if total_blocks == 0:
            percentage = 0
        else:
            percentage = count / total_blocks

        return {
            "lsd_id": lsd_id,
            "lsd_ticker": lsd_ticker,
            "lsd_index": lsd_index,
            "client": client,
            "percentage": percentage,
        }

    def group_to_lsd(data):
        # Take the results from row to json and group them by lsd_id
        # Returns a list of dictionaries with the lsd_id and a list of clients

        result = {}
        for row in data:
            lsd_id = row["lsd_id"]
            if lsd_id not in result:
                result[lsd_id] = {"lsd_id": lsd_id, "lsd_ticker": row["lsd_ticker"],
                                  "lsd_index": row["lsd_index"], "clients": []}
            result[lsd_id]["clients"].append(
                {"client": row["client"], "percentage": row["percentage"]})

        return list(result.values())

    return group_to_lsd([row_to_json(row) for row in rows])


def get_client_diversity_validators(block_db, start_slot, end_slot=None):
    # Gets the percentage of validators that are by each client
    # Data needs to be grouped by client best guess
    # Returns a list of dictionaries with the client name and percentage
    # The percentage is over the total number of validators in the range

    end_slot = end_slot or (1 << 62)

    rows = block_db.execute(
        """SELECT best_guess_single, COUNT(DISTINCT proposer_index) FROM blocks
              WHERE slot >= ? AND slot < ?
              GROUP BY best_guess_single""",
        (start_slot, end_slot),
    )

    total_validators = block_db.execute(
        """SELECT COUNT(DISTINCT proposer_index) FROM blocks
                WHERE slot >= ? AND slot < ?""",
        (start_slot, end_slot),
    )

    total_validators = int(list(total_validators)[0][0])

    def row_to_json(row):
        client = row[0]
        count = row[1]

        if total_validators == 0:
            percentage = 0
        else:
            percentage = count / total_validators

        return {
            "client": client,
            "percentage": percentage,
        }

    return [row_to_json(row) for row in rows]

def get_client_diversity_validators_lsd_overview(block_db, start_slot, end_slot=None):

    end_slot = end_slot or (1 << 62)

    rows = block_db.execute(
        """SELECT best_guess_single, COUNT(DISTINCT proposer_index) FROM blocks
                WHERE slot >= ? AND slot < ? AND lsd_id IS NOT NULL
                GROUP BY best_guess_single""",
        (start_slot, end_slot),
    )

    total_validators = block_db.execute(
        """SELECT COUNT(DISTINCT proposer_index) FROM blocks
                WHERE slot >= ? AND slot < ? AND lsd_id IS NOT NULL""",
        (start_slot, end_slot),
    )

    total_validators = int(list(total_validators)[0][0])

    def row_to_json(row):
        client = row[0]
        count = row[1]

        if total_validators == 0:
            percentage = 0
        else:
            percentage = count / total_validators
            
        return {
            "client": client,
            "percentage": percentage,
        }

    return [row_to_json(row) for row in rows]



def get_client_diversity_validators_lsd(block_db, start_slot, end_slot=None):
    # Gets the percentage of validators that are by each client for each lsd
    # Data needs to be grouped first by lsd_id and then for each lsd,
    # grouped by client best guess
    # Returns a list of dictionaries with the client name and percentage
    # The percentage is over the total number of validators in the range

    end_slot = end_slot or (1 << 62)

    rows = block_db.execute(
        """SELECT lsd_id, lsd_ticker, lsd_index, best_guess_single, COUNT(DISTINCT proposer_index) FROM blocks
                WHERE slot >= ? AND slot < ?
                GROUP BY lsd_id, best_guess_single""",
        (start_slot, end_slot),
    )

    def row_to_json(row):
        lsd_id = row[0]
        lsd_ticker = row[1]
        lsd_index = row[2]
        client = row[3]
        count = row[4]

        if lsd_id is None:
            total_validators = block_db.execute(
                """SELECT COUNT(DISTINCT proposer_index) FROM blocks
                WHERE slot >= ? AND slot < ? AND lsd_id IS NULL""",
                (start_slot, end_slot),
            )
        else:
            total_validators = block_db.execute(
                """SELECT COUNT(DISTINCT proposer_index) FROM blocks
                WHERE slot >= ? AND slot < ? AND lsd_id = ?""",
                (start_slot, end_slot, lsd_id),
            )

        total_validators = int(list(total_validators)[0][0])

        if total_validators == 0:
            percentage = 0
        else:
            percentage = count / total_validators

        return {
            "lsd_id": lsd_id,
            "lsd_ticker": lsd_ticker,
            "lsd_index": lsd_index,
            "client": client,
            "percentage": percentage,
        }

    def group_to_lsd(data):
        # Take the results from row to json and group them by lsd_id
        # Returns a list of dictionaries with the lsd_id and a list of clients

        result = {}
        for row in data:
            lsd_id = row["lsd_id"]
            if lsd_id not in result:
                result[lsd_id] = {"lsd_id": lsd_id, "lsd_ticker": row["lsd_ticker"],
                                  "lsd_index": row["lsd_index"], "clients": []}
            result[lsd_id]["clients"].append(
                {"client": row["client"], "percentage": row["percentage"]})

        return list(result.values())

    return group_to_lsd([row_to_json(row) for row in rows])


def get_client_diversity_blocks_for_lsd(block_db, lsd_id, start_slot, end_slot=None):
    # Gets the percentage of blocks that are by each client for a given lsd
    # Data needs to be grouped by client best guess
    # Returns a list of dictionaries with the client name and percentage
    # The percentage is over the total number of blocks in the range

    end_slot = end_slot or (1 << 62)

    rows = block_db.execute(
        """SELECT best_guess_single, COUNT(*) FROM blocks
              WHERE lsd_id = ? AND slot >= ? AND slot < ?
              GROUP BY best_guess_single""",
        (lsd_id, start_slot, end_slot),
    )

    total_blocks = block_db.execute(
        """SELECT COUNT(*) FROM blocks
                WHERE lsd_id = ? AND slot >= ? AND slot < ?""",
        (lsd_id, start_slot, end_slot),
    )

    total_blocks = int(list(total_blocks)[0][0])

    def row_to_json(row):
        client = row[0]
        count = row[1]

        if total_blocks == 0:
            percentage = 0
        else:
            percentage = count / total_blocks

        return {
            "client": client,
            "percentage": percentage,
        }

    return [row_to_json(row) for row in rows]


def get_client_diversity_validators_for_lsd(block_db, lsd_id, start_slot, end_slot=None):
    # Gets the percentage of validators that are by each client for a given lsd
    # Data needs to be grouped by client best guess
    # Returns a list of dictionaries with the client name and percentage
    # The percentage is over the total number of validators in the range

    end_slot = end_slot or (1 << 62)

    rows = block_db.execute(
        """SELECT best_guess_single, COUNT(DISTINCT proposer_index) FROM blocks
              WHERE lsd_id = ? AND slot >= ? AND slot < ?
              GROUP BY best_guess_single""",
        (lsd_id, start_slot, end_slot),
    )

    total_validators = block_db.execute(
        """SELECT COUNT(DISTINCT proposer_index) FROM blocks
                WHERE lsd_id = ? AND slot >= ? AND slot < ?""",
        (lsd_id, start_slot, end_slot),
    )

    total_validators = int(list(total_validators)[0][0])

    def row_to_json(row):
        client = row[0]
        count = row[1]

        if total_validators == 0:
            percentage = 0
        else:
            percentage = count / total_validators

        return {
            "client": client,
            "percentage": percentage,
        }

    return [row_to_json(row) for row in rows]


def count_true_positives(block_db, client, slot_lower, slot_upper):
    rows = block_db.execute(
        """SELECT COUNT(*) FROM blocks
           WHERE best_guess_single = ? AND graffiti_guess = ? AND
                 slot >= ? AND slot < ?""",
        (client, client, slot_lower, slot_upper),
    )
    return int(list(rows)[0][0])


def count_true_negatives(block_db, client, slot_lower, slot_upper):
    rows = block_db.execute(
        """SELECT COUNT(*) FROM blocks
           WHERE best_guess_single <> ? AND graffiti_guess <> ? AND graffiti_guess IS NOT NULL AND
                 slot >= ? AND slot < ?""",
        (client, client, slot_lower, slot_upper),
    )
    return int(list(rows)[0][0])


def count_false_positives(block_db, client, slot_lower, slot_upper):
    rows = block_db.execute(
        """SELECT COUNT(*) FROM blocks
           WHERE best_guess_single = ? AND graffiti_guess <> ? AND graffiti_guess IS NOT NULL AND
                 slot >= ? AND slot < ?""",
        (client, client, slot_lower, slot_upper),
    )
    return int(list(rows)[0][0])


def count_false_negatives(block_db, client, slot_lower, slot_upper):
    rows = block_db.execute(
        """SELECT COUNT(*) FROM blocks
           WHERE best_guess_single <> ? AND graffiti_guess = ? AND
                 slot >= ? AND slot < ?""",
        (client, client, slot_lower, slot_upper),
    )
    return int(list(rows)[0][0])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", required=True, help="path to sqlite database file")
    parser.add_argument(
        "--data-dir", required=True, help="training data for classifier(s)"
    )
    parser.add_argument("--classify-dir", required=True, help="data to classify")
    parser.add_argument(
        "--multi-classifier",
        default=False,
        action="store_true",
        help="build MultiClassifier from datadir",
    )
    parser.add_argument(
        "--force-rebuild", action="store_true", help="delete any existing database"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    db_path = args.db_path
    data_dir = args.data_dir
    data_to_classify = args.classify_dir

    if args.multi_classifier:
        classifier = MultiClassifier(data_dir)
    else:
        print("loading single KNN classifier")
        classifier = Classifier(data_dir)
        print("loaded")

    conn = build_block_db(
        db_path, classifier, data_to_classify, force_rebuild=args.force_rebuild
    )

    conn.close()


if __name__ == "__main__":
    main()
