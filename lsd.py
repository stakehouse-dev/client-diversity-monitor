'''LSD'''

import requests
from database_config import SessionLocal
from validator_indexes_model import ValidatorLSD
from lsd_subgraph import LSDSubgraph
from apiconfig import APIConfig

cfg = APIConfig()

BN_URL = cfg.get_beacon_node_url()
BN_HEADER = {'x-api-key': cfg.beacon_node_api_key()}


class LSD:
    """
    LSD class to interact with the LSD subgraph and the database

    Primarily used used to check if each slot is proposed by a 
    validator in an LSD and identify the LSD
    """

    def __init__(self):
        self.BN_URL = BN_URL
        self.BN_HEADER = BN_HEADER
        self.lsd_subgraph = LSDSubgraph()

    def beacon_get_validator(self, slot, validator_index):
        """
        Retrives a validator for a given slot and validator index

        :param slot: The slot
        :param validator_index: The validator index
        :return: The validator
        """
        res = requests.get(
            f"{self.BN_URL}/eth/v1/beacon/states/{slot}/validators/{validator_index}", headers=self.BN_HEADER
        )
        res.raise_for_status()
        return res.json()["data"]["validator"]

    def get_validator_lsd_at_epoch(self, validator_pubkey, epoch):
        """
        It returns the LSD index of a validator for a given epoch

        :param validator_pubkey: The public key of the validator
        :param epoch: The epoch
        :return: The LSD index
        """
        with SessionLocal() as session:
            validator_lsd_index = (
                session.query(ValidatorLSD)
                .filter(
                    ValidatorLSD.bls_key == str(validator_pubkey),
                    ValidatorLSD.epoch <= int(epoch),
                )
                .order_by(ValidatorLSD.epoch.desc())
                .first()
            )

            print(f"Validator LSD index: {validator_lsd_index}")

            if validator_lsd_index:
                result = self.lsd_subgraph.get_lsds_by_index(
                    [int(validator_lsd_index.indexes)]
                )
                return {
                    'validator_pubkey': validator_pubkey,
                    'epoch': epoch,
                    'lsd_index': int(validator_lsd_index.indexes),
                    'lsd_ticker': result[0]['ticker'],
                    'lsd_id': result[0]['id'],
                }

            return None
