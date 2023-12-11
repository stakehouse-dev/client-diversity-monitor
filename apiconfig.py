import configparser
from os import environ, path

API_DIR = path.dirname(path.realpath(__file__))
API_CFG_FILE_NAME = 'api.cfg'
PATH_TO_API_CFG = path.join(API_DIR, API_CFG_FILE_NAME)


class APIConfig:
    def __init__(self):
        self._config = configparser.RawConfigParser()
        self._config.read([PATH_TO_API_CFG])
    
    def etl_db(self, name):
        if name == "connection_string":
            return self._etl_db_connection_string()
        try:
            value = self._config.get("ETL_DB", name)
        except (configparser.NoSectionError, configparser.NoOptionError):
            value = environ.get(f"ETL_DB_{name.upper()}")
        return value
    
    def _etl_db_connection_string(self):
        host = self.etl_db("host")
        username = self.etl_db("username")
        password = self.etl_db("password")
        db = self.etl_db("db")

        if not all([host, username, password, db]):
            return None

        return f"{username}:{password}@{host}/{db}"

    def get_beacon_node_url(self):
        try:
            value = self._config.get("BEACON_NODE", "url")
        except (configparser.NoSectionError, configparser.NoOptionError):
            value = environ.get("BEACON_NODE_URL")
        return value
    
    def beacon_node_api_key(self):
        try:
            value = self._config.get("BEACON_NODE", "api_key")
        except (configparser.NoSectionError, configparser.NoOptionError):
            value = environ.get("BEACON_NODE_API_KEY")
        return value
    
    def get_lsd_subgraph_url(self):
        try:
            value = self._config.get("LSD_SUBGRAPH", "url")
        except (configparser.NoSectionError, configparser.NoOptionError):
            value = environ.get("LSD_SUBGRAPH_URL")
        return value