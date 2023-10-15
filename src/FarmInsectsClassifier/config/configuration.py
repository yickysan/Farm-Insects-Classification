from FarmInsectsClassifier.constants import CONFIG_FILE_PATH, PARAM_FILE_PATH
from FarmInsectsClassifier.utils import read_yaml, create_directories
from FarmInsectsClassifier.entity.config_entity import DataIngestionConfig
from FarmInsectsClassifier.logger import logging

from pathlib import Path

class ConfigurationManager:
    def __init__(self,
                 config_file_path = CONFIG_FILE_PATH,
                 param_file_path = PARAM_FILE_PATH):
        
        
        self.config = read_yaml(CONFIG_FILE_PATH)
        self.params = read_yaml(PARAM_FILE_PATH)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion


        logging.info("creating artifacts directory")
        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir = Path(config.root_dir),
            unzip_dir = Path(config.unzip_dir)

        )

        return data_ingestion_config