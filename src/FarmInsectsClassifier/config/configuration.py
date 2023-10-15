from FarmInsectsClassification.constants import CONFIG_FILE_PATH, PARAM_FILE_PATH
from FarmInsectsClassification.utils.utils import read_yaml, create_directories
from FarmInsectsClassification.entity.config_entity import DataIngestionCongig

class ConfigurationManager:
    def __init__(self,
                 config_file_path: CONFIG_FILE_PATH = CONFIG_FILE_PATH,
                 param_file_path: PARAM_FILE_PATH = PARAM_FILE_PATH):
        
        self.config = read_yaml(CONFIG_FILE_PATH)
        self.params = read_yaml(PARAM_FILE_PATH)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionCongig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionCongig(
            root_dir = config.root_dir,
            local_data_file = config.local_data_file
        )

        return data_ingestion_config