from FarmInsectsClassifier.config.configuration import DataIngestionConfigManager
from FarmInsectsClassifier.components.data_ingestion import DataIngestion
from FarmInsectsClassifier.exception import DataIngestionError
from FarmInsectsClassifier.logger import logging

from pathlib import Path


class DataIngestionPipeline:

    def __init__(self) -> None:
        self.stage_name = "Data Ingestion Stage"
    
    def initiate_pipeline(self, **kwargs) -> None:
        try:
            data_path = kwargs["data_path"]
            data_ingestion_config_manager = DataIngestionConfigManager()
            data_ingestion_config = data_ingestion_config_manager.get_config()
            data_ingestion = DataIngestion(data_path=data_path, config=data_ingestion_config)
            data_ingestion.unzip()
            data_ingestion.split_data()
        
        except Exception as e:
            logging.info(e)
            raise DataIngestionError(e)
            
