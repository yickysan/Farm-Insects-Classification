from FarmInsectsClassifier.config.configuration import ConfigurationManager
from FarmInsectsClassifier.components.data_ingestion import DataIngestion
from FarmInsectsClassifier.logger import logging

from pathlib import Path


class DataIngestionPipeline:

    def __init__(self) -> None:
        self.stage_name = "Data Ingestion Stage"
    
    def initiate_pipeline(self, data_path: Path) -> None:
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(data_path=data_path, config=data_ingestion_config)
        data_ingestion.unzip()
        data_ingestion.split_data()
