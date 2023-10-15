from FarmInsectsClassifier.pipeline.pipeline_01_data_ingestion import DataIngestionPipeline
from FarmInsectsClassifier.logger import logging

from pathlib import Path

DATA_PATH = Path("archive.zip").resolve()


if __name__ == "__main__":

    logging.info(f">>>> Pipeline Data Ingestion Initiated <<<<")
    ingestion_pipeline = DataIngestionPipeline()
    ingestion_pipeline.initiate_pipeline(DATA_PATH)

    logging.info(f">>>> Pipeline Data Ingestion Completed <<<<")
    
                        
