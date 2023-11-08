from FarmInsectsClassifier.pipeline.pipeline_01_data_ingestion import DataIngestionPipeline
from FarmInsectsClassifier.pipeline.pipeline_02_base_model_preparation import BaseModelPrepPipeline
from FarmInsectsClassifier.pipeline.pipeline_03_prepare_callbacks import PrepCallBacksPipeline
from FarmInsectsClassifier.logger import logging

from pathlib import Path
from typing import Protocol

class Pipeline(Protocol):

    def initiate_pipeline(self, **kwargs) -> None:
        ...


def main(pipe: Pipeline, stage_name: str, **kwargs) -> None:

    logging.info(f">>>> Pipeline {stage_name} Initiated <<<<")

    pipe.initiate_pipeline(**kwargs)

    logging.info(f">>>> Pipeline {stage_name} Completed <<<<")




if __name__ == "__main__":

    base_model_prep_pipeline = BaseModelPrepPipeline()
    callbacks_pipeline = PrepCallBacksPipeline()
    
    main(pipe=callbacks_pipeline, stage_name="Callbacks Preparation")
    # DATA_PATH = Path("archive.zip").resolve()
    # data_ingestion_pipeline = DataIngestionPipeline()

    # main(pipe=data_ingestion_pipeline, stage_name="Data Ingestion", data_path = DATA_PATH)
    
    
                        
