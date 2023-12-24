from FarmInsectsClassifier.pipeline.pipeline_01_data_ingestion import DataIngestionPipeline

from FarmInsectsClassifier.pipeline.pipeline_02_base_model_preparation import BaseModelPrepPipeline
from FarmInsectsClassifier.torch.pipeline.pipeline_02_base_model_preparation import TorchBaseModelPrepPipeline

from FarmInsectsClassifier.pipeline.pipeline_03_prepare_callbacks import PrepCallBacksPipeline
from FarmInsectsClassifier.torch.pipeline.pipeline_03_get_callbacks import TorchCallBacksPipeline

from FarmInsectsClassifier.pipeline.pipeline_04_model_training import ModelTrainingPipeline
from FarmInsectsClassifier.torch.pipeline.pipeline_04_model_training import TorchModelTrainingPipeline

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

    callbacks_pipeline = TorchCallBacksPipeline()
    callbacks = callbacks_pipeline.initiate_pipeline(patience=10, min_delta = 0.01)
    model_training_pipe = TorchModelTrainingPipeline()
    main(model_training_pipe, stage_name="Model Training", callbacks = callbacks)

    # base_model_prep_pipeline = torch_base_model_pipeline()
    # main(pipe = base_model_prep_pipeline, stage_name = "Base Model Preparation")
    # callbacks_pipeline = PrepCallBacksPipeline()
    # call_backs = callbacks_pipeline.initiate_pipeline()
    # model_training_pipeline = ModelTrainingPipeline()
    
    # main(pipe=model_training_pipeline, stage_name="Model Training", callback_list = call_backs)
    # DATA_PATH = Path("archive.zip").resolve()
    # data_ingestion_pipeline = DataIngestionPipeline()

    # main(pipe=data_ingestion_pipeline, stage_name="Data Ingestion", data_path = DATA_PATH)
    
    
                        
