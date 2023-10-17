from FarmInsectsClassifier.config.configuration import BaseModelConfigManager
from FarmInsectsClassifier.components.base_model import PrepareBaseModel
from FarmInsectsClassifier.exception import BaseModelPrepError
from FarmInsectsClassifier.logger import logging


class BaseModelPrepPipeline:

    def __init__(self) -> None:
        self.stage_name = "Base Model Preparation"
    
    def initiate_pipeline(self, **kwargs) -> None:
        try:
            base_model_config_manager = BaseModelConfigManager()
            base_model_config = base_model_config_manager.get_config()
            base_model = PrepareBaseModel(config=base_model_config)
            base_model.get_base_model()
            base_model.update_base_model()
        
        except Exception as e:
            logging.info(e)
            raise BaseModelPrepError(e)