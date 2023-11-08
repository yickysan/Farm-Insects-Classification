from FarmInsectsClassifier.config.configuration import CallBackConfigManager
from FarmInsectsClassifier.components.callbacks import PrepareCallBacks
from FarmInsectsClassifier.exception import CallBackError
from FarmInsectsClassifier.logger import logging


class PrepCallBacksPipeline:

    def __init__(self) -> None:
        self.stage_name = "Callbacks Preparation"
    
    def initiate_pipeline(self, **kwargs) -> None:
        try:
            callback_config_manager = CallBackConfigManager()
            callback_config = callback_config_manager.get_config()
            callbacks = PrepareCallBacks(config=callback_config)
            callback_list = callbacks.get_tb_ckpt_callbacks()
        
        except Exception as e:
            logging.info(e)
            raise CallBackError(e)