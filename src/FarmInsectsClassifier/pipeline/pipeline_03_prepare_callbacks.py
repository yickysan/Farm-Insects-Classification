from FarmInsectsClassifier.config.configuration import CallBackConfigManager
from FarmInsectsClassifier.components.callbacks import PrepareCallBacks
from FarmInsectsClassifier.exception import CallBackError
from FarmInsectsClassifier.logger import logging
from tensorflow import keras


class PrepCallBacksPipeline:

    def __init__(self) -> None:
        self.stage_name = "Callbacks Preparation"
    
    def initiate_pipeline(self, **kwargs) -> list[
        keras.callbacks.TensorBoard | 
        keras.callbacks.ModelCheckpoint | 
        keras.callbacks.EarlyStopping
        ]:
        
        try:
            callback_config_manager = CallBackConfigManager()
            callback_config = callback_config_manager.get_config()
            callbacks = PrepareCallBacks(config=callback_config)
            callback_list = callbacks.get_tb_ckpt_callbacks()

            return callback_list
        
        except Exception as e:
            logging.info(e)
            raise CallBackError(e)