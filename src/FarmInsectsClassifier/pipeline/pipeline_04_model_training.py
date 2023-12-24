from FarmInsectsClassifier.config.configuration import ModelTrainerConfigManager
from FarmInsectsClassifier.components.model_trainer import TrainModel
from FarmInsectsClassifier.exception import ModelTrainerError
from FarmInsectsClassifier.logger import logging


class ModelTrainingPipeline:

    def __init__(self) -> None:
        self.stage_name = "ModelTraining"
    
    def initiate_pipeline(self, **kwargs) -> None:
        """
        Parameters
        ----------
        kwargs[callback_list] : List
            List of keras.callbacks.ModelCheckpoint, keras.callbacks.TensorBoard, 
            keras.callbacks.EearlyStopping
            
        """
        try:
            callback_list = kwargs["callback_list"]

            model_trainer_config_manager = ModelTrainerConfigManager()
            model_trainer_config = model_trainer_config_manager.get_config()
            modeltrainer = TrainModel(config=model_trainer_config)
            modeltrainer.train(callback_list=callback_list)
        
        except Exception as e:
            logging.info(e)
            raise ModelTrainerError(e)
