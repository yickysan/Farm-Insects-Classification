from FarmInsectsClassifier.torch.config.configuration import ModelTrainerConfigManager
from FarmInsectsClassifier.torch.components.model_trainer import TrainModel
from FarmInsectsClassifier.exception import ModelTrainerError
from FarmInsectsClassifier.logger import logging


class TorchModelTrainingPipeline:

    def __init__(self) -> None:
        self.stage_name = "ModelTraining"
    
    def initiate_pipeline(self, **kwargs) -> None:
        try:
            callbacks = kwargs["callbacks"]

            model_trainer_config_manager = ModelTrainerConfigManager()
            model_trainer_config = model_trainer_config_manager.get_config()
            modeltrainer = TrainModel(config=model_trainer_config)
            modeltrainer.train(callbacks)
        
        except Exception as e:
            logging.info(e)
            raise ModelTrainerError(e)
