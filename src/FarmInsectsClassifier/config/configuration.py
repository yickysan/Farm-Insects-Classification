from __future__ import annotations

from FarmInsectsClassifier.constants import CONFIG_FILE_PATH, PARAM_FILE_PATH
from FarmInsectsClassifier.utils import read_yaml, create_directories
from FarmInsectsClassifier.entity.config_entity import DataIngestionConfig, BaseModelConfig, CallBackConfig, ModelTrainerConfig
from FarmInsectsClassifier.logger import logging

from typing import Protocol, ClassVar
from abc import ABC,  abstractmethod
from pathlib import Path


class Config(Protocol):
    __dataclass_fields__: ClassVar[dict]


class ConfigurationManager(ABC):
    def __init__(self,
                 config_file_path = CONFIG_FILE_PATH,
                 param_file_path = PARAM_FILE_PATH):
        
        
        self.config = read_yaml(CONFIG_FILE_PATH)
        self.params = read_yaml(PARAM_FILE_PATH)

        create_directories([self.config.artifacts_root])

    @abstractmethod
    def get_config(self: ConfigurationManager) -> Config:
        pass


class DataIngestionConfigManager(ConfigurationManager):

    def get_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion


        logging.info("creating artifacts/data_ingestion directory")
        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir = Path(config.root_dir),
            unzip_dir = Path(config.unzip_dir)

        )

        return data_ingestion_config


class BaseModelConfigManager(ConfigurationManager):

    def get_config(self: ConfigurationManager) -> BaseModelConfig:
        
        config = self.config.base_model_config

        logging.info("creating artifacts/models")
        create_directories([config.root_dir])

        base_model_config = BaseModelConfig(
            root_dir = Path(config.root_dir),
            base_model_path = Path(config.base_model_path),
            updated_base_model_path = Path(config.updated_base_model_path),
            params_classes = self.params.CLASSES,
            params_image_size = self.params.IMAGE_SIZE,
            params_learning_rate = self.params.LEARNING_RATE,
            params_include_top = self.params.INCLUDE_TOP,
            params_weights= self.params.WEIGHTS
                                           
            )
        
        return base_model_config
            

class CallBackConfigManager(ConfigurationManager):
    
    def get_config(self: ConfigurationManager) -> CallBackConfig:
        config = self.config.callback_config


        logging.info("creating artifacts/callbacks/tensorboard_long and artifacts/callbacks/checkpoint")
        create_directories([
            Path(config.tensorboard_root_log_dir),
            Path(config.checkpoint_model_path).parent]
            )

        callback_config = CallBackConfig(
            root_dir = Path(config.root_dir),
            tensorboard_root_log_dir = Path(config.tensorboard_root_log_dir),
            checkpoint_model_path = Path(config.checkpoint_model_path)
        )

        return callback_config


class ModelTrainerConfigManager(ConfigurationManager):

    def get_config(self: ConfigurationManager) -> ModelTrainerConfig:
        config = self.config.model_trainer_config

        logging.info("Creating Model Trainer Directory")
        create_directories([Path(config.root_dir)])

        model_trainer_config = ModelTrainerConfig(
            root_dir = Path(config.root_dir),
            model_path = Path(config.model_path),
            updated_base_model_path = Path(self.config.base_model_config.updated_model_path),
            train_data = Path(self.config.data_ingestion.unzip_dir) / "farm-insects-splitted/train",
            validation_data = Path(self.config.data_ingestion.unzip_dir) / "farm-insects-splitted/validation",
            params_epochs = self.params.EPOCHS,
            params_batch_size = self.params.BATCH_SIZE,
            params_is_augmentation = self.params.AUGMENTATION,
            params_image_size = self.params.IMAGE_SIZE
        )

        return model_trainer_config
