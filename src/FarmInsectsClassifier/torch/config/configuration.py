from __future__ import annotations

from FarmInsectsClassifier.constants import CONFIG_FILE_PATH, PARAM_FILE_PATH
from FarmInsectsClassifier.utils import read_yaml, create_directories
from FarmInsectsClassifier.torch.entity.config_entity import BaseModelConfig, CallBackConfig, ModelTrainerConfig
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



class BaseModelConfigManager(ConfigurationManager):

    def get_config(self: ConfigurationManager) -> BaseModelConfig:
        
        config = self.config.base_model_config.torch

        logging.info("creating artifacts/torch/models/")
        create_directories([config.root_dir])

        base_model_config = BaseModelConfig(
            root_dir = Path(config.root_dir),
            base_model_path = Path(config.base_model_path),
            updated_base_model_path = Path(config.updated_base_model_path),
            params_classes = self.params.torch.CLASSES,
            params_image_size = self.params.torch.IMAGE_SIZE,
            params_weights= self.params.torch.WEIGHTS
                                           
            )
        
        return base_model_config
            

class CallBackConfigManager(ConfigurationManager):
    
    def get_config(self: ConfigurationManager) -> CallBackConfig:
        config = self.config.callback_config.torch


        logging.info(f"Creating directory {config.root_dir}")
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
        config = self.config.model_trainer_config.torch
        base_model_config = self.config.base_model_config.torch

        logging.info("Creating Model Trainer Directory")
        create_directories([Path(config.root_dir)])

        model_trainer_config = ModelTrainerConfig(
            root_dir = Path(config.root_dir),
            model_path = Path(config.model_path),
            updated_base_model_path = Path(base_model_config.updated_base_model_path),
            train_data = Path(self.config.data_ingestion.unzip_dir) / "farm-insects-splitted/train",
            validation_data = Path(self.config.data_ingestion.unzip_dir) / "farm-insects-splitted/val",
            params_epochs = self.params.torch.EPOCHS,
            params_batch_size = self.params.torch.BATCH_SIZE,
            params_is_augmentation = self.params.torch.AUGMENTATION,
            params_learning_rate = self.params.torch.LEARNING_RATE
        )

        return model_trainer_config
