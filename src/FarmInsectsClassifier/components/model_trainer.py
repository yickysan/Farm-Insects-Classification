from FarmInsectsClassifier.entity.config_entity import ModelTrainerConfig
from FarmInsectsClassifier.logger import logging

import tensorflow as tf
from tensorflow import keras
import time
import numpy as np
from pathlib import Path


class TrainModel:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def get_base_model(self) -> None:
        self.model = keras.models.load_model(self.config.updated_base_model_path)


    def _preprocess_data(self) -> None:
        train_data = keras.utils.image_dataset_from_directory(
            self.config.train_data,
            batch_size = self.config.params_batch_size,
            image_size = self.config.params_image_size,
            shuffle = False)
        
        self.train_samples = len(train_data.file_paths) #type: ignore
        
        # scale data so min value is 0 and maximum value is 1
        self.train_data = train_data.map(lambda x, y: (x/255, y)) # type: ignore
        
        validation_data = keras.utils.image_dataset_from_directory(
            self.config.validation_data,
            batch_size = self.config.params_batch_size,
            image_size = self.config.params_image_size,
            shuffle = False)
        
        self.validation_samples = len(validation_data.file_paths) #type: ignore
        
        self.validation = validation_data.map(lambda x, y: (x/255, y)) # type: ignore


        if self.config.params_is_augmentation:
            augmentation = keras.Sequential([
                keras.layers.RandomFlip(),
                keras.layers.RandomRotation(0.2),
                keras.layers.RandomZoom(height_factor=(0.2, 0.3), width_factor=(0.2, 0.3))
            ])

            self.train_data = self.train_data.map(lambda x, y: (augmentation(x, traing=True), y))
            self.validation = self.validation.map(lambda x, y: (augmentation(x, traing=True), y))
        
        
    

    @staticmethod
    def save_model(path: Path, model: keras.Model) -> None:
        model.save(path)


    def train(self, callback_list = list[keras.callbacks.TensorBoard | keras.callbacks.ModelCheckpoint]) -> None:
        
        logging.info("Data preprocessing started")
        self._preprocess_data()

        self.steps_per_epochs = (self.train_samples // self.config.params_batch_size) + 1
        self.validation_steps = (self.validation_samples // self.config.params_batch_size) + 1

        logging.info("Model training started")
        self.model.fit( #type: ignore
            self.train_data,
            epochs = self.config.params_epochs,
            steps_per_epochs = self.steps_per_epochs,
            validation_steps = self.validation_steps,
            validation_data = self.validation,
            callbacks = callback_list
        )


        logging.info("Saving model")
        self.save_model(path=self.config.model_path, model=self.model) #type: ignore

        