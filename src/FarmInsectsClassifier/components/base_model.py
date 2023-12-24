from FarmInsectsClassifier.entity.config_entity import BaseModelConfig
from FarmInsectsClassifier.logger import logging

from pathlib import Path
from tensorflow import keras

class PrepareBaseModel:
    def __init__(self, config: BaseModelConfig) -> None:
        self.config = config

    def get_base_model(self) -> None:

        logging.info("Inititalising base model")
        self.model = keras.applications.ResNet50(
            input_shape = self.config.params_image_size,
            weights = self.config.params_weights,
            include_top = self.config.params_include_top

        )

        self.save_model(path=self.config.base_model_path, model=self.model)

        logging.info("Base model saved successfully")


    def update_base_model(self) -> None:

        logging.info("Updating base model")

        self.fullmodel = self.prepare_model(
            model = self.model,
            classes = self.config.params_classes,
            freeze_all = True,
            learning_rate = self.config.params_learning_rate
        )

        self.save_model(path=self.config.updated_base_model_path, model=self.fullmodel)
        
        logging.info("Updated model saved successfully")
    
    @staticmethod
    def prepare_model(model: keras.Model, 
                      classes: int, 
                      learning_rate: float,
                      freeze_all: bool,
                      freeze_till: int = 0) -> keras.Model:

        if freeze_all:
            for layer in model.layers:
                model.trainable = False

        elif freeze_till >  0:
            for layer in model.layers[:freeze_till]:
                model.trainable = False

        flatten_in = keras.layers.Flatten()(model.output)
        pred = keras.layers.Dense(units=classes, activation="softmax")(flatten_in)

        full_model = keras.models.Model(
            inputs = model.input,
            outputs = pred
        )

        full_model.compile(
            optimizer = keras.optimizers.Adam(learning_rate=learning_rate),
            loss = keras.losses.SparseCategoricalCrossentropy(),
            metrics =  ["accuracy"]
        )

        full_model.summary()

        return full_model


    @staticmethod
    def save_model(path: Path, model: keras.Model):
        model.save(path)


