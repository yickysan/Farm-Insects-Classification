from FarmInsectsClassifier.torch.entity.config_entity import BaseModelConfig
from FarmInsectsClassifier.logger import logging
import torchvision.models as models
import torch
from pathlib import Path


class PrepareBaseModel:
    def __init__(self, config: BaseModelConfig) -> None:
        self.config = config

    def get_base_model(self) -> None:

        logging.info("Inititalising base model")
        self.model = models.resnet50(weights = self.config.params_weights)

        self.save_model(path=self.config.base_model_path, model=self.model)

        logging.info("Base model saved successfully")


    def update_base_model(self) -> None:

        logging.info("Updating base model")

        self.fullmodel = self.prepare_model(
            model = self.model,
            classes = self.config.params_classes,
            freeze_all = True,
        )

        self.save_model(path=self.config.updated_base_model_path, model=self.fullmodel)
        
        logging.info("Updated model saved successfully")
    
    @staticmethod
    def prepare_model(model: models.ResNet,
                      classes: int, 
                      freeze_all: bool,
                      freeze_till: int = 0) -> models.ResNet:

        if freeze_all:
            for param in model.parameters():
                param.requires_grad = False

        elif freeze_till >  0:
            for param in list(model.parameters())[:freeze_till]:
                param.requires_grad = False

        num_features = model.fc.in_features
        model.fc = torch.nn.Linear(num_features, classes)

        full_model =  model.cuda() if torch.cuda.is_available() else model

        print(full_model)

        return full_model


    @staticmethod
    def save_model(path: Path, model):
        torch.save(model, path)
