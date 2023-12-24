from FarmInsectsClassifier.torch.entity.config_entity import ModelTrainerConfig
from FarmInsectsClassifier.torch.utils import train, validate
from FarmInsectsClassifier.torch.components.callbacks import EarlyStopper, SaveBestModel, TensorBoardWriter
from FarmInsectsClassifier.logger import logging

import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import numpy as np

from pathlib import Path
from typing import Optional


class TrainModel:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
        


    def _get_base_model(self) -> None:
        self.model = torch.load(self.config.updated_base_model_path)
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr = self.config.params_learning_rate
                                          )
        self.loss = torch.nn.CrossEntropyLoss()


    def _preprocess_data(self) -> None:

        data_transforms =  transforms.Compose([
            transforms.ToTensor(), 
            transforms.Resize((224, 224), antialias=True) #type: ignore
            ])


        self.train_data = DataLoader(
            datasets.ImageFolder(str(self.config.train_data), transform = data_transforms),
            batch_size = self.config.params_batch_size
            )
        
       
        
        self.validation_data = DataLoader(
            datasets.ImageFolder(str(self.config.validation_data), transform = data_transforms),
            batch_size = self.config.params_batch_size
            )
        


        if self.config.params_is_augmentation:
            train_transforms = transforms.Compose(
                [transforms.ToTensor(), 
                transforms.Resize((224, 224), antialias=True), #type: ignore
                transforms.RandomRotation(30),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomPerspective(),
                transforms.RandomGrayscale()
                ]
            )

            self.train_data = DataLoader(
            datasets.ImageFolder(str(self.config.train_data), transform = train_transforms),
            batch_size = self.config.params_batch_size
            )

            

    @staticmethod
    def save_model(path: Path, model) -> None:
        torch.save(model, path)


    def train(self, callbacks: tuple[EarlyStopper, TensorBoardWriter, SaveBestModel],
               save_best: Optional[bool] = True ) -> None:
        
        logging.info("Data preprocessing started")
        self._preprocess_data()

        logging.info("Loading Model")
        self._get_base_model()

        logging.info("Model training started")
        
        early_stopper, writer, model_saver = callbacks

        for epoch in range(self.config.params_epochs):
            loss, train_acc = train(self.train_data, self.model, self.optimizer, self.loss)
            val_loss, val_acc = validate(self.validation_data, self.model, self.loss)

            if early_stopper.early_stop(val_loss):
                break

            writer.write(epoch, loss, val_loss, train_acc, val_acc)

            if save_best:
                model_saver(val_loss, self.model)

            print(f"Epoch: {epoch}  loss - {loss}  train_accuracy - {train_acc},  val_loss - {val_loss}  val_accuracy - {val_acc}")
            print("==============================================================================================================================")

        writer.close()
        logging.info("Saving model")
        self.save_model(path=self.config.model_path, model=self.model) #type: ignore

        