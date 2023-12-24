import torch
from torch import nn
import numpy as np
from sklearn.metrics import accuracy_score
from typing import Iterable
from pathlib import Path

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def train(data: Iterable ,
          model, optimizer, loss_fn) -> tuple[float, float]:

    """
    Function to train a pytorch Model for an Epoch
    """
    train_accuracy = []
    losses = []

    for batch, data in enumerate(data):
        optimizer.zero_grad() 
        images, targets = data
        images = images.to(DEVICE)
        targets = targets.to(DEVICE)
        outputs = model(images.float())

        loss = loss_fn(outputs, targets)
        losses.append(np.expand_dims(loss.detach().cpu().numpy(), 0)[0])

        _, preds = torch.max(outputs.data, 1)
        accuracy = accuracy_score(targets.cpu().numpy(), preds.cpu().numpy())
        train_accuracy.append(accuracy)

        loss.backward()
        optimizer.step()


    return np.array(losses).mean().round(4), np.array(train_accuracy).mean().round(4)




def validate(data: Iterable, model, loss_fn) -> tuple[float, float]:

    """
    Function to evaluate the model on the validation dataset
    """

    val_accuracy = []
    val_losses = []

    model.eval()
    with torch.no_grad():
        for batch, data in enumerate(data):
            images, targets = data
            images = images.to(DEVICE)
            targets = targets.to(DEVICE)
            outputs = model(images.float())

            val_loss = loss_fn(outputs, targets)
            val_losses.append(np.expand_dims(val_loss.cpu().numpy(), 0)[0])

            _, preds = torch.max(outputs.data, 1)


            accuracy = accuracy_score(targets.cpu().numpy(), preds.cpu().numpy())
            val_accuracy.append(accuracy)

        return np.array(val_losses).mean().round(4), np.array(val_accuracy).mean().round(4)


