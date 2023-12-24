from FarmInsectsClassifier.torch.entity.config_entity import CallBackConfig
import torch
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import time

class SaveBestModel:
    """
    This class saves the model state from the epoch where the model performed the best
    """

    def __init__(self, config: CallBackConfig) -> None:

        self.config = config
        self.best_val_loss = float("inf")
        


    def __call__(self, val_loss: float, model) -> None:
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss

            torch.save(model, self.config.checkpoint_model_path)

    


class EarlyStopper:
    """
    This class stops the model training when the validation loss has not improved by `min_delta` in
     the last`patience` epoch

    """

    def __init__(self, patience: int =1, min_delta: float = 0) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss: float) -> bool:
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

class TensorBoardWriter:
    def __init__(self, config: CallBackConfig):

        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
        tb_running_log_dir = Path(config.tensorboard_root_log_dir) / f"tb_logs_at_{timestamp}"
        self.writer = SummaryWriter(tb_running_log_dir)
        

    def write(self, epoch: int, train_loss: float, val_loss: float,
                            train_acc: float, val_acc: float) -> None:
        """
        Function to log model metrics on tensorboard
        """
        self.writer.add_scalar("train_loss", train_loss, epoch)
        self.writer.add_scalar("val_loss", val_loss, epoch)
        self.writer.add_scalar("train_accuracy", train_acc, epoch)
        self.writer.add_scalar("val_accuracy", val_acc, epoch)

    def close(self) -> None:
        self.writer.close()


def get_call_backs(config: CallBackConfig,
                    patience: int = 0,
                    min_delta: float = 1) -> tuple[EarlyStopper, TensorBoardWriter, SaveBestModel]:
    
    """
    Function that returns a tuple the EarlyStopper, TensorBoardWriter and SaveBestModel classes
    
    Parameters
    ----------
    config : CallBackConfig
        CallBackConfig class contains the necessary file paths for both TensorBoardWriter and 
        SaveBestModel class
    
    patience : int
        Same as patience argument for EarlyStopper class. The number of Epochs without improvement before training is stopped
        
    min_delta : float
        Same as min_delta for EarlyStopper class. The mimnimum change in val_loss that counts as model improvement
        
    Returns
    -------
    Tuple of EarlyStopper, TensorBoardWriter, SaveBestModel """
    early_stopper = EarlyStopper(patience = patience, min_delta = min_delta)
    writer = TensorBoardWriter(config)
    save_best = SaveBestModel(config)

    return early_stopper, writer, save_best