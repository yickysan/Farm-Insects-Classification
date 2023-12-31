from FarmInsectsClassifier.entity.config_entity import CallBackConfig
from FarmInsectsClassifier.logger import logging

import tensorflow as tf
import time
from pathlib import Path

class PrepareCallBacks:
    def __init__(self, config: CallBackConfig):
        self.config = config


    @property
    def _create_tb_callbacks(self) -> tf.keras.callbacks.TensorBoard:
        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
        tb_running_log_dir = Path(self.config.tensorboard_root_log_dir) / f"tb_logs_at_{timestamp}"

        return tf.keras.callbacks.TensorBoard(log_dir=str(tb_running_log_dir))


    @property
    def _create_ckpt_callbacks(self) -> tf.keras.callbacks.ModelCheckpoint:

        return tf.keras.callbacks.ModelCheckpoint(
            filepath = str(self.config.checkpoint_model_path),
            save_best_only = True
        )

    @property
    def _early_stopping(self) -> tf.keras.callbacks.EarlyStopping:
        early_stopping = tf.keras.callbacks.EarlyStopping(
                            monitor = "val_loss",
                            patience = 10,
                            min_delta = 0.001, #type: ignore
                            restore_best_weights = True)

        return early_stopping
    

    def get_tb_ckpt_callbacks(self) -> list[tf.keras.callbacks.TensorBoard |
                                            tf.keras.callbacks.ModelCheckpoint |
                                            tf.keras.callbacks.EarlyStopping]:

        return [
            self._create_tb_callbacks,
            self._create_ckpt_callbacks,
            self._early_stopping
            ]
    