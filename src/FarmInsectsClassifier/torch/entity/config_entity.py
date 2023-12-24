from pathlib import Path
from dataclasses import dataclass

@dataclass(frozen=True)
class BaseModelConfig:
    root_dir: Path
    base_model_path: Path
    updated_base_model_path: Path
    params_image_size: list[int]
    params_classes : int
    params_weights: str


@dataclass(frozen=True)
class CallBackConfig:
    root_dir: Path
    tensorboard_root_log_dir: Path
    checkpoint_model_path: Path


@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    model_path: Path
    updated_base_model_path: Path
    train_data: Path
    validation_data: Path
    params_epochs: int
    params_batch_size: int
    params_is_augmentation: bool
    params_learning_rate: float
