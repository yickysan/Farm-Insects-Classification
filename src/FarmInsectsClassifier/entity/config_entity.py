from pathlib import Path
from dataclasses import dataclass

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    unzip_dir: Path


@dataclass(frozen=True)
class BaseModelConfig:
    root_dir: Path
    base_model_path: Path
    updated_base_model_path: Path
    params_image_size: list
    params_learning_rate: float
    params_classes : int
    params_include_top: bool
    params_weights: str


@dataclass(frozen=True)
class CallBackConfig:
    root_dir: Path
    tensorboard_root_log_dir: Path
    checkpoint_model_path: Path
   