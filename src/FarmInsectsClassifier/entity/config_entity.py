from pathlib import Path
from dataclasses import dataclass

@dataclass(frozen=True)
class DataIngestionCongig:
    root_dir = Path
    local_data_file = Path