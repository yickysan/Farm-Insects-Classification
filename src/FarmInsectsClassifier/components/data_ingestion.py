from FarmInsectsClassifier.entity.config_entity import DataIngestionConfig
from FarmInsectsClassifier.logger import logging
from FarmInsectsClassifier.exception import DataIngestionError

from pathlib import Path
from zipfile import ZipFile
import splitfolders

class DataIngestion:
    def __init__(self, data_path: Path, config: DataIngestionConfig):
        self.data_path = data_path
        self.config = config

    def unzip(self) -> None:
        try:
            unzip_path = self.config.unzip_dir
            unzip_path.mkdir(exist_ok=True, parents=True)

            logging.info("Extracting zip file")

            with ZipFile(self.data_path, "r") as zip_ref:
                zip_ref.extractall(unzip_path)

            logging.info("Zipfile extraction completed")

        except Exception as e:
            raise DataIngestionError(e)


    def split_data(self) ->  None:
        try:
            path = list(self.config.unzip_dir.resolve().iterdir())[0]
            output = self.config.unzip_dir / "farm-insects-splitted"

            logging.info("Splitting folder into train, test and validation set")

            splitfolders.ratio(path, seed=1, output=str(output), ratio=(0.6, 0.2, 0.2))

            logging.info("Train, test and validation data successfully created")

        except Exception as e:
            raise DataIngestionError(e)
