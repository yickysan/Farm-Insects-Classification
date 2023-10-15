import sys
from FarmInsectsClassifier.logger import logging
import yaml
import json
import joblib
from box.exceptions import BoxValueError
from box import ConfigBox
from pathlib import Path
from typing import Any
import base64

def read_yaml(yaml_path: Path | str) -> ConfigBox:
    """"
    Function to read a yaml file
    
    Parameters
    ----------
    yaml_path : pathlib.Path, str
        file location where yaml file is stored
    
    Returns
    -------
    box.ConfigBox
    
    Raises
    ------
    ValueError
        if yaml file is empty
    e: yaml file is empty
    
    """
    try:
        with open(yaml_path, "r") as yaml_file:
            content = yaml.safe_load(yaml_file)
            logging.info(f"yaml file: {yaml_path} loaded successfully")
            
            return ConfigBox(content)
        
    except BoxValueError:
        raise ValueError("yaml file is empty")
    
    
def create_directories(dir_paths: list[Path | str], verbose: bool = True) -> None:
    """
    Function to create directories from a list of pathllb.Path objects or strings
    
    Parameters
    ----------
    dir_paths : list
        list of file locations where directories will be created
        
    verbose : bool, optional, default = True
        logs info of the directories being created
    
    """
    for path in dir_paths:
        Path(path).mkdir(parents=True, exist_ok=True)

        if verbose:
            logging.info(f"created directory at {path}")



def save_json(json_path: Path | str, data: dict) -> None:
    """
    Function to save json data
    
    Parameters
    ----------
    json_path : pathlib.Path, str
        file location where json object will be stored
        
    data : dict
        python dictionary containing data to be stored as json
        
    """

    with open(json_path, "w+") as f:
        json.dump(data, f, indent=4)

    logging.info(f"json file saved at: {json_path}") 


def load_json(json_path: Path | str) -> ConfigBox:
    """
    Functiom to load json files
    
    Parameters
    ----------
    json_data : pathlib.Path, str
        file location where json data is stored
    
    Returns
    -------
    box.ConfigBox
    
    """
    with open(json_path, "r") as f:
        content = json.load(f)
        logging.info(f"json file loaded successfully from: {json_path}")

        return ConfigBox(content)
    
def save_bin(data: Any, file_path: Path | str) -> None:
    """
    Function to save data as a binary file
    
    Parameters
    ----------
    data : Any
        any python object to be stored
        
    file_path : pathlib.Path, str
        file location where object will be stored
    
    """
    joblib.dump(data, file_path)
    logging.info(f"binary file saved at: {file_path}")


def load_bin(file_path: Path | str) -> Any:
    """
    Function to load binary file as python object
    
    Parameters
    ----------
    file_path : pathlib.Path, str
        file location where binary file is stored

    Returns
    -------
    data : Any
    
    """
    data = joblib.load(file_path)
    logging.info(f"binary file loaded from: {file_path}")

    return data


def decode_image(imgstring: bytes, filename: str) -> None:
    """
    Function to decode from byte strings and save image data
    
    Parameters
    ----------
    imgstring : bytes
        byte string to be decoded
    
    filename : str
        filename used when storing image data
    
    """
    image_data = base64.b64decode(imgstring)

    with open(filename, "wb") as f:
        f.write(image_data)


def encode_image_to_b64(image_path: Path | str) -> bytes:
    """
    Function to encode image data as byte strings
    
    Parameters
    ----------
    image_path : pathlib.Path, str
        file location where image will be loaded from
    
    Returns
    -------
    encoded_image : bytes
    
    """

    with open(image_path, "rb") as f:
        return base64.b64encode(f.read())
    

    

