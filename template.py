from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="[%(asctime)s]: %(message)s")

project_name = "FarmInsectsClassifier"

required_files = [
    ".github/workflows/.gitkeep",
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/logger.py",
    f"src/{project_name}/exception.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/tests/",
    f"src/{project_name}/utils.py",
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/configuration.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/constants/__init__.py",
    f"src/{project_name}/torch/utils.py",
    f"src/{project_name}/torch/components/__init__.py",
    f"src/{project_name}/torch/tests/",
    f"src/{project_name}/torch/config/__init__.py",
    f"src/{project_name}/torch/config/configuration.py",
    f"src/{project_name}/torch/pipeline/__init__.py",
    f"src/{project_name}/torch/entity/__init__.py",
    "config/config.yaml",
    "dvc.yaml",
    "params.yaml",
    "requirements.txt",
    "setup.py",
    "main.py",
    "notebook"
]

for file_path in required_files:
    file_path = Path(file_path)

    # create empty directories
    if file_path.suffix == "" and file_path.parent != Path(".github/workflows"):
        file_dir = file_path.mkdir(parents=True, exist_ok=True) 

    # split path into directory and filename
    file_dir, file_name = file_path.parent, file_path.name 
    

    if file_dir:
        dir_path = Path(file_dir)
        dir_path.mkdir(parents=True, exist_ok=True)
        logging.info(f"Creating directory: {file_dir} for the file {file_name}")

    if not file_path.exists() or file_path.stat().st_size == 0:
        file_path.touch()
        logging.info(f"Creating empty file: {file_path}")

    else:
        logging.info(f"{file_name} already exists in {file_dir}")






