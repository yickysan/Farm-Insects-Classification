import logging
from pathlib import Path
from datetime import datetime

path = Path(__file__)
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
logs_path = path.cwd() /"logs" / LOG_FILE
logs_path.mkdir(parents=True, exist_ok=True)

LOG_FILE_PATH = logs_path/ LOG_FILE

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)