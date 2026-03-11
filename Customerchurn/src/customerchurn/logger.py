import logging
import os
from datetime import datetime

LOG_DIR = os.path.join(os.getcwd(), "Logs")
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE)

# Configure root logger ONCE
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE_PATH, mode="a", encoding="utf-8"),
        logging.StreamHandler()  # prints to console too
    ]
)
