import os
import logging
from datetime import datetime


def init_logger(log_dir, log_name):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_name)

    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt=log_format, datefmt=date_format)

    logger = logging.getLogger("LoRA-SF")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    logger.info(f"saving log to {log_path}")
    logger.info(f"----------------------------------------")
    logger.info(f"Experiment starte time : {datetime.now().strftime(date_format)}")
    return logger
