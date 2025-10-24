import logging
import os
from datetime import datetime

def get_logger(name: str, log_file: str = None) -> logging.Logger:

    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(log_dir, exist_ok=True)

    if log_file is None:
        log_file = f"data_pipeline_{datetime.now().strftime('%Y%m%d')}.log"
    log_path = os.path.join(log_dir, log_file)

    if name in logging.Logger.manager.loggerDict:
        logging.getLogger(name).handlers.clear()

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Console handler
    c_handler = logging.StreamHandler()
    c_handler.setLevel(logging.INFO)

    # File handler
    f_handler = logging.FileHandler(log_path)
    f_handler.setLevel(logging.DEBUG)

    # Formatter
    log_format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
    formatter = logging.Formatter(log_format, datefmt='%Y-%m-%d %H:%M:%S')
    c_handler.setFormatter(formatter)
    f_handler.setFormatter(formatter)

    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    return logger