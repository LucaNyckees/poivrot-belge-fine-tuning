import logging
from logging import Logger
import sys


def setup_log(name: str, notebook: bool = False) -> Logger:
    logger = logging.getLogger(name)
    formatter = logging.Formatter("[%(name)s %(asctime)s - %(levelname)s]: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = notebook
    return logger
