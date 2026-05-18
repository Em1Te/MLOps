import logging
import sys
from typing import Text, Union


def get_console_handler() -> logging.StreamHandler:
    console_handler = logging.StreamHandler(sys.stdout)

    formatter = logging.Formatter(
        "%(asctime)s — %(name)s — %(levelname)s — %(message)s"
    )

    console_handler.setFormatter(formatter)

    return console_handler


def get_logger(
    name: Text = __name__,
    log_level: Union[Text, int] = logging.DEBUG
) -> logging.Logger:

    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    if logger.hasHandlers():
        logger.handlers.clear()

    logger.addHandler(get_console_handler())
    logger.propagate = False

    return logger