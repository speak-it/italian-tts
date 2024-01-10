import logging
from sys import stdout


def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "[%(asctime)s | %(name)s | %(levelname)s] %(message)s")
    ch = logging.StreamHandler(stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger
