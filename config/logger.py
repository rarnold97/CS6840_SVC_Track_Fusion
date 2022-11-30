import logging
from config.settings import PROJECT_ROOT

__all__ = ["LOGGER"]

def get_logger()->logging.Logger:
    logger = logging.getLogger('cs6840_final_project')

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    fh = logging.FileHandler(PROJECT_ROOT/'cs6840_final_project.log')
    console = logging.StreamHandler()

    fh.setFormatter(formatter)
    console.setFormatter(formatter)

    logger.addHandler(console)
    logger.addHandler(fh)
    logger.setLevel(logging.DEBUG)
    return logger

LOGGER = get_logger()