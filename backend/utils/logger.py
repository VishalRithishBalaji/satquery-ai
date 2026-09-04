import logging
from ..config.settings import settings

def get_logger(name: str):
    logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))
    return logging.getLogger(name)
