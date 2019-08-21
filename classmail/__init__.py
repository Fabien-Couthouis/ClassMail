import os
from . import classification
from . import data_visualisation
from . import nlp
import logging.config

__version__ = "0.1"

logging.config.dictConfig(
    {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {"standard": {"format": "%(asctime)-15s %(message)s"}},
        "handlers": {
            "console": {
                "level": "INFO",
                "class": "logging.StreamHandler",
                "formatter": "standard",
                "stream": "ext://sys.stdout",
            }
        },
        "loggers": {
            "classmail": {"handlers": ["console"], "level": "INFO", "propagate": False}
        },
        "root": {"handlers": ["console"], "level": "WARNING"},
    }
)

logger = logging.getLogger("classmail")
