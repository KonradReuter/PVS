import logging
import logging.config
import sys
from pathlib import Path
import torch
from rich.logging import RichHandler
import json

BASE_DIR = Path(__file__).parent.parent.absolute()
CONFIG_DIR = Path(BASE_DIR, "config")
DATA_DIR = Path("/data/Reuter")  #Path(BASE_DIR, "data")
LOGS_DIR = Path(BASE_DIR, "logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)
#CHECKPOINT_DIR = Path(BASE_DIR, "checkpoints")
CHECKPOINT_DIR = Path("/data/Reuter/checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
ATT_MAP_DIR = Path(BASE_DIR, "attention_maps")
ATT_MAP_DIR.mkdir(parents=True, exist_ok=True)
PRED_DIR = Path(BASE_DIR, "predictions")
PRED_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args = json.load(open("./config/args.json"))

logging_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "minimal": {"format": "%(messages)s"},
        "detailed": {
            "format": "%(levelname)s %(asctime)s [%(name)s:%(filename)s:%(funcName)s:%(lineno)d]\n%(message)s\n"
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "stream": sys.stdout,
            "formatter": "minimal",
            "level": logging.DEBUG,
        },
        "info": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": Path(LOGS_DIR, "info.log"),
            "maxBytes": 10485760,
            "backupCount": 10,
            "formatter": "detailed",
            "level": logging.INFO,
        },
        "error": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": Path(LOGS_DIR, "error.log"),
            "maxBytes": 10485760,
            "backupCount": 10,
            "formatter": "detailed",
            "level": logging.ERROR,
        },
    },
    "root": {"handlers": ["console", "info", "error"], "level": logging.INFO, "propagate": True},
}

logging.config.dictConfig(logging_config)
logger = logging.getLogger()
logger.handlers[0] = RichHandler(markup=True)
