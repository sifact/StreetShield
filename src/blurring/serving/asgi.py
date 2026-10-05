"""ASGI entry point for production servers: `uvicorn blurring.serving.asgi:app`.

The config file comes from BLURRING_CONFIG (e.g. configs/azure.yaml), defaulting to configs/default.yaml.
"""

import os

from blurring.config import load_config
from blurring.logging_utils import setup_logging
from blurring.serving.app import create_app

setup_logging()
app = create_app(load_config(os.environ.get("BLURRING_CONFIG")))
