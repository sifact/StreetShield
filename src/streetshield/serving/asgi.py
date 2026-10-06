"""ASGI entry point for production servers: `uvicorn streetshield.serving.asgi:app`.

The config file comes from STREETSHIELD_CONFIG (e.g. configs/azure.yaml), defaulting to configs/default.yaml.
"""

import os

from streetshield.config import load_config
from streetshield.logging_utils import setup_logging
from streetshield.serving.app import create_app

setup_logging()
app = create_app(load_config(os.environ.get("STREETSHIELD_CONFIG")))
