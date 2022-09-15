import os
from pathlib import Path
from easydict import EasyDict

general_config = EasyDict()
general_config.root = Path(__file__).parent