from easydict import EasyDict
import logging
from pathlib import Path

config_io = EasyDict()
config_io.root = Path(__file__).parent
config_io.logger = EasyDict()
config_io.logger.level = logging.DEBUG
config_io.logger.format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
config_io.logger.full_logs = EasyDict()
config_io.logger.full_logs.path = f"{config_io.root}/logs/full.log"
config_io.logger.full_logs.backup_count = 30
config_io.logger.full_logs.save_time = 'midnight'
config_io.logger.error_logs = EasyDict()
config_io.logger.error_logs.path = f"{config_io.root}/logs/error.log"
config_io.logger.error_logs.backup_count = 10
config_io.logger.error_logs.max_bytes = 15000
config_io.logger.test_logs = EasyDict()
config_io.logger.test_logs.path = f"{config_io.root}/logs/"