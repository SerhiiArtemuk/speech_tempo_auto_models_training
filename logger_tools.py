import sys
import logging
import os
import logging.handlers as handlers


from logger_config import config_io


def setup_logger(name):

    os.makedirs('logs', exist_ok=True)

    logger = logging.getLogger(name)
    urllib_log = logging.getLogger('urllib3')
    urllib_log.setLevel(logging.CRITICAL)
    logger.setLevel(config_io.logger.level)
    logger.propagate = False
    formatter = logging.Formatter(config_io.logger.format)

    log_handler = handlers.TimedRotatingFileHandler(config_io.logger.full_logs.path,
                                                    when=config_io.logger.full_logs.save_time,
                                                    backupCount=config_io.logger.full_logs.backup_count)

    if name.startswith('test_'):
        log_handler = logging.FileHandler(config_io.logger.test_logs.path + f'{name}.log')
        log_handler.setLevel(logging.INFO)

    log_handler.setFormatter(formatter)
    error_log_handler = handlers.TimedRotatingFileHandler(config_io.logger.error_logs.path,
                                                     # maxBytes=config_io.logger.error_logs.backup_count,
                                                     backupCount=config_io.logger.error_logs.max_bytes)

    error_log_handler.setLevel(logging.ERROR)
    error_log_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)

    logger.addHandler(log_handler)
    logger.addHandler(error_log_handler)
    logger.addHandler(stream_handler)
    return logger