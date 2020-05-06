'''
VCS logger.
'''

import os
import ast
import logging
import pathlib
import uuid
import time
from pythonjsonlogger import jsonlogger



class MetaFilter(logging.Filter):
    """
    Add meta field to log if one doesn't exist.
    """
    def filter(self, record):
        if not hasattr(record, 'meta'):
            record.meta = {}
        return True


class CustomJsonFormatter(jsonlogger.JsonFormatter):
    """
    Create custom JSON formatter.
    """
    def add_fields(self, log_record, record, message_dict):
        super(CustomJsonFormatter, self).add_fields(log_record, record, message_dict)
        # rename fields
        log_record['logger'] = record.name
        log_record['level'] = record.levelname


def init_logger():
    """
    setup a logger for all the events in the system.
    """

    job_id = get_job_id()

    logger = logging.getLogger(job_id)
    logger.addFilter(MetaFilter())
    logger.setLevel(logging.DEBUG)

    enable_console_logger = ast.literal_eval(os.getenv('ENABLE_CONSOLE_LOGGER', 'True'))
    if enable_console_logger:
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO) # https://docs.python.org/3/library/logging.html#logging-levels
        stream_formatter = logging.Formatter('[%(asctime)-15s] %(levelname)-8s: %(message)s %(meta)s')
        stream_handler.setFormatter(stream_formatter)
        logger.addHandler(stream_handler)

    enable_file_logger = ast.literal_eval(os.getenv('ENABLE_FILE_LOGGER', 'True'))
    if enable_file_logger:
        log_files_directory = os.getenv('LOG_FILES_DIRECTORY')
        pathlib.Path(log_files_directory).mkdir(parents=True, exist_ok=True)
        file_name = os.getenv('LOG_FILENAME', default = job_id + '.log')
        file_path = os.path.join(log_files_directory, file_name)
        file_handler = logging.FileHandler(file_path)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = CustomJsonFormatter('(created) (logger) (level) (message)')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)


def get_logger():
    """
    Fetch logger.
    """
    return logging.getLogger(get_job_id())


def get_job_id():
    """
    Fetch job id or create new if one doesn't already exist.
    """
    if os.getenv('JOB_ID') is None:
        os.environ['JOB_ID'] = 'job_' + str(int(time.time())) + '_' + uuid.uuid4().hex
    return os.getenv('JOB_ID')
