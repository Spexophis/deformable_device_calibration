import logging
import datetime
import os

def setup_logger(log_path):
    today = datetime.datetime.now()
    log_name = today.strftime("%Y%m%d_%H%M.log")
    log_file = os.path.join(log_path, log_name)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filename=log_file,
                        filemode='w')
    logger = logging.getLogger('shared_logger')
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.info("Logger initialized.")
    return logger


def setup_logging():
    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
    return logging
