# -*- coding: utf-8 -*-
import logging


def create_logger(name,
                  level='info',
                  fmt='%(asctime)s %(levelname)s %(name)s: %(message)s',
                  datefmt='%y-%m-%d %H:%M:%S',
                  add_console_handler=True,
                  add_file_handler=False,
                  logfile='/tmp/tmp.log'):
    """Create a formatted logger at module level
    
    :param fmt: Format of the log message
    :param datefmt: Datetime format of the log message
    :example:
    logger = create_logger(__name__, level='info')
    logger.info('Hello world')
    """
    level = {
        'debug': logging.DEBUG, 'info': logging.INFO,
        'warn': logging.WARN,   'error': logging.ERROR
        }[level]

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logFmt = logging.Formatter(fmt=fmt, datefmt=datefmt)

    if add_console_handler:  # Print on console
        ch = logging.StreamHandler()
        ch.setFormatter(logFmt)
        logger.addHandler(ch)

    if add_file_handler:  # Print in a log file
        th = logging.RotatingFileHandler(logfile, backupCount=5)
        th.doRollover()
        th.setFormatter(logFmt)
        logger.addHandler(th)
    
    return logger