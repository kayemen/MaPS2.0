import logging
from logging.config import dictConfig

logger_config = {
    'version': 1,
    'formatters': {
        'console': {
            'format': '[%(asctime)s]%(name)s-%(levelname)s: %(message)s'
        }
    },
    'handlers': {
        'console_handler': {
            'class': 'logging.StreamHandler',
            'formatter': 'console',
            'level': logging.DEBUG
        }
    },
    'root': {
        'handlers': ['console_handler'],
        'level': logging.DEBUG
    },
    'loggers': {

    }
}


if __name__ == '__main__':
    dictConfig(logger_config)

    logger = logging.getLogger('maps.helpers.logging_config')

    logger.debug('Testing the logger')
