import logging

logger = logging.getLogger("")
ConsoleOutputHandler = logging.StreamHandler()

formatter = logging.Formatter("%(asctime)s;%(levelname)s;%(message)s")
ConsoleOutputHandler.setFormatter(formatter)

logger.addHandler(ConsoleOutputHandler)
logger.setLevel(logging.INFO)