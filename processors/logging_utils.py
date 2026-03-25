import logging

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def get_logger(name: str) -> logging.Logger:
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    return logging.getLogger(name)


def div_file_prefix(division: str) -> str:
    return division


def division_year_label(division: str | int, year: int) -> str:
    return f"{division} {year}"
