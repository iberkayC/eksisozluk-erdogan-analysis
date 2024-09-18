import logging
from pathlib import Path


def setup_logging(log_file: str = "project.log") -> None:
    """
    Sets up logging configuration for the project.

    Args:
        log_file (str, optional): Name of the log file.
            Defaults to "project.log".

    Returns:
        None
    """
    log_dir = Path(__file__).parents[2] / "logs"
    log_dir.mkdir(exist_ok=True)

    log_path = log_dir / log_file

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        filename=str(log_path),
        filemode="a",
    )


def get_logger(name: str) -> logging.Logger:
    """
    Retrieves a logger instance with the specified name.

    Args:
        name (str): Name of the logger.

    Returns:
        logging.Logger: Configured logger instance.
    """
    return logging.getLogger(name)
