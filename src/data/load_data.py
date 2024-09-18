import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def load_data(file_path: str) -> pd.DataFrame:
    """
    Loads data from a JSON file and returns it as a pandas DataFrame.

    The JSON file is expected to contain a list of dictionaries, where each
    dictionary represents a single entry with 'Content', 'Author',
    'Date Created', and 'Last Changed' fields.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        pd.DataFrame: DataFrame containing the loaded data.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        logger.error(f"Data not found at {file_path}")
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    try:
        with file_path.open("r", encoding="utf-8") as file:
            data: List[Dict] = json.load(file)
        logger.info(f"Successfully loaded data from {file_path}")
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON file: {file_path}")
        raise json.JSONDecodeError(
            f"The file {file_path} is not valid JSON: {str(e)}", e.doc, e.pos
        )
    except Exception as e:
        logger.exception(
            f"An unexpected error occurred while loading data: {e}"
        )
        raise e

    df = pd.DataFrame(data)
    logger.info(
        f"Created DataFrame with {len(df)} rows and {len(df.columns)} columns"
    )
    return df


if __name__ == "__main__":
    from src.utils.logging_config import setup_logging

    setup_logging()

    df = load_data(r"data/raw/erdogan_thread.json")
    print(df.head())
    print(f"Total number of posts: {len(df)}")
