import re
from pathlib import Path
from typing import Optional

import pandas as pd

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def preprocess_data(
    df: pd.DataFrame, save_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Preprocesses the input DataFrame by cleaning and transforming data.

    This function performs several preprocessing steps:
    - Converts date columns to datetime objects.
    - Removes rows with missing 'Date Created' values.
    - Sorts data by 'Date Created'.
    - Removes links from the 'Content' column.
    - Converts text to lowercase and removes extra whitespace.
    - Adds additional columns for analysis.

    Args:
        df (pd.DataFrame): Input DataFrame containing raw data.
        save_path (Optional[str], optional): Path to save the preprocessed
            DataFrame. Defaults to None.

    Returns:
        pd.DataFrame: The preprocessed DataFrame.
    """
    logger.info("Starting data preprocessing")
    df_cleaned = df.copy()

    df_cleaned = convert_dates(df_cleaned)
    df_cleaned = clean_content(df_cleaned)
    df_cleaned = add_additional_columns(df_cleaned)

    if save_path:
        save_processed_data(df_cleaned, save_path)

    logger.info("Preprocessing completed")
    return df_cleaned


def convert_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts date columns to datetime objects and handles missing values.

    Args:
        df (pd.DataFrame): DataFrame with raw date columns.

    Returns:
        pd.DataFrame: DataFrame with converted date columns.
    """
    date_columns = ["Date Created", "Last Changed"]
    for col in date_columns:
        df[col] = pd.to_datetime(
            df[col], format="%d.%m.%Y %H:%M", errors="coerce"
        )
    logger.debug("Converted date columns to datetime")

    # Remove rows with missing 'Date Created' values
    # there probably aren't any, but just in case
    initial_row_count = len(df)
    df = df.dropna(subset=["Date Created"])
    rows_dropped = initial_row_count - len(df)
    logger.debug(
        f"Removed {rows_dropped} rows with missing 'Date Created' values"
    )

    # Sort by 'Date Created'
    # probably not necessary, but just in case
    df = df.sort_values("Date Created").reset_index(drop=True)
    logger.debug("Sorted DataFrame by 'Date Created'")

    return df


def clean_content(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the 'Content' column by removing links and normalizing text.

    Args:
        df (pd.DataFrame): DataFrame with the 'Content' column.

    Returns:
        pd.DataFrame: DataFrame with cleaned 'Content' column.
    """
    df["Content"] = df["Content"].apply(remove_links)
    logger.debug("Removed links from 'Content'")

    # Remove extra whitespace and convert to lowercase
    df["Content"] = df["Content"].apply(lambda x: " ".join(x.split()))
    df["Content"] = df["Content"].str.lower()
    logger.debug("Normalized 'Content' text")

    return df


def add_additional_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds additional columns needed for analysis.

    Args:
        df (pd.DataFrame): The DataFrame to modify.

    Returns:
        pd.DataFrame: The DataFrame with new columns added.
    """
    df["content_length"] = df["Content"].str.len()
    df["day_of_week"] = df["Date Created"].dt.day_name()
    df["hour"] = df["Date Created"].dt.hour
    logger.debug("Added 'content_length', 'day_of_week', and 'hour' columns")

    return df


def remove_links(text: str) -> str:
    """
    Remove links from the given text.

    Args:
        text (str): Input text.

    Returns:
        str: Text with links removed.
    """
    url_pattern = re.compile(
        r"(https?://)?(www\.)?\S+?\.\S*(?:\.{3})?[^\s(),]*"
    )
    return url_pattern.sub("", text)


def save_processed_data(df: pd.DataFrame, file_path: str) -> None:
    """
    Save the processed DataFrame to a file.

    Args:
        df (pd.DataFrame): Processed DataFrame to save.
        file_path (str): Path to save the processed data.

    Raises:
        ValueError: If the file extension is not supported.
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    file_extension = file_path.suffix.lower()

    if file_extension == ".csv":
        df.to_csv(file_path, index=False)
        logger.info(f"Saved processed data to CSV: {file_path}")
    elif file_extension == ".json":
        df_json = df.copy()
        for col in df_json.select_dtypes(include=["datetime64"]).columns:
            df_json[col] = df_json[col].dt.strftime("%Y-%m-%d %H:%M:%S")

        df_json.to_json(file_path, orient="records", date_format="iso")
        logger.info(f"Saved processed data to JSON: {file_path}")
    else:
        raise ValueError(
            f"Unsupported file extension: {file_extension}. Use .csv or .json"
        )
