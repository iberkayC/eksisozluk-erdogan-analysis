from typing import Dict, Optional, Tuple, List, Union
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

from src.visualization.visualize import save_plot
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def calculate_basic_stats(df: pd.DataFrame) -> dict:
    """
    Calculate basic statistics of the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        dict: Dictionary containing basic statistics.
    """
    stats = {
        "total_posts": len(df),
        "unique_authors": df["Author"].nunique(),
        "avg_content_length": df["content_length"].mean(),
        "median_content_length": df["content_length"].median(),
        "avg_posts_per_day": df.shape[0]
        / (df["Date Created"].max() - df["Date Created"].min()).days,
    }
    logger.debug(f"Calculated basic stats: {stats}")
    return stats


def get_date_range(df: pd.DataFrame) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """
    Calculates the date range of the DataFrame based on 'Date Created'.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        Tuple[pd.Timestamp, pd.Timestamp]: Start and end dates.
    """
    start_date = df["Date Created"].min()
    end_date = df["Date Created"].max()
    logger.debug(f"Date range from {start_date} to {end_date}")
    return start_date, end_date


def analyze_post_edits(df: pd.DataFrame) -> Dict[str, Union[int, float]]:
    """
    Analyzes how many posts were edited after their initial creation.

    Args:
        df (pd.DataFrame): Input DataFrame containing 'Date Created'
            and 'Last Changed' columns.

    Returns:
        dict: A dictionary containing:
            - total_posts: Total number of posts
            - edited_posts: Number of posts that were edited
            - nonedited_posts: Number of posts that were not edited
            - edit_rate: Percentage of posts that were edited
    """
    total_posts = len(df)
    edited_posts = df["Last Changed"].notna().sum()
    nonedited_posts = total_posts - edited_posts
    edit_rate = (edited_posts / total_posts) * 100

    results = {
        "total_posts": total_posts,
        "edited_posts": edited_posts,
        "nonedited_posts": nonedited_posts,
        "edit_rate": edit_rate,
    }

    logger.info(f"Post edit analysis: {results}")
    return results


def detect_anomalies(df, save_path: Optional[str] = None) -> None:
    """
    Detects anomalies in posting frequency using DBSCAN clustering.

    Args:
        df (pd.DataFrame): Input DataFrame.
        save_path (Optional[str], optional): Path to save anomaly data and plot
        Defaults to None.

    Returns:
        None
    """
    logger.info("Starting anomaly detection")

    # group data by month and count posts
    df_grouped = (
        df.groupby(df["Date Created"].dt.to_period("M"))
        .size()
        .reset_index(name="count")
    )
    df_grouped["Date Created"] = df_grouped["Date Created"].dt.to_timestamp()

    X = df_grouped[["Date Created", "count"]].values
    # convert dates to unix timestamps
    X[:, 0] = (
        np.vectorize(lambda x: x.timestamp())(X[:, 0]).astype(np.int64)
        // 10**9
    )

    dbscan = DBSCAN(eps=2, min_samples=2)
    clusters = dbscan.fit_predict(X)

    anomalies = df_grouped[clusters == -1]

    logger.info(f"Detected {len(anomalies)} anomalies in posting frequency")
    if save_path:
        anomalies_file = Path(save_path) / "data/anomalies.txt"
        anomalies_file.parent.mkdir(parents=True, exist_ok=True)
        with anomalies_file.open("w") as f:
            for _, row in anomalies.iterrows():
                f.write(
                    f"Date: {row['Date Created']}, Count: {row['count']}\n"
                )
        logger.info(f"Saved anomalies data to {anomalies_file}")

    plt.figure(figsize=(12, 6))
    plt.scatter(
        df_grouped["Date Created"],
        df_grouped["count"],
        c=clusters,
        cmap="viridis",
    )
    plt.scatter(
        anomalies["Date Created"], anomalies["count"], color="red", s=50
    )
    plt.title("Posting Frequency Anomalies (DBSCAN)")
    plt.xlabel("Date")
    plt.ylabel("Number of Posts")
    if save_path:
        plot_path = Path(save_path) / "figures/posting_frequency_anomalies.png"
        save_plot(plt.gcf(), str(plot_path))
    plt.show()
    plt.close()

    print("Detected anomalies:")
    print(anomalies)


def get_top_n_items(
    df: pd.DataFrame,
    column: str,
    n: int = 10,
    stopwords: Optional[List[str]] = None,
) -> pd.Series:
    """
    Retrieves the top N most frequent items in a specified DataFrame column.

    Args:
        df (pd.DataFrame): Input DataFrame.
        column (str): Column name to analyze.
        n (int, optional): Number of top items to return. Defaults to 10.
        stopwords (Optional[List[str]], optional): List of words to exclude
            from the analysis. Defaults to None.


    Returns:
        pd.Series: Series of top N items and their counts.
    """
    if stopwords is None:
        stopwords = []

    # Filter out stopwords if the column is of string type
    if df[column].dtype == "object":
        filtered_series = df[column][~df[column].isin(stopwords)]
    else:
        filtered_series = df[column]

    top_items = filtered_series.value_counts().head(n)
    logger.debug(f"Top {n} items in column '{column}': {top_items}")
    return top_items
