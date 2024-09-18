import pytest
import pandas as pd
from src.analysis.analysis import (
    calculate_basic_stats,
    get_date_range,
    detect_anomalies,
    get_top_n_items,
)


@pytest.fixture
def sample_processed_df():
    return pd.DataFrame(
        {
            "Content": ["content1", "content2", "content3"],
            "Author": ["User1", "User2", "User1"],
            "Date Created": pd.to_datetime(
                ["2023-01-01", "2023-01-02", "2023-01-03"]
            ),
            "Last Changed": pd.to_datetime(
                ["2023-01-01", "2023-01-02", "2023-01-03"]
            ),
            "content_length": [10, 20, 15],
        }
    )


def test_calculate_basic_stats(sample_processed_df):
    stats = calculate_basic_stats(sample_processed_df)
    assert "total_posts" in stats
    assert "unique_authors" in stats
    assert "avg_content_length" in stats
    assert "median_content_length" in stats
    assert "avg_posts_per_day" in stats
    assert stats["total_posts"] == 3
    assert stats["unique_authors"] == 2
    assert stats["avg_content_length"] == 15


def test_get_date_range(sample_processed_df):
    start_date, end_date = get_date_range(sample_processed_df)
    assert start_date == pd.Timestamp("2023-01-01")
    assert end_date == pd.Timestamp("2023-01-03")


def test_detect_anomalies(sample_processed_df, tmp_path):
    detect_anomalies(sample_processed_df, save_path=str(tmp_path))
    assert (tmp_path / "data/anomalies.txt").exists()
    assert (tmp_path / "figures/posting_frequency_anomalies.png").exists()


def test_get_top_n_items(sample_processed_df):
    top_authors = get_top_n_items(sample_processed_df, "Author", n=2)
    assert len(top_authors) == 2
    assert top_authors.index[0] == "User1"
    assert top_authors.index[1] == "User2"
