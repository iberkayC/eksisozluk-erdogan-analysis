import pytest
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from src.visualization.visualize import (
    create_posting_frequency_plot,
    create_content_length_distribution_plot,
    create_word_cloud,
    create_author_activity_plot,
    create_posting_patterns_plot,
    create_posting_activity_heatmap,
    create_content_length_over_time_plot,
    save_plot,
    save_plotly_figure,
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
            "day_of_week": ["Monday", "Tuesday", "Wednesday"],
            "hour": [12, 13, 14],
        }
    )


def test_create_posting_frequency_plot(sample_processed_df):
    fig, plotly_fig = create_posting_frequency_plot(sample_processed_df)
    assert isinstance(fig, plt.Figure)
    assert isinstance(plotly_fig, go.Figure)


def test_create_content_length_distribution_plot(sample_processed_df):
    fig, plotly_fig = create_content_length_distribution_plot(
        sample_processed_df
    )
    assert isinstance(fig, plt.Figure)
    assert isinstance(plotly_fig, go.Figure)


def test_create_word_cloud():
    text = "erdogan turkey politics economy"
    stop_words = ["and", "or", "the"]
    fig = create_word_cloud(text, stop_words)
    assert isinstance(fig, plt.Figure)


def test_create_author_activity_plot(sample_processed_df):
    fig, plotly_fig = create_author_activity_plot(sample_processed_df)
    assert isinstance(fig, plt.Figure)
    assert isinstance(plotly_fig, go.Figure)


def test_create_posting_patterns_plot(sample_processed_df):
    fig, plotly_fig = create_posting_patterns_plot(sample_processed_df)
    assert isinstance(fig, plt.Figure)
    assert isinstance(plotly_fig, go.Figure)


def test_create_posting_activity_heatmap(sample_processed_df):
    fig, plotly_fig = create_posting_activity_heatmap(sample_processed_df)
    assert isinstance(fig, plt.Figure)
    assert isinstance(plotly_fig, go.Figure)


def test_create_content_length_over_time_plot(sample_processed_df):
    fig, plotly_fig = create_content_length_over_time_plot(sample_processed_df)
    assert isinstance(fig, plt.Figure)
    assert isinstance(plotly_fig, go.Figure)


def test_save_plot(tmp_path):
    fig, ax = plt.subplots()
    save_path = tmp_path / "test_plot.png"
    save_plot(fig, str(save_path))
    assert save_path.exists()


def test_save_plotly_figure(tmp_path):
    fig = go.Figure()
    save_path = tmp_path / "test_plot.html"
    save_plotly_figure(fig, str(save_path))
    assert save_path.exists()
