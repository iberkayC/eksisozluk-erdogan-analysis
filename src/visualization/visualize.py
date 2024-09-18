from typing import List, Optional, Tuple
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from wordcloud import WordCloud

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def create_base_plots(
    title: str, xlabel: str, ylabel: str, figsize: Tuple[int, int] = (12, 6)
) -> Tuple[plt.Figure, go.Figure]:
    """
    Create base Matplotlib and Plotly figures with common properties.

    Args:
        title (str): Title for both plots
        xlabel (str): Label for x-axis
        ylabel (str): Label for y-axis
        figsize (Tuple[int, int]): Figure size for Matplotlib plot

    Returns:
        Tuple[plt.Figure, go.Figure]: Matplotlib and Plotly figures
    """
    logger.debug("Creating base plots")
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    plotly_fig = go.Figure()
    plotly_fig.update_layout(
        title=title, xaxis_title=xlabel, yaxis_title=ylabel
    )

    return fig, plotly_fig


def create_posting_frequency_plot(
    df: pd.DataFrame,
) -> Tuple[plt.Figure, go.Figure]:
    """
    Creates posting frequency plots using both Matplotlib and Plotly.

    Args:
        df (pd.DataFrame): Input DataFrame containing 'Date Created' column.

    Returns:
        Tuple[plt.Figure, go.Figure]:
            - Matplotlib Figure object.
            - Plotly Figure object for interactive visualization.
    """
    logger.info("Creating posting frequency plot")
    df_grouped = (
        df.groupby(df["Date Created"].dt.to_period("M"))
        .size()
        .reset_index(name="count")
    )
    df_grouped["Date Created"] = df_grouped["Date Created"].dt.to_timestamp()

    fig, plotly_fig = create_base_plots(
        title="Posting Frequency Over Time",
        xlabel="Date",
        ylabel="Number of Posts",
    )

    ax = fig.gca()
    ax.plot(df_grouped["Date Created"], df_grouped["count"])
    plt.xticks(rotation=45)

    plotly_fig.add_trace(
        go.Scatter(
            x=df_grouped["Date Created"], y=df_grouped["count"], mode="lines"
        )
    )

    return fig, plotly_fig


def create_content_length_distribution_plot(
    df: pd.DataFrame, iqr: Optional[bool] = False
) -> Tuple[plt.Figure, go.Figure]:
    """
    Creates content length distribution plots using Matplotlib and Plotly.

    Args:
        df (pd.DataFrame): Input DataFrame containing 'content_length' column.
        iqr (bool, optional): Whether to exclude outliers using the
            interquartile range. Defaults to False.

    Returns:
        Tuple[plt.Figure, go.Figure]:
            - Matplotlib Figure object.
            - Plotly Figure object for interactive visualization.
    """
    if iqr:
        q1 = df["content_length"].quantile(0.25)
        q3 = df["content_length"].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        df = df[
            (df["content_length"] >= lower_bound)
            & (df["content_length"] <= upper_bound)
        ]

    logger.info("Creating content length distribution plot, IQR: %s", iqr)

    fig, plotly_fig = create_base_plots(
        title="Distribution of Content Length",
        xlabel="Content Length",
        ylabel="Frequency",
    )

    ax = fig.gca()
    sns.histplot(data=df, x="content_length", bins=50, kde=True, ax=ax)

    plotly_fig.add_trace(go.Histogram(x=df["content_length"], nbinsx=50))

    return fig, plotly_fig


def create_word_cloud(text: str, stop_words: List[str]) -> plt.Figure:
    """
    Creates a word cloud from the input text.

    Args:
        text (str): Input text for generating the word cloud.
        stop_words (List[str]): List of stop words to exclude.

    Returns:
        plt.Figure: Matplotlib figure containing the word cloud.
    """
    logger.info("Creating word cloud")

    wordcloud = WordCloud(
        width=800, height=400, background_color="white", stopwords=stop_words
    ).generate(text)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("Most Common Words")

    return fig


def analyze_date_range(
    df: pd.DataFrame, start_date: str, end_date: str
) -> None:
    """
    Analyzes posting frequency within a specified date range.

    Args:
        df (pd.DataFrame): Input DataFrame.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.

    Returns:
        None
    """
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    mask = (df["Date Created"] >= start_date) & (
        df["Date Created"] <= end_date
    )
    df_range = df.loc[mask]

    df_grouped = (
        df_range.groupby(df_range["Date Created"].dt.to_period("D"))
        .size()
        .reset_index(name="count")
    )
    df_grouped["Date Created"] = df_grouped["Date Created"].dt.to_timestamp()

    plt.figure(figsize=(12, 6))
    plt.plot(df_grouped["Date Created"], df_grouped["count"])
    plt.title(f"Posting Frequency from {start_date} to {end_date}")
    plt.xlabel("Date")
    plt.ylabel("Number of Posts")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    plt.close()

    print(f"Total posts in this period: {df_range.shape[0]}")
    print(
        "Average posts per day: "
        + f"{df_range.shape[0] / (end_date - start_date).days:.2f}"
    )


def create_author_activity_plot(
    df: pd.DataFrame,
    top_n: int = 20,
    exclude_authors: Optional[List[str]] = None,
) -> Tuple[plt.Figure, go.Figure]:
    """
    Creates author activity plots showing the most active authors.

    Args:
        df (pd.DataFrame): Input DataFrame containing 'Author' column.
        top_n (int, optional): Number of top authors to show. Defaults to 20.
        exclude_authors (Optional[List[str]], optional): List of authors to
                                                         exclude.

    Returns:
        Tuple[plt.Figure, go.Figure]:
            - Matplotlib Figure object.
            - Plotly Figure object for interactive visualization.
    """
    if exclude_authors:
        df = df[~df["Author"].isin(exclude_authors)]

    logger.info(f"Creating top {top_n} author activity plot")
    author_counts = df["Author"].value_counts().head(top_n)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=author_counts.index, y=author_counts.values, ax=ax)
    ax.set_title(f"Top {top_n} Most Active Authors*")

    if exclude_authors:
        # add footnote to the plot if authors are excluded
        ax.annotate(
            "*excluding authors: " + ", ".join(exclude_authors),
            xy=(0.99, 0.97),
            xycoords="axes fraction",
            ha="right",
            va="center",
            fontsize=10,
        )
    ax.set_xlabel("Author")
    ax.set_ylabel("Number of Posts")
    plt.xticks(rotation=90)

    plotly_fig = px.bar(
        x=author_counts.index,
        y=author_counts.values,
        labels={"x": "Author", "y": "Number of Posts"},
        title=f"Top {top_n} Most Active Authors",
    )
    plotly_fig.update_layout(xaxis_tickangle=-45)

    return fig, plotly_fig


def create_posting_patterns_plot(
    df: pd.DataFrame,
) -> Tuple[plt.Figure, go.Figure]:
    """
    Creates plots to analyze posting patterns by day of week and hour of day.

    Args:
        df (pd.DataFrame): Input DataFrame containing 'day_of_week'
                           and 'hour' columns.

    Returns:
        Tuple[plt.Figure, go.Figure]:
            - Matplotlib Figure object.
            - Plotly Figure object for interactive visualization.
    """
    logger.info("Creating posting patterns plot")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))

    sns.countplot(
        data=df,
        x="day_of_week",
        order=[
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ],
        ax=ax1,
    )
    ax1.set_title("Posting Frequency by Day of Week")
    ax1.set_xlabel("Day of Week")
    ax1.set_ylabel("Number of Posts")
    ax1.tick_params(axis="x", rotation=45)

    sns.countplot(data=df, x="hour", ax=ax2)
    ax2.set_title("Posting Frequency by Hour of Day")
    ax2.set_xlabel("Hour of Day")
    ax2.set_ylabel("Number of Posts")

    plt.tight_layout()

    plotly_fig = go.Figure()
    for day in [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]:
        day_data = df[df["day_of_week"] == day]
        plotly_fig.add_trace(
            go.Histogram(x=day_data["hour"], name=day, nbinsx=24)
        )

    plotly_fig.update_layout(
        title="Posting Patterns by Day of Week and Hour",
        xaxis_title="Hour of Day",
        yaxis_title="Number of Posts",
        barmode="group",
    )

    return fig, plotly_fig


def create_posting_activity_heatmap(
    df: pd.DataFrame,
) -> Tuple[plt.Figure, go.Figure]:
    """
    Creates a heatmap of posting activity by day of
    the week and hour of the day.

    Args:
        df (pd.DataFrame): Input DataFrame containing
                           'day_of_week' and 'hour' columns.

    Returns:
        Tuple[plt.Figure, go.Figure]:
            - Matplotlib Figure object containing the heatmap.
            - Plotly Figure object for interactive visualization.
    """
    logger.info("Creating posting activity heatmap")
    # Order days of the week for better visualization
    df["day_of_week"] = pd.Categorical(
        df["day_of_week"],
        categories=[
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ],
        ordered=True,
    )

    pivot_table = df.pivot_table(
        index="day_of_week",
        columns="hour",
        values="Content",
        aggfunc="count",
        fill_value=0,
        observed=False,
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(pivot_table, cmap="YlGnBu", ax=ax)
    ax.set_title("Posting Activity Heatmap")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Day of Week")
    plt.tight_layout()

    plotly_fig = px.imshow(
        pivot_table,
        labels=dict(x="Hour of Day", y="Day of Week", color="Number of Posts"),
        x=pivot_table.columns,
        y=pivot_table.index,
        title="Posting Activity Heatmap",
    )
    plotly_fig.update_xaxes(side="top")

    return fig, plotly_fig


def create_content_length_over_time_plot(
    df: pd.DataFrame,
) -> Tuple[plt.Figure, go.Figure]:
    """
    Creates plots showing average content length over time.

    Args:
        df (pd.DataFrame): Input DataFrame containing 'Date Created'
                           and 'content_length' columns.

    Returns:
        Tuple[plt.Figure, go.Figure]:
            - Matplotlib Figure object.
            - Plotly Figure object for interactive visualization.
    """
    logger.info("Creating content length over time plot")
    df_grouped = (
        df.groupby(df["Date Created"].dt.to_period("M"))
        .agg({"content_length": "mean"})
        .reset_index()
    )
    df_grouped["Date Created"] = df_grouped["Date Created"].dt.to_timestamp()

    fig, plotly_fig = create_base_plots(
        title="Average Content Length Over Time",
        xlabel="Date",
        ylabel="Average Content Length",
    )

    ax = fig.gca()
    ax.plot(df_grouped["Date Created"], df_grouped["content_length"])
    plt.xticks(rotation=45)

    plotly_fig.add_trace(
        go.Scatter(
            x=df_grouped["Date Created"],
            y=df_grouped["content_length"],
            mode="lines",
        )
    )

    return fig, plotly_fig


def save_plot(fig: plt.Figure, save_path: str) -> None:
    """
    Saves a Matplotlib figure to a file, ensuring the directory exists.

    Args:
        fig (plt.Figure): Matplotlib figure to save.
        save_path (str): Path to save the figure.

    Returns:
        None
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved plot to {save_path}")


def save_plotly_figure(fig: go.Figure, html_path: str) -> None:
    """
    Saves a Plotly figure to an HTML file, ensuring the directory exists.

    Args:
        fig (go.Figure): Plotly figure to save.
        html_path (str): Path to save the HTML file.

    Returns:
        None
    """
    html_path = Path(html_path)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(html_path)
    logger.info(f"Saved interactive plot to {html_path}")
