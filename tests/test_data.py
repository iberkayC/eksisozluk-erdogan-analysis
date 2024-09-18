import pytest
import pandas as pd
import json
from src.data.load_data import load_data


def test_load_data_success(sample_json_file):
    """
    Test successful loading of data from a JSON file.
    """
    df = load_data(sample_json_file)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 100
    assert list(df.columns) == [
        "Content",
        "Author",
        "Date Created",
        "Last Changed",
    ]


def test_load_data_file_not_found():
    """
    Test that FileNotFoundError is raised when file doesn't exist.
    """
    with pytest.raises(FileNotFoundError):
        load_data("non_existent_file.json")


def test_load_data_corrupted_json(corrupted_json_file):
    """
    Test that JSONDecodeError is raised when JSON is corrupted.
    """
    with pytest.raises(json.JSONDecodeError) as exc_info:
        load_data(corrupted_json_file)
    assert "is not valid JSON" in str(exc_info.value)


def test_load_data_empty_file(empty_json_file):
    """
    Test loading an empty file.
    """
    with pytest.raises(json.JSONDecodeError) as exc_info:
        load_data(empty_json_file)
    assert "Expecting value" in str(exc_info.value)


def test_load_data_large_file(large_json_file):
    """
    Test loading a large file to ensure it can handle larger datasets.
    """
    df = load_data(large_json_file)
    assert len(df) == 10000


def test_load_data_content(sample_json_file):
    """
    Test that the loaded data contains expected content.
    """
    df = load_data(sample_json_file)
    expected_words = [
        "erdogan",
        "turkiye",
        "siyaset",
        "ekonomi",
        "akp",
        "chp",
        "mhp",
        "hdp",
    ]
    assert df["Content"].str.contains("|".join(expected_words)).any()


def test_load_data_date_format(sample_json_file):
    """
    Test that the date columns are in the correct format.
    """
    df = load_data(sample_json_file)
    date_pattern = r"\d{2}\.\d{2}\.\d{4} \d{2}:\d{2}"
    assert df["Date Created"].str.match(date_pattern).all()
    assert df["Last Changed"].str.match(date_pattern).all()


def test_load_data_author_uniqueness(sample_json_file):
    """
    Test that there are multiple unique authors in the dataset.
    """
    df = load_data(sample_json_file)
    assert df["Author"].nunique() > 1


def test_load_data_chronological_order(sample_json_file):
    """
    Test that the 'Date Created' is always before or equal to 'Last Changed'.
    """
    df = load_data(sample_json_file)
    df["Date Created"] = pd.to_datetime(
        df["Date Created"], format="%d.%m.%Y %H:%M"
    )
    df["Last Changed"] = pd.to_datetime(
        df["Last Changed"], format="%d.%m.%Y %H:%M"
    )
    assert (df["Last Changed"] >= df["Date Created"]).all()
