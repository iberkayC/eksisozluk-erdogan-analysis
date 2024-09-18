# erd/tests/test_preprocess.py
import pytest
import pandas as pd
from src.data.preprocess import (
    preprocess_data,
    remove_links,
    convert_dates,
    clean_content,
    add_additional_columns,
)


@pytest.fixture
def sample_raw_df():
    return pd.DataFrame(
        {
            "Content": [
                r"bu linkte gorebilirsiniz: https://example.com www.test.com",
                r"erdogan turkiyenin cumhurbaskanidir.",
                r"uzun bir link https://www.ex... and www.ex...",
            ],
            "Author": ["User1", "User2", "User3"],
            "Date Created": [
                "01.01.2023 12:00",
                "02.01.2023 13:00",
                "03.01.2023 14:00",
            ],
            "Last Changed": [
                "01.01.2023 12:30",
                "02.01.2023 13:30",
                "03.01.2023 14:30",
            ],
        }
    )


def test_preprocess_data(sample_raw_df):
    processed_df = preprocess_data(sample_raw_df)

    assert "content_length" in processed_df.columns
    assert "day_of_week" in processed_df.columns
    assert "hour" in processed_df.columns
    assert processed_df["Date Created"].dtype == "datetime64[ns]"
    assert processed_df["Last Changed"].dtype == "datetime64[ns]"
    assert processed_df["Content"].str.islower().all()
    assert not processed_df["Content"].str.contains("http|www").any()


def test_remove_links():
    text = """
           Check out this link: https://example.com and www.test.com and
           http://www.milliyet.com.tr/…12/14/yazar/dundar.html
           www.timesonline.co.uk/…17649-1648150,00 www. is a nice protocol
           """
    cleaned_text = remove_links(text)

    assert "https://example.com" not in cleaned_text
    assert "www.test.com" not in cleaned_text
    assert "www" not in cleaned_text


def test_convert_dates(sample_raw_df):
    df = convert_dates(sample_raw_df)
    assert df["Date Created"].dtype == "datetime64[ns]"
    assert df["Last Changed"].dtype == "datetime64[ns]"


def test_clean_content(sample_raw_df):
    df = clean_content(sample_raw_df)
    assert not df["Content"].str.contains("http|www").any()
    assert df["Content"].str.islower().all()


def test_add_additional_columns(sample_raw_df):
    df = convert_dates(sample_raw_df)  # Convert dates first
    df = add_additional_columns(df)
    assert "content_length" in df.columns
    assert "day_of_week" in df.columns
    assert "hour" in df.columns
