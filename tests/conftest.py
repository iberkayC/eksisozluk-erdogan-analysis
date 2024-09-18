import pytest
import pandas as pd
from datetime import datetime, timedelta
import random


def generate_random_date(start_date, end_date):
    """Generate a random date between start_date and end_date."""
    time_between_dates = end_date - start_date
    days_between_dates = time_between_dates.days
    random_number_of_days = random.randrange(days_between_dates)
    return start_date + timedelta(days=random_number_of_days)


def generate_random_content():
    """Generate random content with varying length and characteristics."""
    words = [
        "erdogan",
        "turkiye",
        "siyaset",
        "ekonomi",
        "enflasyon",
        "secim",
        "dis gucler",
        "hukumet",
        "devlet",
        "kriz",
        "reform",
        "demokrasi",
        "insan haklari",
        "adalet",
        "hukuk",
        "akp",
        "chp",
        "mhp",
        "hdp",
        "istanbul",
        "ankara",
    ]
    content_length = random.randint(5, 50)
    content = " ".join(random.choices(words, k=content_length))

    key_words = ["erdogan", "turkiye", "siyaset", "ekonomi"]
    content += " " + random.choice(key_words)

    # Randomly add URLs or hashtags
    if random.random() < 0.3:
        content += " https://example.com"
    if random.random() < 0.2:
        content += " #" + random.choice(words)
    if random.random() < 0.1:
        content += "https://www.fgdgdfg..."

    return content


@pytest.fixture
def sample_df():
    start_date = datetime(2010, 1, 1)
    end_date = datetime(2023, 12, 31)

    data = []
    for _ in range(100):  # Generate 100 entries
        created_date = generate_random_date(start_date, end_date)
        last_changed = created_date + timedelta(
            minutes=random.randint(0, 1440)
        )  # Random time within 24 hours

        entry = {
            "Content": generate_random_content(),
            "Author": f"User{random.randint(1, 1000)}",
            "Date Created": created_date.strftime("%d.%m.%Y %H:%M"),
            "Last Changed": last_changed.strftime("%d.%m.%Y %H:%M"),
        }
        data.append(entry)

    return pd.DataFrame(data)


@pytest.fixture
def sample_json_file(tmp_path, sample_df):
    file_path = tmp_path / "test_data.json"
    sample_df.to_json(file_path, orient="records", date_format="iso")
    return str(file_path)


@pytest.fixture
def corrupted_json_file(tmp_path, sample_df):
    file_path = tmp_path / "corrupted_data.json"
    json_str = sample_df.to_json(orient="records", date_format="iso")
    corrupted_json = (
        json_str[:-10] + "}"
    )  # Remove last 10 characters and add a closing brace
    file_path.write_text(corrupted_json)
    return str(file_path)


@pytest.fixture
def empty_json_file(tmp_path):
    file_path = tmp_path / "empty_data.json"
    file_path.write_text("")
    return str(file_path)


@pytest.fixture
def large_json_file(tmp_path):
    file_path = tmp_path / "large_data.json"

    start_date = datetime(2010, 1, 1)
    end_date = datetime(2023, 12, 31)

    data = []
    for _ in range(10000):  # Generate 10,000 entries
        created_date = generate_random_date(start_date, end_date)
        last_changed = created_date + timedelta(
            minutes=random.randint(0, 1440)
        )

        entry = {
            "Content": generate_random_content(),
            "Author": f"User{random.randint(1, 1000)}",
            "Date Created": created_date.strftime("%d.%m.%Y %H:%M"),
            "Last Changed": last_changed.strftime("%d.%m.%Y %H:%M"),
        }
        data.append(entry)

    pd.DataFrame(data).to_json(file_path, orient="records", date_format="iso")
    return str(file_path)
