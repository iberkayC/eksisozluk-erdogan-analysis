from typing import List
from pathlib import Path

from nltk.corpus import stopwords

from src.utils.logging_config import get_logger


logger = get_logger(__name__)


def get_turkish_stopwords() -> List[str]:
    """
    Retrieves a list of Turkish stopwords for text processing from a file.

    Returns:
        List[str]: List of Turkish stopwords.
    """
    stop_words = set(stopwords.words("turkish"))
    stopwords_path = (
        Path(__file__).parent.parent.parent / "data" / "turkish_stopwords.txt"
    )

    with open(stopwords_path, "r", encoding="utf-8") as file:
        custom_stopwords = [line.strip() for line in file if line.strip()]

    stop_words.update(custom_stopwords)

    logger.debug(f"Retrieved {len(stop_words)} Turkish stopwords")
    return list(stop_words)
