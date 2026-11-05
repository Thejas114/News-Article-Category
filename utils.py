"""Shared helpers used by train_model.py and app.py."""
import re

CATEGORY_NAMES = {1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tech"}
CATEGORY_ICONS = {"World": "🌍", "Sports": "⚽", "Business": "💼", "Sci/Tech": "🔬"}


def clean_text(text: str) -> str:
    """Lowercase, remove URLs/HTML/numbers/punctuation, collapse spaces.
    Stop-word removal is done later by TF-IDF."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = text.replace("\\", " ")          # AG News uses '\' as a line break
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text