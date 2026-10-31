"""
Step 2 - Clean and normalize Arabic review text, build weak labels,
and export a sample for manual labeling.

Run:  python 02_clean_text.py
"""
import re
from pathlib import Path

import pandas as pd

RAW_FILE = Path("data/raw_reviews.csv")
CLEAN_FILE = Path("data/clean_reviews.csv")
LABEL_SAMPLE_FILE = Path("data/label_sample.csv")

TASHKEEL = re.compile(r"[\u0617-\u061A\u064B-\u0652]")
TATWEEL = "\u0640"
URL = re.compile(r"https?://\S+|www\.\S+")
EMOJI = re.compile("[\U0001F300-\U0001FAFF\u2600-\u27BF]")
NON_ARABIC_KEEP = re.compile(r"[^\u0600-\u06FF\s]")  # keep Arabic letters only
ARABIC_CHAR = re.compile(r"[\u0600-\u06FF]")


def extract_emojis(text: str) -> str:
    """Emojis carry sentiment, so keep them as a separate feature."""
    return "".join(EMOJI.findall(text))


def normalize(text: str) -> str:
    text = URL.sub(" ", text)
    text = TASHKEEL.sub("", text)
    text = text.replace(TATWEEL, "")
    text = re.sub("[إأآ]", "ا", text)          # unify alef forms
    text = text.replace("ى", "ي")              # alef maqsura -> ya
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)  # "ممتااااز" -> "ممتاز"
    text = NON_ARABIC_KEEP.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def arabic_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(bool(ARABIC_CHAR.match(c)) for c in letters) / len(letters)


def weak_label(score: int) -> str:
    """Star rating as a noisy proxy for sentiment (validate it manually!)."""
    if score <= 2:
        return "negative"
    if score == 3:
        return "neutral"
    return "positive"


def main() -> None:
    df = pd.read_csv(RAW_FILE)
    n_start = len(df)

    df = df.dropna(subset=["content"]).copy()
    df["content"] = df["content"].astype(str)

    df["emojis"] = df["content"].apply(extract_emojis)
    df["text_clean"] = df["content"].apply(normalize)

    # Keep mostly-Arabic reviews. One-word reviews ("ممتاز") are genuine
    # sentiment data, so we keep them and just record the length. Filter on
    # n_words later when needed (e.g. topic modeling wants >= 4 words).
    df = df[df["content"].apply(arabic_ratio) > 0.5].copy()
    df["n_words"] = df["text_clean"].str.split().str.len()
    df = df[df["n_words"] >= 1]

    # Remove only duplicate LONG reviews (copy-paste spam). Identical short
    # reviews are real, and dropping them would distort the sentiment mix.
    is_long_dup = (df["n_words"] >= 4) & df.duplicated(subset=["app", "text_clean"])
    df = df[~is_long_dup].copy()

    df["label_weak"] = df["score"].apply(weak_label)
    df["at"] = pd.to_datetime(df["at"])

    CLEAN_FILE.parent.mkdir(exist_ok=True)
    df.to_csv(CLEAN_FILE, index=False, encoding="utf-8-sig")
    print(f"{n_start} raw -> {len(df)} clean reviews saved to {CLEAN_FILE}")
    print(df["label_weak"].value_counts())

    # Sample ~300 reviews (balanced by stars) to label BY HAND.
    # This becomes your trusted test set.
    per_star = 60
    sample = pd.concat(
        [
            g.sample(min(len(g), per_star), random_state=42)
            for _, g in df[df["n_words"] >= 3].groupby("score")
        ]
    )[["reviewId", "app", "score", "content"]].assign(label_manual="")
    sample.to_csv(LABEL_SAMPLE_FILE, index=False, encoding="utf-8-sig")
    print(f"Wrote {len(sample)} rows to {LABEL_SAMPLE_FILE} - fill label_manual "
          f"with positive / neutral / negative.")


if __name__ == "__main__":
    main()
