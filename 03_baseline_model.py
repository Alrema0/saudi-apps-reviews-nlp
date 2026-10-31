"""
Step 3 - Baseline sentiment model: TF-IDF + Logistic Regression.

Train: reviews labeled by STARS (weak labels): 1-2 stars = negative, 4-5 = positive.
Test : the 300 reviews labeled by MEANING (data/label_sample_labeled.csv).

The test reviews are removed from the training data, so the model never sees them.

Run:  python 03_baseline_model.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save figures to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
)

CLEAN_FILE = Path("data/clean_reviews.csv")
LABELED_FILE = Path("data/label_sample_labeled.csv")
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

CLASSES = ["negative", "positive"]


def load_data():
    clean = pd.read_csv(CLEAN_FILE, encoding="utf-8-sig")
    clean["text_clean"] = clean["text_clean"].fillna("")

    labeled = pd.read_csv(LABELED_FILE, encoding="utf-8-sig")
    labeled.columns = [c.lstrip("﻿") for c in labeled.columns]

    # Test set = manually labeled reviews (positive / negative only; neutral and
    # skip are too rare to measure reliably, so we report them but do not score them)
    print("Test set label counts:")
    print(labeled["label_manual"].value_counts(), "\n")
    labeled = labeled.merge(clean[["reviewId", "text_clean"]], on="reviewId", how="left")
    labeled["text_clean"] = labeled["text_clean"].fillna("")
    test = labeled[labeled["label_manual"].isin(CLASSES)].copy()

    # Train set = everything else with a clear star signal
    train = clean[~clean["reviewId"].isin(labeled["reviewId"]) & (clean["score"] != 3)].copy()
    train["y"] = np.where(train["score"] >= 4, "positive", "negative")
    return train, test


def main() -> None:
    train, test = load_data()
    print(f"Train: {len(train)} reviews | Test: {len(test)} reviews")
    print("Train label counts:")
    print(train["y"].value_counts(), "\n")

    # Two views of the text: whole words (+ word pairs) and character pieces.
    # Character n-grams help with dialect spelling ("مايفتح" / "ما يفتح").
    word_vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3,
                               sublinear_tf=True)

    X_train = hstack([word_vec.fit_transform(train["text_clean"]),
                      char_vec.fit_transform(train["text_clean"])]).tocsr()
    X_test = hstack([word_vec.transform(test["text_clean"]),
                     char_vec.transform(test["text_clean"])]).tocsr()

    # class_weight="balanced": the training data is ~90% positive, so we make
    # mistakes on the rare class count more.
    model = LogisticRegression(max_iter=2000, C=2.0, class_weight="balanced")
    model.fit(X_train, train["y"])

    test["pred"] = model.predict(X_test)
    y_true = test["label_manual"]

    print("=== MODEL (trained on stars) vs MEANING labels ===")
    print(classification_report(y_true, test["pred"], digits=3))
    model_f1 = f1_score(y_true, test["pred"], average="macro")

    # Reference point: what if we simply trusted the stars?
    stars_pred = np.where(test["score"] >= 4, "positive", "negative")
    stars_f1 = f1_score(y_true, stars_pred, average="macro")
    print("=== STARS alone vs MEANING labels ===")
    print(classification_report(y_true, stars_pred, digits=3))

    print(f"Macro-F1 -> model: {model_f1:.3f} | stars-as-prediction: {stars_f1:.3f}\n")

    # Confusion matrix picture
    fig, ax = plt.subplots(figsize=(4.5, 4))
    cm = confusion_matrix(y_true, test["pred"], labels=CLASSES)
    ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(ax=ax, colorbar=False)
    ax.set_title("Baseline model (TF-IDF + LogReg)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "confusion_matrix_baseline.png", dpi=150)

    # Most influential WORDS for each class (first block of features = words)
    n_words = len(word_vec.vocabulary_)
    words = np.array(word_vec.get_feature_names_out())
    coefs = model.coef_[0][:n_words]  # positive coef -> pushes toward "positive"
    top_pos = words[np.argsort(coefs)[-15:][::-1]]
    top_neg = words[np.argsort(coefs)[:15]]
    print("Words pushing toward POSITIVE:", " | ".join(top_pos))
    print("Words pushing toward NEGATIVE:", " | ".join(top_neg), "\n")

    # Save mistakes so you can read them (error analysis)
    test[["reviewId", "app", "score", "content", "label_manual", "pred"]].assign(
        correct=lambda d: d["label_manual"] == d["pred"]
    ).to_csv(OUT_DIR / "test_predictions.csv", index=False, encoding="utf-8-sig")
    errors = test[test["label_manual"] != test["pred"]]
    print(f"Mistakes: {len(errors)} of {len(test)} (saved in outputs/test_predictions.csv)")


if __name__ == "__main__":
    main()
