"""
Step 5 - Arabic transformer vs. the TF-IDF baseline, on the SAME 260 test reviews.

The model is CAMeLBERT, already fine-tuned for Arabic sentiment by its authors
(CAMeL-Lab). We do NOT train it here: we only run it on our test reviews, so the
test reviews are as unseen for it as they are for the baseline.

Needs:  pip install torch transformers        (first run downloads ~450 MB)
Input : data/label_sample_labeled.csv   (test labels)
        outputs/test_predictions.csv    (baseline predictions, from 03_baseline_model.py)
Output: outputs/arabic_model_predictions.csv   model probabilities (cached, so re-runs are instant)
        outputs/model_comparison.csv           per-review answers of both models
        outputs/confusion_matrix_arabic.png

Run:  python 06_arabic_model.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
)

# If this one fails to download, try the "-mix-sentiment" or "-msa-sentiment" version.
MODEL_NAME = "CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment"
LABELED_FILE = Path("data/label_sample_labeled.csv")
BASELINE_FILE = Path("outputs/test_predictions.csv")
OUT_DIR = Path("outputs")
CACHE_FILE = OUT_DIR / "arabic_model_predictions.csv"

CLASSES = ["negative", "positive"]
BATCH_SIZE = 16
MAX_LEN = 128


def load_test() -> pd.DataFrame:
    df = pd.read_csv(LABELED_FILE, encoding="utf-8-sig")
    df.columns = [c.lstrip("﻿") for c in df.columns]
    return df[df["label_manual"].isin(CLASSES)].reset_index(drop=True)


def score_with_model(texts: list[str]) -> pd.DataFrame:
    """Return the model's probabilities p_negative / p_neutral / p_positive per text."""
    # Imported here so the rest of the script can be read and tested without torch.
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    print(f"Loading {MODEL_NAME} (first run downloads it, please wait) ...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).eval()

    # Read the label names from the model itself instead of assuming their order.
    id2label = {int(i): str(l).lower() for i, l in model.config.id2label.items()}
    wanted = {"negative", "neutral", "positive"}
    if set(id2label.values()) != wanted:
        raise SystemExit(f"Unexpected labels {id2label}; expected {sorted(wanted)}. "
                         "Send me this message and I will adjust the script.")
    print("Model labels:", id2label)

    rows = []
    with torch.no_grad():
        for start in range(0, len(texts), BATCH_SIZE):
            batch = texts[start:start + BATCH_SIZE]
            enc = tok(batch, truncation=True, max_length=MAX_LEN,
                      padding=True, return_tensors="pt")
            probs = torch.softmax(model(**enc).logits, dim=-1).numpy()
            rows.append(probs)
            print(f"  scored {min(start + BATCH_SIZE, len(texts))}/{len(texts)}", end="\r")
    print()
    probs = np.vstack(rows)
    return pd.DataFrame({f"p_{id2label[i]}": probs[:, i] for i in range(probs.shape[1])})


def get_model_scores(test: pd.DataFrame) -> pd.DataFrame:
    """Score once, then reuse the saved file on later runs."""
    if CACHE_FILE.exists():
        cached = pd.read_csv(CACHE_FILE, encoding="utf-8-sig")
        if (cached["model"].iloc[0] == MODEL_NAME
                and cached["reviewId"].tolist() == test["reviewId"].tolist()):
            print(f"Using saved model scores from {CACHE_FILE}")
            return cached
    scores = score_with_model(test["content"].astype(str).tolist())
    out = pd.concat([test[["reviewId"]], scores], axis=1).assign(model=MODEL_NAME)
    OUT_DIR.mkdir(exist_ok=True)
    out.to_csv(CACHE_FILE, index=False, encoding="utf-8-sig")
    return out


def bootstrap_diff(y, a, b, n=2000, seed=0):
    """95% interval for macro-F1(a) - macro-F1(b) by resampling the test reviews."""
    rng = np.random.default_rng(seed)
    y, a, b = (np.asarray(v) for v in (y, a, b))
    idx = rng.integers(0, len(y), size=(n, len(y)))
    f = lambda t, p: f1_score(t, p, labels=CLASSES, average="macro", zero_division=0)
    diffs = np.array([f(y[i], a[i]) - f(y[i], b[i]) for i in idx])
    return np.percentile(diffs, [2.5, 97.5])


def main() -> None:
    test = load_test()
    print(f"Test reviews: {len(test)}  "
          f"({(test['label_manual'] == 'negative').sum()} negative, "
          f"{(test['label_manual'] == 'positive').sum()} positive)\n")

    scores = get_model_scores(test)
    test = test.merge(scores, on="reviewId", how="left")

    # Our labels are positive/negative only, so the model must pick one of the two:
    # compare its positive and negative probabilities and ignore "neutral".
    test["pred_arabic"] = np.where(test["p_positive"] > test["p_negative"],
                                   "positive", "negative")
    top = test[["p_negative", "p_neutral", "p_positive"]].idxmax(axis=1)
    print(f"Reviews where the model's top choice was 'neutral': "
          f"{(top == 'p_neutral').mean():.0%} (ignored in the score below)\n")

    y = test["label_manual"]
    print("=== ARABIC MODEL (CAMeLBERT) vs MEANING labels ===")
    print(classification_report(y, test["pred_arabic"], labels=CLASSES, digits=3,
                                zero_division=0))
    f1_arabic = f1_score(y, test["pred_arabic"], labels=CLASSES, average="macro")

    fig, ax = plt.subplots(figsize=(4.5, 4))
    cm = confusion_matrix(y, test["pred_arabic"], labels=CLASSES)
    ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(ax=ax, colorbar=False)
    ax.set_title("Arabic model (CAMeLBERT)")
    fig.tight_layout()
    OUT_DIR.mkdir(exist_ok=True)
    fig.savefig(OUT_DIR / "confusion_matrix_arabic.png", dpi=150)

    stars_pred = np.where(test["score"] >= 4, "positive", "negative")
    f1_stars = f1_score(y, stars_pred, labels=CLASSES, average="macro")
    print(f"Macro-F1 -> stars: {f1_stars:.3f} | Arabic model: {f1_arabic:.3f}")

    if not BASELINE_FILE.exists():
        print(f"\n{BASELINE_FILE} not found: run 03_baseline_model.py first to compare "
              "with the TF-IDF baseline.")
        return

    base = pd.read_csv(BASELINE_FILE, encoding="utf-8-sig")[["reviewId", "pred"]]
    test = test.merge(base.rename(columns={"pred": "pred_baseline"}),
                      on="reviewId", how="left")
    test = test.dropna(subset=["pred_baseline"])
    y = test["label_manual"]
    f1_base = f1_score(y, test["pred_baseline"], labels=CLASSES, average="macro")
    f1_arabic = f1_score(y, test["pred_arabic"], labels=CLASSES, average="macro")
    print(f"Macro-F1 -> stars: {f1_stars:.3f} | TF-IDF baseline: {f1_base:.3f} "
          f"| Arabic model: {f1_arabic:.3f}")

    lo, hi = bootstrap_diff(y, test["pred_arabic"], test["pred_baseline"])
    print(f"\nArabic model minus baseline, macro-F1 difference: "
          f"{f1_arabic - f1_base:+.3f}  (95% interval {lo:+.3f} to {hi:+.3f})")
    if lo <= 0 <= hi:
        print("The interval includes 0: with only this many test reviews we cannot say "
              "that one model is better. Report it as 'similar'.")
    else:
        print("The interval does not include 0: the difference is unlikely to be chance.")

    a_ok = test["pred_arabic"] == y
    b_ok = test["pred_baseline"] == y
    print("\nWho is right?")
    print(f"  both right           : {(a_ok & b_ok).sum()}")
    print(f"  only Arabic right    : {(a_ok & ~b_ok).sum()}")
    print(f"  only baseline right  : {(~a_ok & b_ok).sum()}")
    print(f"  both wrong           : {(~a_ok & ~b_ok).sum()}")

    keep = ["reviewId", "app", "score", "content", "label_manual",
            "pred_baseline", "pred_arabic", "p_negative", "p_neutral", "p_positive"]
    test[keep].to_csv(OUT_DIR / "model_comparison.csv", index=False, encoding="utf-8-sig")
    print("\nArabic model mistakes (first 10):")
    for _, r in test[~a_ok].head(10).iterrows():
        print(f"  true={r['label_manual']:8s} arabic={r['pred_arabic']:8s} "
              f"baseline={r['pred_baseline']:8s} | {str(r['content'])[:90]}")
    print("\nSaved: outputs/model_comparison.csv, confusion_matrix_arabic.png")


if __name__ == "__main__":
    main()
