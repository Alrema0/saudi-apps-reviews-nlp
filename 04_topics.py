"""
Step 4 - What do unhappy users complain about?  Topic discovery with TF-IDF + NMF.

Input : data/clean_reviews.csv   (output of 02_clean_text.py)
Output: outputs/topics_summary.csv            one row per topic (top words, size, share per app)
        outputs/negative_reviews_topics.csv   every negative review with its topic
        outputs/topics_by_app.png             chart

Run:  python 04_topics.py          (6 topics)
      python 04_topics.py 8        (try 8 topics)
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

CLEAN_FILE = Path("data/clean_reviews.csv")
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

N_TOPICS = int(sys.argv[1]) if len(sys.argv) > 1 else 6
MIN_WORDS = 3          # shorter reviews carry too little text to place in a topic
MAX_STAR = 2           # "unhappy" = 1-2 stars (a noisy but practical proxy)

# Words that appear in almost every review and say nothing about WHAT went wrong.
STOPWORDS = """
في من على الى الي عن مع هذا هذه هذي ذلك هذاك هاذي هاذه هو هي انا انت نحن
ان انه انها اني لا ما لم لن لما ليش ليه لماذا كيف وش ايش شو هل كل كان كانت
يكون صار صارت اصبح قد بعد قبل عند عندي معي لي لك له لها لنا لكم لهم بس لكن ولكن
او ثم حتى اذا اذ بين فقط جدا جدآ دائما دايما مره مرة اكثر اقل شي شيء
الله ارجو ارجوا نرجو نرجوا اتمنى نتمنى شكرا شكرآ وشكرا مشكور
السلام عليكم ورحمة وبركاته
التطبيق تطبيق البرنامج برنامج ابشر نفاذ نفاد توكلنا
ياريت يارب والله ولا وما ولي ومعي وعندي وهو
حل الحل المشكله مشكله المشكلة مشكلة للاسف الان اخر بشكل غير الا يوجد علي
خطا بسبب فيه مو مش
سيء فاشل جيد جيدا للغاية الرجاء كثيرة كثير وكل زي وانا عليه عليها اليوم
""".split()

# Arabic glues negation onto the verb ("لايفتح", "مايفتح") or splits it ("لا يفتح").
# Without this, one complaint becomes three different "words".
NEGATION_GLUED = __import__("re").compile(r"\b(لا|ما|لم|لن)(?=[يتنا][ء-ي]{2,})")

# Different spellings of the same word -> one form
VARIANTS = {
    "سيئ": "سيء", "سئ": "سيء", "سي": "سيء", "سيئه": "سيء", "سيئة": "سيء",
    "بطيئ": "بطيء", "بطي": "بطيء", "بطئ": "بطيء", "بطيئة": "بطيء",
    "يهنق": "يعلق", "معلق": "يعلق", "تعليق": "يعلق", "يهنج": "يعلق", "هنق": "يعلق",
    "الواي": "واي", "وايفاي": "واي فاي",
}


def prep(text: str) -> str:
    text = NEGATION_GLUED.sub(lambda m: m.group(1) + " ", text)
    words = [VARIANTS.get(w, w) for w in text.split()]
    return " ".join(words)


def main() -> None:
    df = pd.read_csv(CLEAN_FILE, encoding="utf-8-sig")
    df["text_clean"] = df["text_clean"].fillna("")
    neg = df[(df["score"] <= MAX_STAR) & (df["n_words"] >= MIN_WORDS)].copy()
    neg = neg.reset_index(drop=True)
    neg["text_topic"] = neg["text_clean"].apply(prep)
    print(f"Negative reviews used: {len(neg)} "
          f"(stars <= {MAX_STAR}, >= {MIN_WORDS} words)")
    print(neg["app"].value_counts().to_string(), "\n")

    vec = TfidfVectorizer(
        ngram_range=(1, 2), min_df=5, max_df=0.4,
        stop_words=STOPWORDS, sublinear_tf=True,
    )
    X = vec.fit_transform(neg["text_topic"])
    terms = np.array(vec.get_feature_names_out())
    print(f"Vocabulary: {len(terms)} terms\n")

    nmf = NMF(n_components=N_TOPICS, init="nndsvda", random_state=0, max_iter=600)
    W = nmf.fit_transform(X)          # review x topic strength
    H = nmf.components_               # topic x term weight

    neg["topic"] = W.argmax(axis=1)
    neg["topic_strength"] = W.max(axis=1)
    # reviews that match no topic at all (all zeros) are "other"
    neg.loc[W.sum(axis=1) == 0, "topic"] = -1

    rows = []
    app_names = sorted(neg["app"].unique())
    for t in range(N_TOPICS):
        top_terms = terms[np.argsort(H[t])[::-1][:10]]
        members = neg[neg["topic"] == t].sort_values("topic_strength", ascending=False)
        share = members["app"].value_counts(normalize=True)
        examples = members["content"].head(3).tolist()
        rows.append({
            "topic": t,
            "n_reviews": len(members),
            "pct_of_negative": round(100 * len(members) / len(neg), 1),
            "top_words": " | ".join(top_terms),
            **{f"pct_{a}": round(100 * share.get(a, 0), 1) for a in app_names},
            "examples": " ## ".join(e[:140] for e in examples),
        })
        print(f"--- Topic {t}: {len(members)} reviews ({100*len(members)/len(neg):.0f}%) ---")
        print("words   :", " | ".join(top_terms))
        print("apps    :", ", ".join(f"{a} {100*share.get(a,0):.0f}%" for a in app_names))
        for e in examples:
            print("example :", e[:130])
        print()

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "topics_summary.csv", index=False, encoding="utf-8-sig")
    neg[["reviewId", "app", "score", "at", "content", "topic", "topic_strength"]].to_csv(
        OUT_DIR / "negative_reviews_topics.csv", index=False, encoding="utf-8-sig")

    # How many complaints per topic, split by app (counts, so bars are comparable)
    counts = pd.crosstab(neg["topic"], neg["app"]).reindex(range(N_TOPICS), fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    counts.plot(kind="bar", stacked=True, ax=ax, width=0.7)
    ax.set_xlabel("topic number (see topics_summary.csv)")
    ax.set_ylabel("number of negative reviews")
    ax.set_title(f"Complaint topics by app ({len(neg)} reviews, 1-2 stars)")
    ax.legend(title="app")
    plt.xticks(rotation=0)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "topics_by_app.png", dpi=150)
    print("Saved: outputs/topics_summary.csv, negative_reviews_topics.csv, topics_by_app.png")


if __name__ == "__main__":
    main()
