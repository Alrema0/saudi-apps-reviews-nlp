"""
Step 1 - Collect Arabic reviews from Google Play (Saudi store).

How to use:
1. Open the app page on Google Play in a browser.
2. Copy the id from the URL:  play.google.com/store/apps/details?id=THIS.PART
3. Put it in APPS below, then run:  python 01_scrape_reviews.py
"""
from pathlib import Path

import pandas as pd
from google_play_scraper import Sort, reviews

# Display name -> Google Play package id (fill these in yourself)
APPS = {
    "Absher": "sa.gov.moi",
    "Tawakkalna": "sa.gov.nic.twkhayat",
    "Nafath": "sa.gov.nic.myid",
}
COUNT_PER_APP = 3000  # newest reviews per app
OUT_FILE = Path("data/raw_reviews.csv")

# Drop personal fields on purpose: the analysis does not need user identities.
KEEP_COLS = [
    "reviewId", "content", "score", "thumbsUpCount",
    "reviewCreatedVersion", "at", "replyContent", "repliedAt",
]


def fetch(app_id: str, count: int) -> pd.DataFrame:
    result, _ = reviews(
        app_id, lang="ar", country="sa", sort=Sort.NEWEST, count=count
    )
    return pd.DataFrame(result)


def main() -> None:
    if not APPS:
        raise SystemExit("Fill the APPS dictionary first (see the docstring).")

    frames = []
    for name, app_id in APPS.items():
        print(f"Fetching {name} ({app_id}) ...")
        df = fetch(app_id, COUNT_PER_APP)
        df = df[[c for c in KEEP_COLS if c in df.columns]].copy()
        df.insert(0, "app", name)
        frames.append(df)
        print(f"  got {len(df)} reviews")

    all_reviews = pd.concat(frames, ignore_index=True)
    all_reviews = all_reviews.drop_duplicates(subset="reviewId")

    OUT_FILE.parent.mkdir(exist_ok=True)
    all_reviews.to_csv(OUT_FILE, index=False, encoding="utf-8-sig")
    print(f"Saved {len(all_reviews)} reviews to {OUT_FILE}")


if __name__ == "__main__":
    main()
