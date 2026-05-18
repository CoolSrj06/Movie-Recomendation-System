from pathlib import Path

import pickle

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


ROOT = Path(__file__).resolve().parent
MOVIE_LIST_PATH = ROOT / "movie_list.pkl"
SIMILARITY_PATH = ROOT / "similarity.pkl"


def build_similarity() -> None:
    # If movie_list.pkl doesn't exist, try to build it from CSVs
    if not MOVIE_LIST_PATH.exists():
        try:
            from build_movie_list import build_movie_list

            build_movie_list()
        except Exception as e:
            raise RuntimeError("movie_list.pkl missing and failed to build from CSVs: " + str(e))

    df = pd.read_pickle(MOVIE_LIST_PATH)

    # Ensure tags are strings
    df["tags"] = df["tags"].apply(
        lambda x: " ".join(map(str, x)) if isinstance(x, list) else str(x)
    )

    if "tags" not in df.columns:
        raise ValueError("movie_list.pkl must contain a 'tags' column to build similarity.pkl")

    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 2))
    vector = vectorizer.fit_transform(df["tags"]).toarray()
    similarity = cosine_similarity(vector)

    with open(SIMILARITY_PATH, "wb") as file_handle:
        pickle.dump(similarity, file_handle)


if __name__ == "__main__":
    build_similarity()