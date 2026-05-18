import ast
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
MOVIES_CSV = ROOT / "tmdb_5000_movies.csv"
CREDITS_CSV = ROOT / "tmdb_5000_credits.csv"
OUT_PKL = ROOT / "movie_list.pkl"


def _parse_field(cell):
    if pd.isna(cell):
        return []
    try:
        data = ast.literal_eval(cell)
    except Exception:
        try:
            import json

            data = json.loads(cell)
        except Exception:
            return []
    return data


def extract_names_from_list_of_dicts(cell, key="name"):
    items = _parse_field(cell)
    names = [i.get(key, "").replace(" ", "") for i in items if isinstance(i, dict) and i.get(key)]
    return names


def extract_top_cast(cell, n=3):
    items = _parse_field(cell)
    names = [i.get("name", "") for i in items if isinstance(i, dict) and i.get("name")]
    return names[:n]


def extract_director(cell):
    items = _parse_field(cell)
    for d in items:
        if isinstance(d, dict) and d.get("job") == "Director":
            return d.get("name", "")
    return ""


def build_movie_list():
    movies = pd.read_csv(MOVIES_CSV)
    credits = pd.read_csv(CREDITS_CSV)

    # align ids
    movies = movies.rename(columns={"id": "movie_id"})
    credits = credits.rename(columns={"id": "movie_id"})

    df = movies.merge(credits, on="movie_id")

    # pick columns and clean
    df = df[["movie_id", "title", "overview", "genres", "keywords", "cast", "crew", "vote_average"]]

    df["genres"] = df["genres"].apply(lambda c: extract_names_from_list_of_dicts(c, "name"))
    df["keywords"] = df["keywords"].apply(lambda c: extract_names_from_list_of_dicts(c, "name"))
    df["cast"] = df["cast"].apply(lambda c: extract_top_cast(c, 3))
    df["crew"] = df["crew"].apply(lambda c: extract_director(c))

    # build tags: overview + genres + keywords + cast + director
    def make_tags(row):
        parts = []
        if pd.notna(row.get("overview")):
            parts.append(row.get("overview", ""))
        parts.extend(row.get("genres") or [])
        parts.extend(row.get("keywords") or [])
        parts.extend(row.get("cast") or [])
        if row.get("crew"):
            parts.append(row.get("crew"))
        return " ".join(parts)

    df["tags"] = df.apply(make_tags, axis=1)

    df.to_pickle(OUT_PKL)


if __name__ == "__main__":
    build_movie_list()