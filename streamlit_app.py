# streamlit_app.py – Front‑end for the TMDB content‑based recommender

"""Streamlit application that loads the TMDB 5000 dataset, builds a TF‑IDF
bag‑of‑words representation of each movie (tags = overview + genres + keywords
+ cast + crew) and provides an interactive UI to retrieve weighted recommendations.

Features
--------
- Select a movie title (autocomplete drop‑down).
- Choose number of recommendations (1‑20).
- Optional filters: genre, actor, director, minimum IMDb rating.
- Advanced weighting for genre, cast and keywords.
- Results displayed as a nicely formatted table.

The heavy data loading / model building is cached so the app starts fast.
"""

import json
import ast
import os
import pickle
import pandas as pd
import numpy as np
import streamlit as st
import requests

st.set_page_config(page_title="TMDB Recommender", layout="wide")

@st.cache_data(show_spinner=False)
def fetch_poster(movie_id):
    api_key = os.getenv("TMDB_API_KEY")
    if not api_key:
        return "https://via.placeholder.com/500x750?text=No+Poster"

    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}&language=en-US"
    try:
        data = requests.get(url).json()
        poster_path = data.get('poster_path')
        if poster_path:
            return "https://image.tmdb.org/t/p/w500/" + poster_path
    except:
        pass
    return "https://via.placeholder.com/500x750?text=No+Poster"
# ---------------------------------------------------------------------------
# Helper functions – extracted from the original notebook
# ---------------------------------------------------------------------------

# Data loading – cached to avoid recomputation on every UI interaction
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data():
    """Load the pre-processed dataframe from pickle."""
    return pd.read_pickle("movie_list.pkl")

# ---------------------------------------------------------------------------
# Model building – cached as a resource (does not change across runs)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def build_model():
    """Load the pre-computed similarity matrix from pickle."""
    with open("similarity.pkl", "rb") as f:
        similarity = pickle.load(f)
    return similarity

# ---------------------------------------------------------------------------
# Recommendation logic – returns a DataFrame for display
# ---------------------------------------------------------------------------
def recommend_weighted(
    df,
    similarity,
    movie_title,
    num_recommendations=5,
    actor=None,
    director=None,
    imdb_rating=None,
    genre=None,
):
    """Compute weighted recommendations with optional filters.

    Parameters
    ----------
    df : pd.DataFrame
        The cleaned dataframe (must contain columns 'title', 'genres',
        'cast', 'crew', 'vote_average', etc.).
    similarity : np.ndarray
        Pre‑computed cosine similarity matrix.
    movie_title : str
        Title of the pivot movie.
    num_recommendations : int, default 5
        How many movies to return.
    actor, director, imdb_rating, genre : optional
        Additional filters – treated as AND conditions.
    """
    if movie_title not in df["title"].values:
        st.warning(f"Movie '{movie_title}' not found in the database.")
        return pd.DataFrame()

    movie_idx = df.index[df["title"] == movie_title][0]
    base_scores = similarity[movie_idx]

    weighted_scores = []
    for idx, base in enumerate(base_scores):
        if idx == movie_idx:
            continue  # skip the movie itself
        weighted_scores.append((idx, base))

    # Sort by descending weighted similarity
    weighted_scores.sort(key=lambda x: x[1], reverse=True)

    # -------------------------------------------------------------------
    # Apply filters – stop‑early when we have enough results
    # -------------------------------------------------------------------
    results = []
    for idx, score in weighted_scores:
        row = df.iloc[idx]
        # IMDb rating filter
        if imdb_rating is not None and row.get("vote_average", 0) < imdb_rating:
            continue
        # Genre filter – note that `df['genres']` is a space‑joined string of names
        if genre is not None:
            # The dataset stores genres without spaces (e.g., "ScienceFiction")
            # but the UI will present the human‑readable version.
            genre_clean = genre.replace(" ", "")
            if genre_clean not in row["genres"]:
                continue
        # Actor filter – check if the actor appears in the top‑3 cast list
        if actor is not None:
            if actor not in row["cast"]:
                continue
        # Director filter – compare cleaned director name
        if director is not None:
            dir_clean = director.replace(" ", "")
            if isinstance(row["crew"], list):
                if dir_clean not in row["crew"]:
                    continue
            else:
                if dir_clean != row["crew"]:
                    continue
        results.append({
            "movie_id": row["movie_id"],
            "title": row["title"],
            "score": round(score * 100, 2),
            "genres": ", ".join(row["genres"])[:30],
        })
        if len(results) >= num_recommendations:
            break

    return pd.DataFrame(results)

# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
st.title("🎬 TMDB Content‑Based Movie Recommender")
st.caption("Powered by TF‑IDF on movie overview, genres, keywords, cast & director")

# Load & build model (cached)
with st.spinner("Loading pre-computed model…"):
    df = load_data()
    similarity = build_model()

# User inputs & Filters
col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 1])

with col1:
    movie_options = df["title"].sort_values().unique()
    selected_movie = st.selectbox("Select a movie", movie_options)
    
    # Extract properties of the selected movie to prioritize in filters
    selected_row = df[df["title"] == selected_movie].iloc[0]
    sel_genres = selected_row["genres"] if isinstance(selected_row["genres"], list) else []
    sel_cast = selected_row["cast"] if isinstance(selected_row["cast"], list) else []
    sel_crew = selected_row["crew"] if isinstance(selected_row["crew"], list) else [selected_row["crew"]] if isinstance(selected_row["crew"], str) else []

with col2:
    genre_list = sorted({g for genres in df["genres"] for g in (genres if isinstance(genres, list) else [])})
    genre_list = sel_genres + [g for g in genre_list if g not in sel_genres]
    genre_choice = st.selectbox("Genre", ["None"] + genre_list, index=0)
    if genre_choice == "None":
        genre_choice = None

with col3:
    actor_list = sorted({a for cast in df["cast"] for a in (cast if isinstance(cast, list) else [])})
    actor_list = sel_cast + [a for a in actor_list if a not in sel_cast]
    actor_choice = st.selectbox("Actor", ["None"] + actor_list, index=0)
    if actor_choice == "None":
        actor_choice = None

with col4:
    director_list = sorted({d for crew in df["crew"] for d in (crew if isinstance(crew, list) else [])})
    director_list = sel_crew + [d for d in director_list if d not in sel_crew]
    director_choice = st.selectbox("Director", ["None"] + director_list, index=0)
    if director_choice == "None":
        director_choice = None

with col5:
    st.markdown("<br>", unsafe_allow_html=True)
    get_recs = st.button("🔎 Get Recommendations")

num_rec = st.slider("Number of recommendations", 1, 20, 5)

imdb_min = None
if "vote_average" in df.columns:
    imdb_min = st.slider("Minimum IMDb rating", 0.0, 10.0, 0.0, 0.1)
    if imdb_min == 0.0:
        imdb_min = None


# Run recommendation when button pressed
if get_recs:
    with st.spinner("Computing recommendations…"):
        rec_df = recommend_weighted(
            df,
            similarity,
            movie_title=selected_movie,
            num_recommendations=num_rec,

            actor=actor_choice,
            director=director_choice,
            imdb_rating=imdb_min,
            genre=genre_choice,
        )
    if rec_df.empty:
        st.info("No movies matched the selected criteria.")
    else:
        st.success(f"Top {len(rec_df)} recommendations similar to **{selected_movie}**")
        
        num_cols = 5
        for i in range(0, len(rec_df), num_cols):
            cols = st.columns(num_cols)
            for j in range(num_cols):
                if i + j < len(rec_df):
                    movie = rec_df.iloc[i + j]
                    poster_url = fetch_poster(movie["movie_id"])
                    with cols[j]:
                        st.image(poster_url, use_container_width=True)
                        st.markdown(f"**{movie['title']}**")

st.markdown("---")
st.caption(
    "*Note:* The dataset used here is the TMDB 5000 movies/credits CSV files. "
    "Filters are case‑sensitive and expect the exact name as it appears in the data."
)
