# TMDB Content-Based Movie Recommender

Small Streamlit app that provides content-based movie recommendations using the TMDB 5000 dataset (overview + genres + keywords + top cast + director). The project precomputes a TF‑IDF representation and a cosine-similarity matrix for fast retrieval.

Dataset
- Source: https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata

Quick Start (local)
1. Create a Python venv and activate it.
2. Install dependencies:

```
pip install -r requirements.txt
```

3. Build required artifacts (this will create `movie_list.pkl` and `similarity.pkl`):

```
python build_assets.py
```

4. Run the app:

```
streamlit run streamlit_app.py
```

Render deployment

Recommended Render web service settings (also in `render.yaml`):

- Environment: Python
- Build Command: `pip install -r requirements.txt && python build_assets.py`
- Start Command: `streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port $PORT`
- Environment variable: `TMDB_API_KEY` — set this to your TMDB API Key (v3 auth). See below for how to get it.

TMDB API key
- Create an account at https://www.themoviedb.org/.
- In Account → Settings → API, request an API Key (v3 auth) and copy the value.
- Add the key to Render as `TMDB_API_KEY` (or set locally in your environment for local testing).

Notes about large artifacts
- The full cosine similarity matrix (`similarity.pkl`) can be large (~176 MB). To avoid storing that binary in the Git repo, this project generates it during the Render build using `build_assets.py`.
- Do NOT commit `similarity.pkl`. If you previously committed it, remove it from the index with:

```
git rm --cached similarity.pkl
git commit -m "Remove tracked generated similarity.pkl"
git push
```

If Render build fails due to memory limits while computing the similarity matrix, consider one of these options:
- Reduce `TfidfVectorizer(max_features)` in `build_assets.py` to a lower value (e.g., 2000).
- Host the precomputed pickle in an external storage (GitHub Release, S3) and download it at build/runtime.

Files of interest
- `streamlit_app.py` — Streamlit app entrypoint and UI.
- `build_movie_list.py` — Generates `movie_list.pkl` from the CSV files.
- `build_assets.py` — Builds `similarity.pkl` (and will build `movie_list.pkl` if missing).
- `render.yaml` — Render configuration used by the service.

Contact
If you want I can: add a smaller default `max_features` env var, or wire downloading a hosted similarity artifact instead of building it at deploy time.
