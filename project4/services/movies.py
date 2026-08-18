from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import random
import re

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "movie_metadata.csv"


@dataclass(frozen=True)
class Movie:
    source_id: int
    title: str
    year: int
    genres: tuple[str, ...]
    director: str
    actors: tuple[str, ...]
    duration: int
    content_rating: str
    language: str
    country: str
    keywords: tuple[str, ...]

    def as_card(self):
        return {
            "id": self.source_id,
            "title": self.title,
            "year": self.year,
            "genres": self.genres,
            "director": self.director,
            "actors": self.actors,
            "duration": self.duration,
            "content_rating": self.content_rating,
            "language": self.language,
            "country": self.country,
            "keywords": self.keywords[:4],
            "initial": self.title[:1].upper(),
        }


@dataclass(frozen=True)
class MovieFeatureBundle:
    movies: tuple[Movie, ...]
    matrix: object
    feature_names: tuple[str, ...]
    row_by_source_id: dict[int, int]


def _clean_text(value, fallback="Unknown"):
    if pd.isna(value):
        return fallback
    cleaned = str(value).replace("\u00c2", "").replace("\u00a0", " ").strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned or fallback


def _split_pipe(value):
    if pd.isna(value):
        return ()
    return tuple(part.strip() for part in str(value).split("|") if part.strip())


def _safe_int(value, default=0):
    try:
        if pd.isna(value):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


@lru_cache(maxsize=1)
def load_movie_frame():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"IMDb 5000 dataset not found at {DATA_PATH}. "
            "Add movie_metadata.csv from the source linked in the assignment."
        )

    frame = pd.read_csv(DATA_PATH)
    required = {
        "actor_1_name",
        "actor_2_name",
        "actor_3_name",
        "content_rating",
        "country",
        "language",
        "movie_title",
        "genres",
        "director_name",
        "duration",
        "num_voted_users",
        "title_year",
        "plot_keywords",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    frame = frame.copy()
    frame["source_id"] = frame.index.astype(int)
    frame["clean_title"] = frame["movie_title"].map(_clean_text)
    frame["year"] = frame["title_year"].map(_safe_int)
    frame["runtime"] = frame["duration"].map(_safe_int)
    frame["votes"] = frame.get("num_voted_users", 0)
    frame["votes"] = frame["votes"].map(_safe_int)

    frame = frame[
        (frame["clean_title"] != "Unknown")
        & frame["genres"].notna()
        & frame["director_name"].notna()
        & frame["year"].between(1910, 2025)
        & frame["runtime"].between(45, 300)
    ]
    frame = frame.sort_values("votes", ascending=False)
    frame = frame.drop_duplicates(subset=("clean_title", "year"), keep="first")
    return frame.reset_index(drop=True)


def _movies_from_frame(frame):
    movies = []
    for row in frame.to_dict(orient="records"):
        actors = tuple(
            actor
            for actor in (
                _clean_text(row.get("actor_1_name"), ""),
                _clean_text(row.get("actor_2_name"), ""),
                _clean_text(row.get("actor_3_name"), ""),
            )
            if actor
        )
        movies.append(
            Movie(
                source_id=int(row["source_id"]),
                title=row["clean_title"],
                year=int(row["year"]),
                genres=_split_pipe(row.get("genres")),
                director=_clean_text(row.get("director_name")),
                actors=actors,
                duration=int(row["runtime"]),
                content_rating=_clean_text(row.get("content_rating")),
                language=_clean_text(row.get("language")),
                country=_clean_text(row.get("country")),
                keywords=_split_pipe(row.get("plot_keywords")),
            )
        )
    return tuple(movies)


def _most_common(values, limit):
    return {value for value, _ in Counter(values).most_common(limit)}


def _keyword_tokenizer(document):
    return [token for token in document.split("|") if token]


@lru_cache(maxsize=1)
def get_feature_bundle():
    movies = _movies_from_frame(load_movie_frame())
    common_languages = _most_common((movie.language for movie in movies), 8)
    common_countries = _most_common((movie.country for movie in movies), 10)
    rating_counts = Counter(movie.content_rating for movie in movies)
    common_ratings = {value for value, count in rating_counts.items() if count >= 20}
    durations = np.asarray([movie.duration for movie in movies], dtype=float)
    duration_mean, duration_std = durations.mean(), max(durations.std(), 1.0)

    feature_dicts = []
    for movie in movies:
        decade = f"{(movie.year // 10) * 10}s"
        language = movie.language if movie.language in common_languages else "OTHER"
        country = movie.country if movie.country in common_countries else "OTHER"
        rating = movie.content_rating if movie.content_rating in common_ratings else "OTHER"
        features = {
            "numeric::duration": (movie.duration - duration_mean) / duration_std,
            f"decade::{decade}": 1.0,
            f"language::{language}": 1.0,
            f"country::{country}": 1.0,
            f"rating::{rating}": 1.0,
        }
        for genre in movie.genres:
            features[f"genre::{genre}"] = 1.0
        feature_dicts.append(features)

    vectorizer = DictVectorizer(sparse=True, sort=True)
    categorical_matrix = vectorizer.fit_transform(feature_dicts).tocsr()

    keyword_documents = ["|".join(movie.keywords) for movie in movies]
    keyword_vectorizer = TfidfVectorizer(
        tokenizer=_keyword_tokenizer,
        token_pattern=None,
        lowercase=True,
        min_df=8,
        sublinear_tf=True,
    )
    keyword_matrix = keyword_vectorizer.fit_transform(keyword_documents)
    component_count = min(16, max(1, keyword_matrix.shape[1] - 1))
    keyword_svd = TruncatedSVD(n_components=component_count, random_state=0)
    keyword_components = keyword_svd.fit_transform(keyword_matrix)
    matrix = hstack(
        (categorical_matrix, csr_matrix(keyword_components)), format="csr"
    ).astype(np.float64)
    feature_names = tuple(vectorizer.get_feature_names_out()) + tuple(
        f"keyword_topic::{index + 1}" for index in range(component_count)
    )
    return MovieFeatureBundle(
        movies=movies,
        matrix=matrix,
        feature_names=feature_names,
        row_by_source_id={movie.source_id: row for row, movie in enumerate(movies)},
    )


def get_movie(source_id):
    bundle = get_feature_bundle()
    row = bundle.row_by_source_id[int(source_id)]
    return bundle.movies[row]


def build_movie_plan(seed, pairwise_trials=10, ranking_trials=2, evaluation_trials=20):
    frame = load_movie_frame()
    familiar_pool = frame.loc[frame["votes"] >= 1000, "source_id"].astype(int).tolist()
    required = pairwise_trials * 2 + ranking_trials * 10 + evaluation_trials * 2
    if len(familiar_pool) < required:
        familiar_pool = frame["source_id"].astype(int).tolist()
    if len(familiar_pool) < required:
        raise ValueError("The movie dataset does not contain enough eligible movies.")

    rng = random.Random(int(seed))
    selected = rng.sample(familiar_pool, required)
    cursor = 0

    pairwise = []
    for _ in range(pairwise_trials):
        pairwise.append(selected[cursor : cursor + 2])
        cursor += 2

    rankings = []
    for _ in range(ranking_trials):
        rankings.append(selected[cursor : cursor + 10])
        cursor += 10

    evaluation = []
    for _ in range(evaluation_trials):
        evaluation.append(selected[cursor : cursor + 2])
        cursor += 2

    return {"pairwise": pairwise, "rankings": rankings, "evaluation": evaluation}
