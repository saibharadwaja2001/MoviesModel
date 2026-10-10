from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from movie_model import CSV_PATH

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "movie_models.joblib"


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MODEL_PATH.exists():
        raise RuntimeError(
            "Trained model file not found. Run `python train.py` before starting the API."
        )
    app.state.models = joblib.load(MODEL_PATH)
    app.state.movies = pd.read_csv(CSV_PATH)
    yield


app = FastAPI(
    title="Hollywood Movie Suggestion API",
    description=(
        "Ranks movies from HollywoodMovies.csv using Linear Regression "
        "and Logistic Regression. Logistic Regression estimates the probability "
        "that AudienceScore is at least 75."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


class MovieInput(BaseModel):
    # API clients can send either Python field names or CSV-style aliases.
    model_config = ConfigDict(populate_by_name=True)

    rotten_tomatoes: Optional[float] = Field(
        default=None, alias="RottenTomatoes", ge=0, le=100
    )
    year: Optional[int] = Field(default=None, alias="Year", ge=1888, le=2100)
    budget: Optional[float] = Field(default=None, alias="Budget", ge=0)
    genre: Optional[str] = Field(default=None, alias="Genre")
    lead_studio: Optional[str] = Field(default=None, alias="LeadStudio")
    story: Optional[str] = Field(default=None, alias="Story")


def get_models():
    return app.state.models


@app.get("/")
def home():
    return {
        "message": "Hollywood Movie Suggestion API is running",
        "docs": "/docs",
        "endpoints": ["/health", "/recommendations", "/predict"],
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": hasattr(app.state, "models")}


@app.get("/recommendations")
def recommendations(
    limit: int = Query(default=10, ge=1, le=50),
    genre: Optional[str] = Query(default=None, description="Optional genre filter"),
    high_rating_only: bool = Query(
        default=False,
        description="Only return movies whose estimated high-rating probability is >= 50%",
    ),
):
    models = get_models()
    df = app.state.movies.copy()
    features = models["feature_columns"]

    if genre:
        df = df[df["Genre"].fillna("").str.casefold() == genre.casefold()]

    if df.empty:
        return {
            "count": 0,
            "rating_cutoff": models["rating_cutoff"],
            "recommendations": [],
        }

    candidates = df.copy()
    candidates["PredictedAudienceScore"] = np.clip(
        models["linear_model"].predict(candidates[features]), 0, 100
    )
    candidates["HighRatingProbability"] = (
        models["logistic_model"].predict_proba(candidates[features])[:, 1]
    )

    if high_rating_only:
        candidates = candidates[candidates["HighRatingProbability"] >= 0.5]

    candidates = candidates.sort_values(
        ["PredictedAudienceScore", "HighRatingProbability"],
        ascending=False,
    )

    display_columns = [
        "Movie", "Genre", "RottenTomatoes", "AudienceScore",
        "PredictedAudienceScore", "HighRatingProbability", "Year",
    ]
    result = candidates[display_columns].head(limit).copy()
    result["PredictedAudienceScore"] = result["PredictedAudienceScore"].round(1)
    result["HighRatingProbability"] = (
        result["HighRatingProbability"].mul(100).round(1)
    )

    # Convert NaN values to JSON nulls.
    records = result.astype(object).where(pd.notna(result), None).to_dict(orient="records")

    return {
        "count": len(records),
        "rating_cutoff": models["rating_cutoff"],
        "probability_unit": "percent",
        "recommendations": records,
    }


@app.post("/predict")
def predict_movie(movie: MovieInput):
    models = get_models()

    # Transform API-friendly field names into the exact feature names used in training.
    input_row = pd.DataFrame(
        [movie.model_dump(by_alias=True)],
        columns=models["feature_columns"],
    )

    predicted_score = float(models["linear_model"].predict(input_row)[0])
    predicted_score = float(np.clip(predicted_score, 0, 100))

    high_probability = float(
        models["logistic_model"].predict_proba(input_row)[0, 1]
    )
    cutoff = models["rating_cutoff"]

    return {
        "predicted_audience_score": round(predicted_score, 1),
        "high_rating_probability": round(high_probability * 100, 1),
        "probability_unit": "percent",
        "rating_cutoff": cutoff,
        "predicted_high_rating": high_probability >= 0.5,
        "note": (
            "These are model estimates based on historical movie data, "
            "not a guarantee of personal preference."
        ),
    }
