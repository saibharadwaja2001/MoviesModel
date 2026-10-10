from pathlib import Path
import joblib

from movie_model import CSV_PATH, train_models

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "movie_models.joblib"
RECOMMENDATIONS_PATH = BASE_DIR / "movie_recommendations.csv"


def main():
    models = train_models(CSV_PATH, verbose=True)
    joblib.dump(models, MODEL_PATH)
    print(f"\nSaved trained models to: {MODEL_PATH}")

    # Generate and save a ranked recommendation list for convenience.
    import pandas as pd
    import numpy as np

    df = pd.read_csv(CSV_PATH)
    features = models["feature_columns"]
    candidates = df.copy()

    candidates["PredictedAudienceScore"] = np.clip(
        models["linear_model"].predict(candidates[features]), 0, 100
    )
    candidates["HighRatingProbability"] = (
        models["logistic_model"].predict_proba(candidates[features])[:, 1]
    )

    display_columns = [
        "Movie", "Genre", "RottenTomatoes", "AudienceScore",
        "PredictedAudienceScore", "HighRatingProbability", "Year",
    ]
    candidates = candidates.sort_values(
        ["PredictedAudienceScore", "HighRatingProbability"],
        ascending=False,
    )
    candidates[display_columns].to_csv(RECOMMENDATIONS_PATH, index=False)
    print(f"Saved ranked recommendations to: {RECOMMENDATIONS_PATH}")
    print("\nTop 10 suggestions:")
    preview = candidates[display_columns].head(10).copy()
    preview["PredictedAudienceScore"] = preview["PredictedAudienceScore"].round(1)
    preview["HighRatingProbability"] = (preview["HighRatingProbability"] * 100).round(1)
    print(preview.to_string(index=False))


if __name__ == "__main__":
    main()
