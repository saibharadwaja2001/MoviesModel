# Hollywood Movie Suggestion API

A small machine-learning project built from `HollywoodMovies.csv`. It trains:

- **Linear Regression** to estimate `AudienceScore` (0–100).
- **Logistic Regression** to estimate the probability that `AudienceScore >= 75`.

The preprocessing steps (missing-value imputation, scaling, and one-hot encoding) are bundled inside each fitted scikit-learn pipeline and saved with the model.

## Project files

- `HollywoodMovies.csv` — source dataset.
- `movie_model.py` — reusable model training/evaluation function.
- `train.py` — trains both models, saves `movie_models.joblib`, and exports ranked recommendations.
- `main.py` — FastAPI application.
- `requirements.txt` — Python dependencies.

## Setup

Use Python 3.10+ (Python 3.11 recommended).

### 1. Create and activate a virtual environment

Windows PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 3. Train and save the models

Run this once after setup, and again whenever you intentionally retrain:

```bash
python train.py
```

This prints evaluation metrics, saves `movie_models.joblib`, and writes `movie_recommendations.csv`.

### 4. Start the API

```bash
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Open:

- API root: http://127.0.0.1:8000/
- Health: http://127.0.0.1:8000/health
- Swagger docs: http://127.0.0.1:8000/docs
- Recommendations: http://127.0.0.1:8000/recommendations

## Endpoints

### `GET /recommendations?limit=10`

Returns top-ranked movies from the CSV.

Optional query parameters:

- `limit`: 1–50 (default 10)
- `genre`: exact case-insensitive genre filter, e.g. `Animation`
- `high_rating_only`: `true` to keep only movies with at least 50% predicted probability of AudienceScore >= 75

Example:

```bash
curl "http://127.0.0.1:8000/recommendations?limit=10&genre=Animation"
```

The response includes the predicted audience score and the logistic model's high-rating probability as a percentage.

### `POST /predict`

Predicts a score and high-rating probability for a movie described by its available attributes. Fields can be omitted; the saved pipeline imputes missing values.

Example JSON:

```json
{
  "RottenTomatoes": 90,
  "Year": 2010,
  "Budget": 150,
  "Genre": "Animation",
  "LeadStudio": "Disney",
  "Story": "Quest"
}
```

Example request:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d "{\"RottenTomatoes\":90,\"Year\":2010,\"Budget\":150,\"Genre\":\"Animation\",\"LeadStudio\":\"Disney\",\"Story\":\"Quest\"}"
```

`Budget` should use the same units as the CSV. In the supplied dataset, budget values appear to be in millions; confirm the dataset's convention before sending values.

## Swift/iOS integration

If FastAPI runs on the same Mac as the iOS simulator, use `http://127.0.0.1:8000`. If FastAPI runs on a Windows computer and the app runs on a different Mac/device, use the Windows computer's reachable LAN IP, such as `http://192.168.1.25:8000`.

For cross-device access:

1. Start Uvicorn with `--host 0.0.0.0`.
2. Allow inbound TCP port 8000 in Windows Firewall for your private network.
3. Ensure the devices are on a network that permits connections.
4. Use HTTPS for production. Plain HTTP is suitable only for local development/testing; iOS App Transport Security may block cleartext HTTP depending on configuration.

Example Windows command:

```powershell
uvicorn main:app --host 0.0.0.0 --port 8000
```

Then visit `http://WINDOWS_LAN_IP:8000/docs` from a machine that can reach the Windows host.

## Important notes

- `AudienceScore >= 75` is the definition of a “high rating” used for Logistic Regression.
- The predicted probability is a model estimate, not a guarantee or necessarily a calibrated probability.
- Recommendations are based on rating-related historical features, not personal taste or viewing history.
- `joblib` model files should only be loaded from trusted sources.
- Do not expose this development server directly to the public internet.
