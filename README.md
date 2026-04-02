# Formation Energy Predictor

A FastAPI web app for predicting formation energy of materials from chemical formulas using a Random Forest model trained on the [JARVIS Leaderboard](https://github.com/atomgptlab/jarvis_leaderboard) SSUB benchmark.

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model

```bash
python train.py
```

This will:
- Clone and install `jarvis_leaderboard`
- Download the SSUB formation energy benchmark data
- Train a Random Forest (200 trees) on element fraction descriptors
- Save the model to `rf_form_energy_model.pkl`
- Print test MAE and save prediction results

### 3. Run the app

```bash
python app.py
```

Visit [http://localhost:8000](http://localhost:8000) for the web UI, or [http://localhost:8000/docs](http://localhost:8000/docs) for the Swagger API docs.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/` | Web UI |
| `GET`  | `/predict?formula=GaN` | Single prediction |
| `POST` | `/predict_batch` | Batch prediction (JSON body: `{"formulas": ["GaN", "MoS2"]}`) |
| `GET`  | `/health` | Health check |
| `GET`  | `/docs` | Swagger UI |

## Example

```bash
curl "http://localhost:8000/predict?formula=SrTiO3"
```

```json
{
  "formula": "SrTiO3",
  "predicted_formation_energy_eV_per_atom": -1.8234
}
```

## Project Structure

```
├── app.py              # FastAPI server
├── train.py            # Model training script
├── static/
│   └── index.html      # Web UI
├── requirements.txt
└── README.md
```

## References

- [JARVIS-Tools](https://github.com/atomgptlab/jarvis)
- [JARVIS Leaderboard](https://github.com/atomgptlab/jarvis_leaderboard)
- [AtomGPTLab](https://atomgpt.org)
