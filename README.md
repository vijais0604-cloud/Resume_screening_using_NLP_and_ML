# Resume Screening ML Project

This project trains regression models for resume/job matching using embedded resume and job data. It includes data preprocessing, model training, hyperparameter tuning for XGBoost, MLflow logging, and model artifact storage.

## Project explanation

The goal of this project is to predict a match score between resumes and job descriptions. The process is:

- Load preprocessed resume/job data from `data/embbeded_data.csv`.
- Convert embedded feature columns (arrays/lists) into numeric feature vectors.
- Train and evaluate regression models to estimate the `matched_score` target.
- Perform hyperparameter tuning for XGBoost using randomized search to find the best configuration.
- Save the best XGBoost model as a `.pkl` artifact and export search results to CSV.
- Track experiments, parameters, metrics, and artifacts with MLflow for reproducibility.

The main script `src/training.py` is designed for XGBoost tuning, while `src/training_models.py` contains a broader training workflow for multiple model types.

## Repository structure

- `data/` - source datasets and preprocessed CSVs
- `models/` - trained model artifacts saved as `.pkl`
- `mlruns/` - MLflow local experiment tracking directory
- `src/` - Python scripts for training and preprocessing
  - `src/training_models.py` - legacy training script for Linear Regression, Random Forest, and XGBoost
  - `src/training.py` - XGBoost hyperparameter tuning, CSV result export, MLflow logging, and best model artifact creation
- `requirements.txt` - Python package dependencies

## Setup

1. Clone or copy the repository into your workspace.

2. Create and activate a Python virtual environment.

```bash
python3 -m venv .resume
source .resume/bin/activate
```

3. Install dependencies.

```bash
pip install -r requirements.txt
```

4. Ensure the `data/embbeded_data.csv` file is present and contains the expected features and `matched_score` target column.

## Docker

This repository includes a Dockerfile named `dockerfile` that builds a Python 3.11 image and starts the FastAPI backend on port `8000`.

Build the Docker image from the project root:

```bash
docker build -t resume-screener -f dockerfile .
```

Run the backend container:

```bash
docker run --rm -p 8000:8000 resume-screener
```

If you want the running container to use model artifacts from the local repository path, mount `src/model`:

```bash
docker run --rm -p 8000:8000 -v "$(pwd)/src/model:/app/src/model" resume-screener
```

Then open the API at:

```text
http://localhost:8000/predict
```

## Running hyperparameter tuning for XGBoost

The tuning script performs randomized search over XGBoost hyperparameters, logs the best model to MLflow, and saves the tuning results to CSV.

```bash
source .resume/bin/activate
python src/training.py
```

### What it produces

- `models/best_xgboost_model.pkl` - saved best XGBoost model
- `models/xgboost_hyperparameter_results.csv` - grid search result summary
- MLflow run logged to the tracking server defined in `src/training.py`

## Backend deployment and serving

This project includes a FastAPI backend in `app/main.py` that exposes a `/predict` endpoint for resume-to-job matching.

### Start the backend server

From the project root, activate the virtual environment and run:

```bash
source .resume/bin/activate
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Request format

The backend expects a PDF resume upload plus a job description string:

- `resume`: PDF file uploaded as form data
- `job_description`: text field in the same form

Example using `curl`:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -F "resume=@/path/to/resume.pdf" \
  -F "job_description=Hiring software engineer with Python and ML experience"
```

### Deployment notes

- `app/predictor.py` loads `models/best_xgboost_model.pkl` and `models/sentence_transformer_model.joblib`
- Ensure you run `uvicorn` from the project root so those relative paths resolve correctly
- The prediction output includes:
  - `ats_score`
  - `skill_score`
  - `semantic_score`
  - `experience_score`

## Viewing MLflow logs

By default this project uses a local SQLite MLflow store at `sqlite:///mlflow.db`.

Start the MLflow UI from the project root:

```bash
source .resume/bin/activate
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Then open:

```text
http://localhost:5000
```

## Notes

- The project also contains `src/training_models.py` for training multiple models and logging them to MLflow.
- If you want to retrain only the existing models, run:

```bash
source .resume/bin/activate
python src/training_models.py
```

- Ensure `models/` directory exists or is created automatically by the training scripts.

## Troubleshooting

- If `python` is not found, use `python3`.
- If MLflow logs do not appear, verify the `mlflow ui` command is using the same tracking URI as the script.
- If dependencies are missing, re-run:

```bash
pip install -r requirements.txt
```
