import ast
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from xgboost import XGBRegressor


DATA_PATH = Path("data/embbeded_data.csv")
MODELS_DIR = Path("models")
RESULTS_CSV = MODELS_DIR / "xgboost_hyperparameter_results.csv"
BEST_MODEL_PATH = MODELS_DIR / "best_xgboost_model.pkl"
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
MLFLOW_EXPERIMENT_NAME = "ResumeScreening_Models"


def prepare_training_data(df, target_column="matched_score", test_size=0.2, random_state=42):
    df_processed = df.copy()

    for col in df_processed.columns:
        if col != target_column:
            first_val = df_processed[col].iloc[0]
            if isinstance(first_val, str) and first_val.startswith("["):
                try:
                    df_processed[col] = df_processed[col].apply(
                        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
                    )
                except (ValueError, SyntaxError):
                    pass

    expanded_data = {}
    for col in df_processed.columns:
        if col == target_column:
            continue

        first_val = df_processed[col].iloc[0]
        if isinstance(first_val, (list, np.ndarray)):
            embeddings = []
            for val in df_processed[col]:
                if isinstance(val, (list, np.ndarray)):
                    embeddings.append(np.array(val).flatten())
                else:
                    embeddings.append(np.array([val]))

            embedding_array = np.vstack(embeddings)
            for i in range(embedding_array.shape[1]):
                expanded_data[f"{col}_dim_{i}"] = embedding_array[:, i].astype(float)
        else:
            expanded_data[col] = pd.to_numeric(df_processed[col], errors="coerce").astype(float)

    X = pd.DataFrame(expanded_data)
    y = df_processed[target_column].astype(float)
    X = X.fillna(X.mean())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    print(f"Training set shape: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"Testing set shape: X_test={X_test.shape}, y_test={y_test.shape}")

    return X_train, X_test, y_train, y_test


def save_search_results(search, csv_path: Path):
    results = pd.DataFrame(search.cv_results_)
    params = pd.DataFrame(results["params"].tolist())
    results = pd.concat(
        [
            results[
                ["mean_test_score", "std_test_score", "rank_test_score"]
            ].reset_index(drop=True),
            params.reset_index(drop=True),
        ],
        axis=1,
    )
    results["mean_test_mse"] = -results["mean_test_score"]
    results["mean_test_rmse"] = np.sqrt(results["mean_test_mse"])
    results = results.drop(columns=["mean_test_score"])
    results.to_csv(csv_path, index=False)
    print(f"Saved hyperparameter search results to {csv_path}")


def tune_xgb_model(
    X_train,
    y_train,
    X_test,
    y_test,
    models_dir: Path,
    results_csv_path: Path,
    best_model_path: Path,
):
    models_dir.mkdir(parents=True, exist_ok=True)

    param_distributions = {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.7, 1.0],
        "colsample_bytree": [0.7, 1.0],
    }

    estimator = XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )

    randomized_search = RandomizedSearchCV(
        estimator,
        param_distributions,
        n_iter=20,
        cv=5,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        refit=True,
        random_state=42,
        verbose=1,
    )

    print("Starting XGBoost randomized hyperparameter tuning...")
    randomized_search.fit(X_train, y_train)

    save_search_results(randomized_search, results_csv_path)

    best_model = randomized_search.best_estimator_
    y_pred = best_model.predict(X_test)
    best_metrics = {
        "mae": mean_absolute_error(y_test, y_pred),
        "mse": mean_squared_error(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "r2": r2_score(y_test, y_pred),
        "best_cv_mse": -randomized_search.best_score_,
    }

    joblib.dump(best_model, best_model_path)
    print(f"Best XGBoost model saved to {best_model_path}")
    print(
        "Best parameters:", randomized_search.best_params_,
    )
    print(
        "Test metrics - MAE: {:.4f}, RMSE: {:.4f}, R2: {:.4f}".format(
            best_metrics["mae"], best_metrics["rmse"], best_metrics["r2"]
        )
    )

    with mlflow.start_run(run_name="XGBHyperparameterSearch"):
        mlflow.log_param("model_type", "XGBoost")
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_params(randomized_search.best_params_)
        mlflow.log_metric("best_cv_mse", best_metrics["best_cv_mse"])
        mlflow.log_metric("mae", best_metrics["mae"])
        mlflow.log_metric("mse", best_metrics["mse"])
        mlflow.log_metric("rmse", best_metrics["rmse"])
        mlflow.log_metric("r2", best_metrics["r2"])
        mlflow.sklearn.log_model(best_model, "best_xgboost_model")
        mlflow.log_artifact(str(best_model_path))
        mlflow.log_artifact(str(results_csv_path))

    return randomized_search, best_metrics


def main():
    df = pd.read_csv(DATA_PATH)
    X_train, X_test, y_train, y_test = prepare_training_data(df)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    tune_xgb_model(
        X_train,
        y_train,
        X_test,
        y_test,
        models_dir=MODELS_DIR,
        results_csv_path=RESULTS_CSV,
        best_model_path=BEST_MODEL_PATH,
    )


if __name__ == "__main__":
    main()





