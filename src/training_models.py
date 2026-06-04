import pandas as pd
import numpy as np
import joblib 
import ast
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split    
from pathlib import Path
import mlflow
import mlflow.sklearn
    

df = pd.read_csv("data/embedded_data.csv")

global_db = "sqlite:///mlflow.db"


def prepare_training_data(df, target_column='matched_score', test_size=0.2, random_state=42):
    
    df_processed = df.copy()
    
    # Parse string representations of lists in embedding columns
    for col in df_processed.columns:
        if col != target_column:
            # Check if the column contains string representations of lists
            first_val = df_processed[col].iloc[0]
            if isinstance(first_val, str) and first_val.startswith('['):
                try:
                    # Convert string representations to actual lists
                    df_processed[col] = df_processed[col].apply(
                        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
                    )
                except (ValueError, SyntaxError):
                    pass
    
    # Expand list/array columns into separate numeric columns
    expanded_data = {}
    for col in df_processed.columns:
        if col == target_column:
            continue
        
        first_val = df_processed[col].iloc[0]
        
        # If column contains lists/arrays, expand them
        if isinstance(first_val, (list, np.ndarray)):
            # Convert each embedding to a separate row of features
            embeddings = []
            for val in df_processed[col]:
                if isinstance(val, (list, np.ndarray)):
                    embeddings.append(np.array(val).flatten())
                else:
                    embeddings.append(np.array([val]))
            
            # Stack all embeddings into a 2D array
            embedding_array = np.vstack(embeddings)
            
            # Add each dimension as a separate column
            for i in range(embedding_array.shape[1]):
                expanded_data[f"{col}_dim_{i}"] = embedding_array[:, i].astype(float)
        else:
            # For scalar columns, just convert to numeric
            expanded_data[col] = pd.to_numeric(df_processed[col], errors='coerce').astype(float)
    
    # Create new dataframe with expanded features
    X = pd.DataFrame(expanded_data)
    y = df_processed[target_column].astype(float)  # Ensure target is numeric
    
    # Handle any NaN values
    X = X.fillna(X.mean())
    
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state
    )
    
    print(f"Training set shape: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"Testing set shape: X_test={X_test.shape}, y_test={y_test.shape}")
    
    return X_train, X_test, y_train, y_test


# Prepare the data
X_train, X_test, y_train, y_test = prepare_training_data(df)

# ============================================================================
# Train Models
# ============================================================================

 # Create models folder if it doesn't exist
models_dir = Path("models")
models_dir.mkdir(exist_ok=True)
    
models_dict = {}


mlflow.set_tracking_uri(global_db)

# Set MLflow experiment
mlflow.set_experiment("ResumeScreening_Models")
    
 # ---- 1. Linear Regression ----
print("\n" + "="*50)
print("Training Linear Regression Model...")
print("="*50)
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
    
lr_metrics = {
        'model': lr_model,
        'mae': mean_absolute_error(y_test, y_pred_lr),
        'mse': mean_squared_error(y_test, y_pred_lr),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred_lr)),
        'r2': r2_score(y_test, y_pred_lr)
    }
    
lr_path = models_dir / "linear_regression_model.pkl"
joblib.dump(lr_model, lr_path)
with mlflow.start_run(run_name="LinearRegression"):
    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_param("n_features", X_train.shape[1])
    mlflow.log_metric("mae", lr_metrics['mae'])
    mlflow.log_metric("mse", lr_metrics['mse'])
    mlflow.log_metric("rmse", lr_metrics['rmse'])
    mlflow.log_metric("r2", lr_metrics['r2'])
    # save and log model artifact
    mlflow.sklearn.log_model(lr_model, "model")
    mlflow.log_artifact(str(lr_path))

print(f"Linear Regression model saved to {lr_path}")
print(f"Metrics - MAE: {lr_metrics['mae']:.4f}, RMSE: {lr_metrics['rmse']:.4f}, R2: {lr_metrics['r2']:.4f}")
    

    
# ---- 2. Random Forest Regressor ----
print("\n" + "="*50)
print("Training Random Forest Regressor Model...")
print("="*50)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
    
rf_metrics = {
        'model': rf_model,
        'mae': mean_absolute_error(y_test, y_pred_rf),
        'mse': mean_squared_error(y_test, y_pred_rf),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred_rf)),
        'r2': r2_score(y_test, y_pred_rf)
    }
    
rf_path = models_dir / "random_forest_model.pkl"
joblib.dump(rf_model, rf_path)
with mlflow.start_run(run_name="RandomForest"):
    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_param("n_estimators", rf_model.n_estimators)
    mlflow.log_param("n_features", X_train.shape[1])
    mlflow.log_metric("mae", rf_metrics['mae'])
    mlflow.log_metric("mse", rf_metrics['mse'])
    mlflow.log_metric("rmse", rf_metrics['rmse'])
    mlflow.log_metric("r2", rf_metrics['r2'])
    mlflow.sklearn.log_model(rf_model, "model")
    mlflow.log_artifact(str(rf_path))

print(f"Random Forest model saved to {rf_path}")
print(f"Metrics - MAE: {rf_metrics['mae']:.4f}, RMSE: {rf_metrics['rmse']:.4f}, R2: {rf_metrics['r2']:.4f}")
    

    
# ---- 3. XGBoost Regressor ----
print("\n" + "="*50)
print("Training XGBoost Regressor Model...")
print("="*50)
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
    
xgb_metrics = {
        'model': xgb_model,
        'mae': mean_absolute_error(y_test, y_pred_xgb),
        'mse': mean_squared_error(y_test, y_pred_xgb),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred_xgb)),
        'r2': r2_score(y_test, y_pred_xgb)
    }
    
xgb_path = models_dir / "xgboost_model.pkl"
joblib.dump(xgb_model, xgb_path)
with mlflow.start_run(run_name="XGBoost"):
    mlflow.log_param("model_type", "XGBoost")
    try:
        mlflow.log_param("n_estimators", xgb_model.get_params().get('n_estimators'))
        mlflow.log_param("learning_rate", xgb_model.get_params().get('learning_rate'))
    except Exception:
        pass
    mlflow.log_param("n_features", X_train.shape[1])
    mlflow.log_metric("mae", xgb_metrics['mae'])
    mlflow.log_metric("mse", xgb_metrics['mse'])
    mlflow.log_metric("rmse", xgb_metrics['rmse'])
    mlflow.log_metric("r2", xgb_metrics['r2'])
    # XGBoost model can be logged with sklearn wrapper
    mlflow.sklearn.log_model(xgb_model, "model")
    mlflow.log_artifact(str(xgb_path))

print(f"XGBoost model saved to {xgb_path}")
print(f"Metrics - MAE: {xgb_metrics['mae']:.4f}, RMSE: {xgb_metrics['rmse']:.4f}, R2: {xgb_metrics['r2']:.4f}")
    





