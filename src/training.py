import pandas as pd
import numpy as np
import ast
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split    
from pathlib import Path
import mlflow
import mlflow.sklearn


df = pd.read_csv("data/embbeded_data.csv")

global_db = "sqlite:///Users/vijais/ml_history/mlflow.db"
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


mlflow.set_tracking_uri(global_db)

# Set MLflow experiment
mlflow.set_experiment("ResumeScreening_Models")





