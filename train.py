import os
import re
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import joblib

mlflow.set_tracking_uri("sqlite:///D:/PythonProject/assigment/mlflow.db")
mlflow.set_experiment("ML_Assignment_Tuning")

def clean_pakistani_price(val):
    if pd.isna(val):
        return 0.0
    val_str = str(val).upper().strip()
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", val_str)
    
    # CRITICAL FIX: Extract the string element from the list before converting to float
    if not numbers:
        return 0.0
    num = float(numbers[0])
    
    if "CRORE" in val_str:
        num *= 10000000
    elif "LAKH" in val_str:
        num *= 100000
    elif "MILLION" in val_str:
        num *= 1000000
    return num

def train_model():
    data_path = os.path.join("data", "data.csv")
    if not os.path.exists(data_path):
        print(f"Error: Please move your data.csv to {data_path}")
        return
        
    df = pd.read_csv(data_path, nrows=3000)
    target_column = df.columns[-1]
    
    print(f"Cleaning prices inside target column: '{target_column}'...")
    y = df[target_column].apply(clean_pakistani_price)
    
    q1 = np.percentile(y, 25)
    q3 = np.percentile(y, 75)
    iqr = q3 - q1
    valid_indices = (y >= q1 - 1.5*iqr) & (y <= q3 + 1.5*iqr)
    df_cleaned = df[valid_indices].copy()
    y = y[valid_indices]
    
    X_raw = df_cleaned.drop(columns=[target_column])
    X = pd.get_dummies(X_raw, drop_first=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    param_grid = dict()
    param_grid['n_estimators'] = list((50, 100))
    param_grid['max_depth'] = list((8, 12))
    
    # Kept at n_jobs=1 to preserve your laptop's memory
    base_rf = RandomForestRegressor(random_state=42, n_jobs=1)
    
    print("\nStarting memory-safe Grid Search hyperparameter tuning...")
    print("Running sequential iterations on a single CPU core...")
    
    grid_search = GridSearchCV(estimator=base_rf, param_grid=param_grid, cv=3, scoring='r2', n_jobs=1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    print(f"\nBest Parameters Found: {grid_search.best_params_}")

    with mlflow.start_run():
        predictions = best_model.predict(X_test)
        r2 = r2_score(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        
        for param_name, param_val in grid_search.best_params_.items():
            mlflow.log_param(param_name, param_val)
        mlflow.log_param("model_type", "Tuned_RandomForest")
        mlflow.log_metric("R2_Score", r2)
        mlflow.log_metric("MAE", mae)
        mlflow.sklearn.log_model(best_model, "model")
        
        print("\n--- Hyperparameter Tuning Complete ---")
        print(f"Optimized R-squared Score: {r2 * 100:.2f}%")
        print(f"Optimized Mean Absolute Error (MAE): PKR {mae:,.2f}")
        
        joblib.dump(best_model, "trained_model.pkl")
        joblib.dump(X.columns.tolist(), "model_columns.pkl")
        print("Optimized model assets successfully updated locally!")

if __name__ == "__main__":
    train_model()
