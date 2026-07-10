# ==========================================================
# STUDENT PERFORMANCE PREDICTION MODEL
# ==========================================================

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ==========================================================
# LOAD DATASET
# ==========================================================

df = pd.read_csv("sample_dataset.csv")

print("=" * 60)
print("DATASET LOADED")
print("=" * 60)
print(df.head())

print("\nDataset Shape:", df.shape)

# ==========================================================
# FEATURES & TARGET
# ==========================================================

drop_columns = [
    "performance_index",
    "result",
    "grade",
    "performance_category"
]

X = df.drop(columns=drop_columns)
y = df["performance_index"]

# ==========================================================
# COLUMN TYPES
# ==========================================================

categorical_features = X.select_dtypes(include="object").columns.tolist()
numerical_features = X.select_dtypes(exclude="object").columns.tolist()

print("\nCategorical Features")
print(categorical_features)

print("\nNumerical Features")
print(numerical_features)

# ==========================================================
# NUMERIC PIPELINE
# ==========================================================

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

# ==========================================================
# CATEGORICAL PIPELINE
# ==========================================================

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)

# ==========================================================
# PREPROCESSOR
# ==========================================================

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# ==========================================================
# RANDOM FOREST MODEL
# ==========================================================

rf = RandomForestRegressor(
    n_estimators=50,      # Reduced model size
    random_state=42,
    n_jobs=-1
)

# ==========================================================
# COMPLETE PIPELINE
# ==========================================================

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", rf)
    ]
)

# ==========================================================
# TRAIN TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ==========================================================
# TRAIN MODEL
# ==========================================================

print("\nTraining Model...\n")

model.fit(X_train, y_train)

print("Training Completed!")

# ==========================================================
# PREDICTIONS
# ==========================================================

predictions = model.predict(X_test)

# ==========================================================
# EVALUATION
# ==========================================================

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(f"MAE       : {mae:.2f}")
print(f"RMSE      : {rmse:.2f}")
print(f"R² Score  : {r2:.4f}")

# ==========================================================
# ACTUAL VS PREDICTED
# ==========================================================

comparison = pd.DataFrame({
    "Actual": y_test,
    "Predicted": predictions
})

print("\nSample Predictions")

print(comparison.head(10))

# ==========================================================
# SAVE PREDICTION RESULTS
# ==========================================================

comparison.to_csv(
    "prediction_results.csv",
    index=False
)

print("\nPrediction Results Saved!")

# ==========================================================
# SAVE MODEL
# ==========================================================

joblib.dump(
    model,
    "student_performance_pipeline.pkl",
    compress=5
)

print("\nPipeline Saved Successfully!")

# ==========================================================
# PROJECT COMPLETE
# ==========================================================

print("\n" + "=" * 60)
print("PROJECT COMPLETED SUCCESSFULLY")
print("=" * 60)