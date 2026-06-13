"""
Car Price Prediction with Machine Learning
============================================

Trains a regression model to predict the price of a car based on its
specifications (brand, fuel type, engine size, horsepower, mileage, etc.)

Dataset: data/CarPrice.csv
Source: https://www.kaggle.com/datasets/vijayaadithyanvg/car-price-predictionused-cars

Outputs (saved to outputs/):
    correlation_heatmap.png
    price_distribution.png
    feature_importance.png
    actual_vs_predicted.png
    model.pkl
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

sns.set_style("whitegrid")

DATA_PATH = os.path.join("data", "CarPrice.csv")
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    # Extract the car brand/company name from CarName (e.g. "alfa-romero giulia" -> "alfa-romero")
    df["CompanyName"] = df["CarName"].apply(lambda x: x.split(" ")[0].lower())

    # Fix common misspellings of brand names
    brand_fixes = {
        "maxda": "mazda", "porcshce": "porsche", "toyouta": "toyota",
        "vokswagen": "volkswagen", "vw": "volkswagen",
    }
    df["CompanyName"] = df["CompanyName"].replace(brand_fixes)

    df = df.drop(columns=["car_ID", "CarName"])
    return df


def explore(df):
    print("Dataset shape:", df.shape)
    print("\nSummary statistics (price):\n", df["price"].describe())

    # Price distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(df["price"], kde=True, color="#4C72B0")
    plt.title("Distribution of Car Prices")
    plt.xlabel("Price")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "price_distribution.png"), dpi=120)
    plt.close()

    # Correlation heatmap of numeric features
    numeric_df = df.select_dtypes(include=np.number)
    plt.figure(figsize=(12, 10))
    sns.heatmap(numeric_df.corr(), annot=False, cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap of Numeric Features")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "correlation_heatmap.png"), dpi=120)
    plt.close()

    print("\nTop correlations with price:")
    print(numeric_df.corr()["price"].sort_values(ascending=False).head(10))


def prepare_features(df):
    X = df.drop(columns=["price"])
    y = df["price"]

    # One-hot encode categorical columns
    X = pd.get_dummies(X, drop_first=True)
    return X, y


def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"\nR2 Score: {r2:.4f}")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")

    # Actual vs predicted plot
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, color="#55A868")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("Actual vs Predicted Car Prices")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "actual_vs_predicted.png"), dpi=120)
    plt.close()

    # Feature importance (top 15)
    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False).head(15)
    plt.figure(figsize=(10, 8))
    sns.barplot(x=importances.values, y=importances.index, hue=importances.index,
                palette="viridis", legend=False)
    plt.title("Top 15 Feature Importances")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"), dpi=120)
    plt.close()

    joblib.dump({"model": model, "columns": list(X.columns)}, os.path.join(OUTPUT_DIR, "model.pkl"))
    print(f"\nModel saved to {OUTPUT_DIR}/model.pkl")

    return model


def main():
    df = load_data(DATA_PATH)
    explore(df)

    X, y = prepare_features(df)
    train_model(X, y)


if __name__ == "__main__":
    main()
