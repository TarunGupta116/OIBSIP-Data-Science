"""
Iris Flower Classification
===========================

Trains a machine learning model to classify Iris flowers (Setosa,
Versicolor, Virginica) based on their sepal and petal measurements.

Dataset: data/IRIS.csv
Source: https://www.kaggle.com/datasets/saurabh00007/iriscsv

Outputs (saved to outputs/):
    pairplot.png        - pairwise feature relationships by species
    confusion_matrix.png
    feature_importance.png
    model.pkl            - trained model
"""

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

sns.set_style("whitegrid")

DATA_PATH = os.path.join("data", "IRIS.csv")
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def explore(df):
    print("Dataset shape:", df.shape)
    print("\nClass distribution:\n", df["species"].value_counts())
    print("\nSummary statistics:\n", df.describe())

    sns.pairplot(df, hue="species", diag_kind="hist")
    plt.savefig(os.path.join(OUTPUT_DIR, "pairplot.png"), dpi=120)
    plt.close()


def train_model(df):
    X = df.drop(columns=["species"])
    y = df["species"]

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {acc:.2%}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=120)
    plt.close()

    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    plt.figure(figsize=(8, 5))
    sns.barplot(x=importances.values, y=importances.index, hue=importances.index,
                palette="viridis", legend=False)
    plt.title("Feature Importance")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"), dpi=120)
    plt.close()

    joblib.dump({"model": model, "label_encoder": le}, os.path.join(OUTPUT_DIR, "model.pkl"))
    print(f"\nModel saved to {OUTPUT_DIR}/model.pkl")

    return model, le


def predict_sample(model, le, sample):
    """sample: list of [sepal_length, sepal_width, petal_length, petal_width]"""
    pred = model.predict(sample)
    return le.inverse_transform(pred)


def main():
    df = load_data(DATA_PATH)
    explore(df)
    model, le = train_model(df)

    sample = [[5.1, 3.5, 1.4, 0.2]]
    print(f"\nSample prediction for {sample[0]}: {predict_sample(model, le, sample)[0]}")


if __name__ == "__main__":
    main()
