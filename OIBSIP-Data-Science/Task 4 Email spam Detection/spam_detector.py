"""
Email/SMS Spam Detection with Machine Learning
================================================

Builds a spam classifier that labels messages as "spam" or "ham"
(not spam) using TF-IDF features and a Naive Bayes classifier.

Dataset: data/spam.csv
Source: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

Outputs (saved to outputs/):
    class_distribution.png
    confusion_matrix.png
    top_spam_words.png
    model.pkl
"""

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

sns.set_style("whitegrid")

DATA_PATH = os.path.join("data", "spam.csv")
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df = df.dropna(subset=["label", "message"])
    return df


def explore(df):
    print("Dataset shape:", df.shape)
    print("\nClass distribution:\n", df["label"].value_counts())

    plt.figure(figsize=(6, 5))
    sns.countplot(data=df, x="label", hue="label",
                   palette={"ham": "#55A868", "spam": "#C44E52"}, legend=False)
    plt.title("Class Distribution: Ham vs Spam")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "class_distribution.png"), dpi=120)
    plt.close()


def train_model(df):
    X = df["message"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = MultinomialNB(alpha=0.1)
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)

    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {acc:.2%}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred, labels=["ham", "spam"])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["ham", "spam"], yticklabels=["ham", "spam"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=120)
    plt.close()

    # Top words associated with spam
    feature_names = vectorizer.get_feature_names_out()
    spam_idx = list(model.classes_).index("spam")
    log_prob = model.feature_log_prob_[spam_idx]
    top_n = 15
    top_indices = log_prob.argsort()[-top_n:][::-1]
    top_words = pd.Series(log_prob[top_indices], index=feature_names[top_indices])

    plt.figure(figsize=(8, 6))
    sns.barplot(x=top_words.values, y=top_words.index, hue=top_words.index,
                palette="Reds_r", legend=False)
    plt.title("Top Words Associated with Spam")
    plt.xlabel("Log Probability")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "top_spam_words.png"), dpi=120)
    plt.close()

    joblib.dump({"model": model, "vectorizer": vectorizer}, os.path.join(OUTPUT_DIR, "model.pkl"))
    print(f"\nModel saved to {OUTPUT_DIR}/model.pkl")

    return model, vectorizer


def predict_message(model, vectorizer, message):
    vec = vectorizer.transform([message])
    return model.predict(vec)[0]


def main():
    df = load_data(DATA_PATH)
    explore(df)
    model, vectorizer = train_model(df)

    samples = [
        "Congratulations! You've won a free ticket to Bahamas, call now!",
        "Hey, are we still meeting for lunch tomorrow?",
    ]
    for msg in samples:
        print(f"\nMessage: {msg!r}\nPrediction: {predict_message(model, vectorizer, msg)}")


if __name__ == "__main__":
    main()
