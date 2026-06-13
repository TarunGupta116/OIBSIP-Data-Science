# Iris Flower Classification 🌸

A machine learning project that classifies Iris flowers into one of three
species — **Setosa**, **Versicolor**, and **Virginica** — based on their
sepal and petal measurements.

## 📌 Overview

The Iris dataset is a classic dataset for classification tasks. Each
sample has four numeric features (sepal length, sepal width, petal length,
petal width) and a target species label. This project trains a Random
Forest classifier to predict the species from these measurements.

## 📂 Dataset

`data/IRIS.csv` — 150 samples, 50 per species.

Columns:
- `sepal_length`
- `sepal_width`
- `petal_length`
- `petal_width`
- `species` (Iris-setosa, Iris-versicolor, Iris-virginica)

Source: [Iris Dataset (Kaggle)](https://www.kaggle.com/datasets/saurabh00007/iriscsv)

## 🛠️ Setup

```bash
git clone https://github.com/<your-username>/iris-classification.git
cd iris-classification
pip install -r requirements.txt
```

## ▶️ Usage

```bash
python classify.py
```

This will:
1. Explore the dataset and save a pairplot of feature relationships.
2. Train a Random Forest classifier on an 80/20 train-test split.
3. Print accuracy and a classification report.
4. Save a confusion matrix and feature importance chart.
5. Save the trained model to `outputs/model.pkl`.
6. Run a sample prediction.

## 📊 Results

| Metric | Score |
|---|---|
| Accuracy | 90% |
| Setosa F1-score | 1.00 |
| Versicolor F1-score | 0.86 |
| Virginica F1-score | 0.84 |

Setosa is perfectly separable from the other two species. Most
misclassifications occur between Versicolor and Virginica, which have
overlapping petal measurements — a well-known characteristic of this dataset.

### Outputs

| File | Description |
|---|---|
| `outputs/pairplot.png` | Pairwise feature relationships colored by species |
| `outputs/confusion_matrix.png` | Confusion matrix on the test set |
| `outputs/feature_importance.png` | Relative importance of each feature |
| `outputs/model.pkl` | Trained model + label encoder (generated, not committed) |

## 🔮 Predicting New Samples

```python
import joblib

artifact = joblib.load("outputs/model.pkl")
model, le = artifact["model"], artifact["label_encoder"]

sample = [[5.1, 3.5, 1.4, 0.2]]  # sepal_length, sepal_width, petal_length, petal_width
prediction = le.inverse_transform(model.predict(sample))
print(prediction[0])  # Iris-setosa
```

## 📁 Project Structure

```
iris-classification/
├── data/
│   └── IRIS.csv
├── outputs/
│   ├── pairplot.png
│   ├── confusion_matrix.png
│   └── feature_importance.png
├── classify.py
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## 📜 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
