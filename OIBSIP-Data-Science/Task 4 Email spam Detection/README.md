# Email/SMS Spam Detection with Machine Learning 📧

A text classification project that detects whether a message is **spam**
or **ham** (not spam) using TF-IDF feature extraction and a Multinomial
Naive Bayes classifier.

## 📌 Overview

Spam mail is sent to a massive number of users at once and often contains
scams or phishing content. This project builds a spam detector that
learns from a labeled dataset of SMS messages and classifies new messages
as spam or ham.

## 📂 Dataset

`data/spam.csv` — 5,572 labeled SMS messages.

Columns:
- `label` — "ham" or "spam"
- `message` — the text of the message

Source: [SMS Spam Collection Dataset (Kaggle)](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)

## 🛠️ Setup

```bash
git clone https://github.com/<your-username>/email-spam-detection.git
cd email-spam-detection
pip install -r requirements.txt
```

## ▶️ Usage

```bash
python spam_detector.py
```

This will:
1. Load the dataset and show the class distribution (ham vs spam).
2. Convert messages to TF-IDF vectors.
3. Train a Multinomial Naive Bayes classifier on an 80/20 train-test split.
4. Print accuracy and a classification report.
5. Save evaluation charts and the trained model to `outputs/`.
6. Run sample predictions on example messages.

## 📊 Results

| Metric | Value |
|---|---|
| Accuracy | 98.65% |
| Spam Precision | 0.98 |
| Spam Recall | 0.92 |
| Spam F1-score | 0.95 |

The dataset is imbalanced (~87% ham, ~13% spam); the model still detects
spam with high precision and recall thanks to TF-IDF features and a
tuned smoothing parameter (`alpha=0.1`).

### Outputs

| File | Description |
|---|---|
| `outputs/class_distribution.png` | Ham vs spam class distribution |
| `outputs/confusion_matrix.png` | Confusion matrix on the test set |
| `outputs/top_spam_words.png` | Words most strongly associated with spam |
| `outputs/model.pkl` | Trained model + TF-IDF vectorizer (generated, not committed) |

## 🔮 Predicting New Messages

```python
import joblib

artifact = joblib.load("outputs/model.pkl")
model, vectorizer = artifact["model"], artifact["vectorizer"]

message = "Congratulations! You've won a free prize, call now!"
prediction = model.predict(vectorizer.transform([message]))
print(prediction[0])  # 'spam'
```

## 📁 Project Structure

```
email-spam-detection/
├── data/
│   └── spam.csv
├── outputs/
│   ├── class_distribution.png
│   ├── confusion_matrix.png
│   └── top_spam_words.png
├── spam_detector.py
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## 📜 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
