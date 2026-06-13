# Car Price Prediction with Machine Learning 🚗

A regression project that predicts the price of a car based on its
specifications — brand, fuel type, body style, engine size, horsepower,
mileage, and more.

## 📌 Overview

The price of a car depends on many factors: brand goodwill, features,
engine specs, horsepower, and mileage, among others. This project trains
a Random Forest regression model to predict a car's price from its
specifications.

## 📂 Dataset

`data/CarPrice.csv` — 205 cars with 26 features including:

- `CarName` (parsed into `CompanyName`)
- `fueltype`, `aspiration`, `doornumber`, `carbody`, `drivewheel`, `enginelocation`
- `wheelbase`, `carlength`, `carwidth`, `carheight`, `curbweight`
- `enginetype`, `cylindernumber`, `enginesize`, `fuelsystem`
- `boreratio`, `stroke`, `compressionratio`, `horsepower`, `peakrpm`
- `citympg`, `highwaympg`
- `price` (target)

Source: [Car Price Prediction (Kaggle)](https://www.kaggle.com/datasets/vijayaadithyanvg/car-price-predictionused-cars)

## 🛠️ Setup

```bash
git clone https://github.com/<your-username>/car-price-prediction.git
cd car-price-prediction
pip install -r requirements.txt
```

## ▶️ Usage

```bash
python predict_price.py
```

This will:
1. Load and clean the data (parsing brand names, fixing misspellings).
2. Explore price distribution and feature correlations.
3. One-hot encode categorical features.
4. Train a Random Forest regressor on an 80/20 train-test split.
5. Print R², MAE, and RMSE on the test set.
6. Save evaluation charts and the trained model to `outputs/`.

## 📊 Results

| Metric | Value |
|---|---|
| R² Score | 0.96 |
| MAE | ~1248 |
| RMSE | ~1787 |

The model explains about **96%** of the variance in car prices.

### Top price correlations

| Feature | Correlation with price |
|---|---|
| Engine size | 0.87 |
| Curb weight | 0.84 |
| Horsepower | 0.81 |
| Car width | 0.76 |
| Car length | 0.68 |

Larger, heavier cars with bigger engines and more horsepower command
higher prices, as expected.

### Outputs

| File | Description |
|---|---|
| `outputs/price_distribution.png` | Distribution of car prices |
| `outputs/correlation_heatmap.png` | Correlation heatmap of numeric features |
| `outputs/feature_importance.png` | Top 15 most important features for prediction |
| `outputs/actual_vs_predicted.png` | Actual vs predicted prices on the test set |
| `outputs/model.pkl` | Trained model + feature columns (generated, not committed) |

## 🔮 Predicting New Cars

```python
import joblib
import pandas as pd

artifact = joblib.load("outputs/model.pkl")
model, columns = artifact["model"], artifact["columns"]

# Build a single-row dataframe matching the training features,
# then one-hot encode and align columns before predicting:
# new_car = pd.get_dummies(new_car_df).reindex(columns=columns, fill_value=0)
# predicted_price = model.predict(new_car)
```

## 📁 Project Structure

```
car-price-prediction/
├── data/
│   └── CarPrice.csv
├── outputs/
│   ├── price_distribution.png
│   ├── correlation_heatmap.png
│   ├── feature_importance.png
│   └── actual_vs_predicted.png
├── predict_price.py
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## 📜 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
