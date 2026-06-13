# Unemployment Analysis in India 🇮🇳

A data science project analyzing the unemployment rate in India, with a
focus on the sharp spike caused by the COVID-19 lockdown in 2020.

## 📌 Overview

Unemployment is measured by the **unemployment rate**: the percentage of
the labour force that is unemployed. This project explores unemployment
data across Indian states, comparing rural vs urban areas and tracking
how the rate changed before and during the COVID-19 pandemic.

## 📂 Dataset

- `data/Unemployment_in_India.csv` — state-wise unemployment data (May 2019 – Jun 2020)
- `data/Unemployment_Rate_upto_11_2020.csv` — extended data through November 2020, including region and coordinates

Source: [Unemployment in India (Kaggle)](https://www.kaggle.com/datasets/gokulrajkmv/unemployment-in-india)

Columns include:
- `Region` — Indian state/UT
- `Date` — observation date
- `Frequency` — monthly
- `Estimated Unemployment Rate (%)`
- `Estimated Employed`
- `Estimated Labour Participation Rate (%)`
- `Area` — Rural / Urban

## 🛠️ Setup

```bash
git clone https://github.com/<your-username>/unemployment-analysis-india.git
cd unemployment-analysis-india
pip install -r requirements.txt
```

## ▶️ Usage

```bash
python analysis.py
```

This will print summary statistics and save the following charts to `outputs/`:

| File | Description |
|---|---|
| `unemployment_trend.png` | Overall unemployment rate trend over time |
| `unemployment_by_region.png` | Average unemployment rate by state/region |
| `rural_vs_urban.png` | Rural vs urban unemployment rate over time |
| `lockdown_impact.png` | Monthly average unemployment rate in 2020 (COVID impact) |

## 📊 Key Findings

- India's average monthly unemployment rate hovered around **10%** from January–March 2020.
- It spiked to **23.6%** in April 2020 and peaked at **24.9%** in May 2020, coinciding with the COVID-19 lockdown.
- By June 2020, it had partially recovered to **11.9%**.
- On April 30, 2020, **Puducherry** recorded the highest unemployment rates (76.7% urban, 74.5% rural), followed by **Jharkhand**, **Bihar**, and **Tamil Nadu**.
- States like **Tripura**, **Haryana**, and **Jharkhand** had the highest average unemployment rates over the full period.

## 📁 Project Structure

```
unemployment-analysis-india/
├── data/
│   ├── Unemployment_in_India.csv
│   └── Unemployment_Rate_upto_11_2020.csv
├── outputs/
│   ├── unemployment_trend.png
│   ├── unemployment_by_region.png
│   ├── rural_vs_urban.png
│   └── lockdown_impact.png
├── analysis.py
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## 📜 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
