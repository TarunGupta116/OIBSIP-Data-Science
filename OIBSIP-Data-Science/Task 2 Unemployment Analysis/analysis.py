"""
Unemployment Analysis in India (with COVID-19 impact)
======================================================

This script loads, cleans, and analyzes unemployment data for India,
producing visualizations that highlight regional differences, rural
vs urban trends, and the impact of the COVID-19 lockdown on the
unemployment rate.

Datasets:
    data/Unemployment_in_India.csv
    data/Unemployment_Rate_upto_11_2020.csv

Outputs (saved to outputs/):
    unemployment_trend.png
    unemployment_by_region.png
    rural_vs_urban.png
    lockdown_impact.png
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

DATA_DIR = "data"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_and_clean(path, date_format="%d-%m-%Y"):
    """Load a CSV, strip whitespace from columns/values, parse dates, drop NA."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()

    df["Date"] = pd.to_datetime(df["Date"], format=date_format)
    df = df.dropna()
    return df


def plot_overall_trend(df, out_path):
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x="Date", y="Estimated Unemployment Rate (%)", errorbar=None)
    plt.title("Estimated Unemployment Rate in India Over Time")
    plt.xlabel("Date")
    plt.ylabel("Unemployment Rate (%)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def plot_region_avg(df, out_path):
    region_avg = (
        df.groupby("Region")["Estimated Unemployment Rate (%)"]
        .mean()
        .sort_values(ascending=False)
    )
    plt.figure(figsize=(12, 8))
    sns.barplot(x=region_avg.values, y=region_avg.index, hue=region_avg.index,
                palette="viridis", legend=False)
    plt.title("Average Unemployment Rate by Region (India)")
    plt.xlabel("Average Unemployment Rate (%)")
    plt.ylabel("Region")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    return region_avg


def plot_rural_vs_urban(df, out_path):
    area_trend = (
        df.groupby(["Date", "Area"])["Estimated Unemployment Rate (%)"]
        .mean()
        .reset_index()
    )
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=area_trend, x="Date", y="Estimated Unemployment Rate (%)", hue="Area")
    plt.title("Unemployment Rate: Rural vs Urban Over Time")
    plt.xlabel("Date")
    plt.ylabel("Unemployment Rate (%)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def plot_lockdown_impact(df, out_path):
    monthly_avg = (
        df[df["Date"] >= "2020-01-01"]
        .groupby("Date")["Estimated Unemployment Rate (%)"]
        .mean()
    )
    plt.figure(figsize=(10, 6))
    labels = monthly_avg.index.strftime("%Y-%m")
    plt.bar(labels, monthly_avg.values, color="#FF8C42")
    plt.title("Average Monthly Unemployment Rate in India (2020)")
    plt.xlabel("Month")
    plt.ylabel("Unemployment Rate (%)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    return monthly_avg


def top_states_during_lockdown(df, date="2020-04-30", n=10):
    subset = df[df["Date"] == date]
    if subset.empty:
        return None
    return subset.sort_values(
        "Estimated Unemployment Rate (%)", ascending=False
    ).head(n)[["Region", "Area", "Estimated Unemployment Rate (%)"]]


def main():
    df = load_and_clean(os.path.join(DATA_DIR, "Unemployment_in_India.csv"))

    print(f"Dataset shape after cleaning: {df.shape}")
    print("\nSummary statistics:")
    print(df.describe())

    plot_overall_trend(df, os.path.join(OUTPUT_DIR, "unemployment_trend.png"))

    region_avg = plot_region_avg(df, os.path.join(OUTPUT_DIR, "unemployment_by_region.png"))
    print("\nTop 5 regions by average unemployment rate:")
    print(region_avg.head())

    plot_rural_vs_urban(df, os.path.join(OUTPUT_DIR, "rural_vs_urban.png"))

    monthly_avg = plot_lockdown_impact(df, os.path.join(OUTPUT_DIR, "lockdown_impact.png"))
    print("\nMonthly average unemployment rate (2020):")
    print(monthly_avg)

    top_states = top_states_during_lockdown(df)
    if top_states is not None:
        print("\nTop 10 states by unemployment rate during lockdown (Apr 30, 2020):")
        print(top_states.to_string(index=False))

    print(f"\nAll charts saved to '{OUTPUT_DIR}/'")


if __name__ == "__main__":
    main()
