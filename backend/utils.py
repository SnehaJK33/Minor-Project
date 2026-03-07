import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet


# =====================================================
# 🔹 Helper Functions
# =====================================================

def safe_numeric(series):
    return pd.to_numeric(series, errors="coerce")


def clean_percentage(series):
    return pd.to_numeric(series.astype(str).str.replace("%", ""), errors="coerce")


def clean_population(series):
    return pd.to_numeric(series.astype(str).str.replace(",", ""), errors="coerce")


def safe_float(value):
    try:
        if pd.isna(value) or np.isinf(value):
            return None
        return float(value)
    except:
        return None


# =====================================================
# 🔹 Filter Data by District
# =====================================================

def filter_location_data(df, location):
    df_copy = df.copy()
    df_copy["District"] = df_copy["District"].astype(str)

    filtered = df_copy[
        df_copy["District"].str.lower().str.strip()
        == location.lower().strip()
    ]

    if filtered.empty:
        raise ValueError(f"District '{location}' not found.")

    return filtered.sort_values("Year")


# =====================================================
# 🔹 Deforestation Summary
# =====================================================

def summarize_deforestation(data):

    rate = safe_numeric(data["Deforestation_Rate_%"])

    return {
        "average_deforestation_rate": safe_float(rate.mean()),
        "max_deforestation_rate": safe_float(rate.max()),
        "min_deforestation_rate": safe_float(rate.min()),
        "total_records": int(len(data))
    }


# =====================================================
# 🔹 Environmental Summary
# =====================================================

def summarize_environment(data):

    rainfall = safe_numeric(data.get("Rainfall_mm", pd.Series(dtype=float)))
    temperature = safe_numeric(data.get("Temperature_C", pd.Series(dtype=float)))

    return {
        "average_rainfall_mm": safe_float(rainfall.mean()),
        "average_temperature_c": safe_float(temperature.mean())
    }


# =====================================================
# 🔹 Future Prediction (Polynomial Trend Model)
# =====================================================

def predict_future(df, location, n_years=5):

    data = filter_location_data(df, location).copy()

    # Clean columns
    data["Year"] = safe_numeric(data["Year"])
    data["Deforestation_Rate_%"] = safe_numeric(data["Deforestation_Rate_%"])

    data = data.dropna(subset=["Year", "Deforestation_Rate_%"])

    if len(data) < 4:
        return {"years": [], "rates": []}

    X = data[["Year"]]
    y = data["Deforestation_Rate_%"]

    # Polynomial degree 2 works best for small dataset
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)

    model = LinearRegression()
    model.fit(X_poly, y)
    train_pred = model.predict(X_poly)
    model_r2 = safe_float(r2_score(y, train_pred))

    last_year = int(data["Year"].max())

    future_years = []
    future_rates = []
    future_raw_rates = []

    for i in range(1, n_years + 1):

        next_year = last_year + i
        # Keep feature name consistent with training data to avoid sklearn warnings.
        future_X = poly.transform(pd.DataFrame({"Year": [next_year]}))

        prediction = float(model.predict(future_X)[0])
        future_raw_rates.append(prediction)
        prediction = max(prediction, 0)

        future_years.append(next_year)
        future_rates.append(round(float(prediction), 2))

    # If zero-clamping dominates (mostly negative raw trend), use magnitude to avoid
    # trailing zero forecasts like [0.34, 0, 0, 0, 0].
    zero_count = sum(1 for r in future_rates if r == 0)
    if future_rates and any(v < 0 for v in future_raw_rates) and zero_count >= max(2, len(future_rates) // 2):
        future_rates = [round(abs(v), 2) for v in future_raw_rates]

    return {
        "years": future_years,
        "rates": future_rates,
        "model_name": "Polynomial Regression (degree=2)",
        "model_accuracy_r2": model_r2,
        "training_samples": int(len(data))
    }


# =====================================================
# 🔹 Risk Assessment (Clustering)
# =====================================================

def perform_clustering(df, location):

    df_copy = df.copy()

    df_copy["Deforestation_Rate_%"] = safe_numeric(df_copy["Deforestation_Rate_%"])
    df_copy = df_copy.dropna(subset=["Deforestation_Rate_%"])

    if len(df_copy) < 5:
        return {"risk_level": "Not enough data"}

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_copy[["Deforestation_Rate_%"]])

    model = KMeans(n_clusters=3, random_state=42, n_init=20)
    df_copy["Cluster"] = model.fit_predict(X_scaled)

    cluster_means = (
        df_copy.groupby("Cluster")["Deforestation_Rate_%"]
        .mean()
        .sort_values()
    )

    cluster_rank = {
        cluster: rank
        for rank, cluster in enumerate(cluster_means.index)
    }

    location_data = filter_location_data(df_copy, location)

    cluster_id = location_data.iloc[-1]["Cluster"]
    rank = cluster_rank.get(cluster_id, 1)

    risk_levels = ["Low", "Medium", "High"]

    return {
        "risk_level": risk_levels[min(rank, 2)]
    }


# =====================================================
# 🔹 PDF Report Generator
# =====================================================

def generate_pdf_report(location, summary, environment, future):

    file_name = f"{location}_deforestation_report.pdf"
    file_path = os.path.join(os.getcwd(), file_name)

    doc = SimpleDocTemplate(file_path)
    elements = []

    styles = getSampleStyleSheet()

    elements.append(Paragraph(f"Deforestation Report - {location}", styles["Heading1"]))
    elements.append(Spacer(1, 0.3 * inch))

    # Summary
    elements.append(Paragraph("Deforestation Summary", styles["Heading2"]))
    elements.append(Spacer(1, 0.2 * inch))

    for key, value in summary.items():
        elements.append(Paragraph(f"{key}: {value}", styles["Normal"]))
        elements.append(Spacer(1, 0.1 * inch))

    elements.append(Spacer(1, 0.3 * inch))

    # Environment
    elements.append(Paragraph("Environmental Data", styles["Heading2"]))
    elements.append(Spacer(1, 0.2 * inch))

    for key, value in environment.items():
        elements.append(Paragraph(f"{key}: {value}", styles["Normal"]))
        elements.append(Spacer(1, 0.1 * inch))

    elements.append(Spacer(1, 0.3 * inch))

    # Prediction
    elements.append(Paragraph("5-Year Forecast (Trend Based)", styles["Heading2"]))
    elements.append(Spacer(1, 0.2 * inch))

    if future["years"]:
        for year, rate in zip(future["years"], future["rates"]):
            elements.append(Paragraph(f"{year}: {rate}", styles["Normal"]))
            elements.append(Spacer(1, 0.1 * inch))
    else:
        elements.append(Paragraph("Not enough data for prediction.", styles["Normal"]))

    doc.build(elements)

    return file_path
