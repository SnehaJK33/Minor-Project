import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from fpdf import FPDF

# ---------------- Directories ----------------
BASE_DIR = os.path.abspath(os.path.dirname(_file)) if 'file_' in globals() else os.getcwd()
PLOT_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# ---------------- Static Data ----------------
DEFORESTATION_CAUSES = {
    "Alipurduar": ["Logging", "Agricultural expansion", "Urbanization"],
    "Bankura": ["Mining", "Agriculture", "Deforestation for fuel"],
    "Birbhum": ["Industrial development", "Agriculture"],
    "Cooch Behar": ["Urbanization", "Timber collection"],
    "Darjeeling": ["Tea plantations", "Logging"],
    "Hooghly": ["Industrial growth", "Urban spread"],
    "Jalpaiguri": ["Timber extraction", "Agriculture expansion"],
    "Murshidabad": ["Agriculture", "Urbanization"],
    "Nadia": ["Urbanization", "Agriculture"],
    "North 24 Parganas": ["Construction", "Deforestation for fuel"],
    "South 24 Parganas": ["Construction", "Deforestation for fuel"],
    "Paschim Medinipur": ["Logging", "Industrial expansion"],
    "Purulia": ["Mining", "Agriculture"]
}

DEFORESTATION_REDUCTION = [
    "Afforestation & Reforestation",
    "Sustainable agriculture",
    "Strict logging regulations",
    "Community awareness",
    "Protected forest areas"
]

# ---------------- Utility Functions ----------------

def filter_location_data(df, location):
    df = df.rename(columns=lambda x: x.strip())
    df['District'] = df['District'].astype(str)
    filtered = df[df['District'].str.lower() == location.lower()]
    if filtered.empty:
        raise ValueError(f"No data available for {location}")
    return filtered

def summarize_deforestation(df):
    df = df.copy()
    for col in ['Deforestation_Rate_%', 'Temperature_C', 'Rainfall_mm', 'Pollution_Index']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['Deforestation_Rate_%'])
    
    avg_rate = round(df['Deforestation_Rate_%'].mean(), 2)
    min_rate = round(df['Deforestation_Rate_%'].min(), 2)
    max_rate = round(df['Deforestation_Rate_%'].max(), 2)
    causes = DEFORESTATION_CAUSES.get(df['District'].iloc[0], [])
    reduction_methods = DEFORESTATION_REDUCTION

    text = (
        f"The deforestation rate in {df['District'].iloc[0]} has fluctuated over the years "
        f"with an average rate of {avg_rate}%. The main causes are {', '.join(causes)}. "
        f"Recommended reduction methods include {', '.join(reduction_methods)} "
        f"to maintain forest cover and ensure sustainable growth."
    )

    return {
        "average_rate": avg_rate,
        "min_rate": min_rate,
        "max_rate": max_rate,
        "causes": causes,
        "reduction_methods": reduction_methods,
        "text": text
    }

def summarize_environment(df):
    df = df.copy()
    for col in ["Temperature_C", "Rainfall_mm", "Pollution_Index"]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return {
        "avg_temp": round(df["Temperature_C"].mean(), 2),
        "avg_rainfall": round(df["Rainfall_mm"].mean(), 2),
        "avg_pollution": round(df["Pollution_Index"].mean(), 2)
    }

def plot_history(df, location, column, color, title):
    if column not in df.columns:
        return None
    df[column] = pd.to_numeric(df[column], errors='coerce')
    df = df.dropna(subset=[column])
    if df.empty:
        return None
    plt.figure(figsize=(6, 4))
    plt.plot(df['Year'], df[column], marker='o', color=color)
    plt.title(f"{title} - {location}")
    plt.xlabel("Year")
    plt.ylabel(column.replace("_", " "))
    path = os.path.join(PLOT_DIR, f"{location}_{column}.png")
    plt.savefig(path)
    plt.close()
    return path

def predict_future_10_years(df, years=10):
    # Ensure the rate column is numeric
    df['Deforestation_Rate_%'] = pd.to_numeric(df['Deforestation_Rate_%'], errors='coerce')
    df = df.dropna(subset=['Deforestation_Rate_%'])
    if df.empty or len(df) < 2:
        return {"years": [], "rates": []}

    # Sort by year in case CSV is unordered
    df = df.sort_values('Year')

    # Calculate the actual yearly changes from your data
    year_diffs = df['Year'].diff().dropna()
    rate_diffs = df['Deforestation_Rate_%'].diff().dropna()

    # Average yearly change based on real data trend
    avg_change_per_year = (rate_diffs / year_diffs).mean()

    # Get last known values
    last_year = int(df['Year'].iloc[-1])
    last_rate = float(df['Deforestation_Rate_%'].iloc[-1])

    # Predict next 'years' years using actual average change
    future_years = [last_year + i + 1 for i in range(years)]
    future_rates = [round(last_rate + avg_change_per_year * (i + 1), 2) for i in range(years)]

    return {"years": future_years, "rates": future_rates}


# ---------------- PDF Generator ----------------
def safe_text(text):
    """Remove any emoji/unicode unsupported by Latin-1."""
    if not isinstance(text, str):
        text = str(text)
    text = text.encode("latin-1", "replace").decode("latin-1")
    return text

def generate_pdf_report(location, df, summary, future_data):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, safe_text(f"Deforestation Report - {location}"), ln=True, align="C")
    pdf.ln(10)

    # ---------------- Historical Table ----------------
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 8, safe_text("Historical Data"), ln=True)
    pdf.set_font("Arial", "", 12)
    for col in ['Year','Deforestation_Rate_%','Rainfall_mm','Temperature_C','Pollution_Index']:
        if col not in df.columns:
            df[col] = "N/A"
    pdf.cell(25, 8, "Year", 1)
    pdf.cell(45, 8, "Deforestation Rate (%)", 1)
    pdf.cell(35, 8, "Rainfall (mm)", 1)
    pdf.cell(35, 8, "Temperature (°C)", 1)
    pdf.cell(35, 8, "Pollution Index", 1)
    pdf.ln()
    for _, row in df.iterrows():
        pdf.cell(25, 8, str(row['Year']), 1)
        pdf.cell(45, 8, str(row['Deforestation_Rate_%']), 1)
        pdf.cell(35, 8, str(row['Rainfall_mm']), 1)
        pdf.cell(35, 8, str(row['Temperature_C']), 1)
        pdf.cell(35, 8, str(row['Pollution_Index']), 1)
        pdf.ln()
    pdf.ln(5)

    # ---------------- Prediction Table ----------------
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 8, safe_text("10-Year Prediction"), ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.cell(25, 8, "Year", 1)
    pdf.cell(45, 8, "Predicted Rate (%)", 1)
    pdf.cell(35, 8, "Rainfall (mm)", 1)
    pdf.cell(35, 8, "Temperature (°C)", 1)
    pdf.cell(35, 8, "Pollution Index", 1)
    pdf.ln()
    last_row = df.iloc[-1]
    for i, year in enumerate(future_data["years"]):
        pdf.cell(25, 8, str(year), 1)
        pdf.cell(45, 8, str(future_data["rates"][i]), 1)
        pdf.cell(35, 8, str(last_row["Rainfall_mm"]), 1)
        pdf.cell(35, 8, str(last_row["Temperature_C"]), 1)
        pdf.cell(35, 8, str(last_row["Pollution_Index"]), 1)
        pdf.ln()
    pdf.ln(5)

    # ---------------- Charts ----------------
    for col, color, title in [
        ("Deforestation_Rate_%", "red", "Deforestation History"),
        ("Temperature_C", "orange", "Temperature Over Years"),
        ("Rainfall_mm", "blue", "Rainfall Over Years"),
        ("Pollution_Index", "purple", "Pollution Over Years")
    ]:
        path = plot_history(df, location, col, color, title)
        if path:
            pdf.image(path, x=15, w=180)
            pdf.ln(5)

    # ---------------- Summary ----------------
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 8, safe_text("Deforestation Summary:"), ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 8, safe_text(summary['text']))
    pdf.cell(0, 8, safe_text(f"Average Rate: {summary['average_rate']}%"), ln=True)
    pdf.cell(0, 8, safe_text(f"Min Rate: {summary['min_rate']}% | Max Rate: {summary['max_rate']}%"), ln=True)
    pdf.ln(5)

    pdf.cell(0, 8, safe_text("Causes of Deforestation:"), ln=True)
    for cause in summary['causes']:
        pdf.cell(0, 8, safe_text(f"- {cause}"), ln=True)

    pdf.cell(0, 8, safe_text("Reduction Methods:"), ln=True)
    for method in summary['reduction_methods']:
        pdf.cell(0, 8, safe_text(f"- {method}"), ln=True)

    pdf_path = os.path.join(BASE_DIR, f"report_{location}.pdf")
    pdf.output(pdf_path)
    return pdf_path
