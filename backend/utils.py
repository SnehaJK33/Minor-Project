import os
import matplotlib.pyplot as plt
from fpdf import FPDF
import pandas as pd

# Ensure plot directory exists
PLOT_DIR = os.path.join(os.path.dirname(__file__), 'plots')
os.makedirs(PLOT_DIR, exist_ok=True)

# Causes of deforestation (example data)
# Causes of deforestation (updated)
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


# Methods to reduce deforestation
DEFORESTATION_REDUCTION = [
    "Afforestation & Reforestation",
    "Sustainable agriculture",
    "Strict logging regulations",
    "Community awareness",
    "Protected forest areas"
]

# ------------------- Functions -------------------

def filter_location_data(df, location):
    """Filter dataset for a specific location (case-insensitive)."""
    df = df.rename(columns=lambda x: x.strip())
    df['District'] = df['District'].astype(str)
    return df[df['District'].str.lower() == location.lower()]

def summarize_deforestation(df):
    avg_rate = round(df['Deforestation_Rate_%'].mean(), 2)
    min_rate = round(df['Deforestation_Rate_%'].min(), 2)
    max_rate = round(df['Deforestation_Rate_%'].max(), 2)
    causes = DEFORESTATION_CAUSES.get(df['District'].iloc[0], [])
    reduction_methods = DEFORESTATION_REDUCTION

    # 50-word summary text
    text = f"The deforestation rate in {df['District'].iloc[0]} has fluctuated over the years with an average rate of {avg_rate}%. The major causes include {', '.join(causes)}. Recommended reduction methods are {', '.join(reduction_methods)} to ensure forest conservation and sustainable growth."

    return {
        "average_rate": avg_rate,
        "min_rate": min_rate,
        "max_rate": max_rate,
        "causes": causes,
        "reduction_methods": reduction_methods,
        "text": text
    }


def predict_future_10_years(df, years=10):
    """Predict future deforestation rates (simple 2% growth per year)."""
    last_val = float(df['Deforestation_Rate_%'].iloc[-1])
    future_rates = [round(last_val * (1 + 0.02*(i+1)), 2) for i in range(years)]
    future_years = [int(df['Year'].iloc[-1]) + i + 1 for i in range(years)]
    return {"years": future_years, "rates": future_rates}

def plot_history(df, location):
    """Plot historical deforestation graph."""
    plt.figure(figsize=(6,4))
    plt.plot(df['Year'], df['Deforestation_Rate_%'], marker='o', color='red')
    plt.title(f'Deforestation History - {location}')
    plt.xlabel('Year')
    plt.ylabel('Deforestation Rate (%)')
    path = os.path.join(PLOT_DIR, f'{location}_history.png')
    plt.savefig(path)
    plt.close()
    return path

def plot_prediction(future_data, location):
    """Plot predicted deforestation graph."""
    plt.figure(figsize=(6,4))
    plt.plot(future_data['years'], future_data['rates'], marker='o', color='green')
    plt.title(f'Deforestation Prediction - {location}')
    plt.xlabel('Year')
    plt.ylabel('Predicted Deforestation Rate (%)')
    path = os.path.join(PLOT_DIR, f'{location}_prediction.png')
    plt.savefig(path)
    plt.close()
    return path

def generate_pdf_report(location, df, summary, future_data):
    """Generate a PDF report including history, prediction, and summary."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, f"Deforestation Report - {location}", ln=True, align='C')
    pdf.ln(10)

    # Historical plot
    history_path = plot_history(df, location)
    pdf.image(history_path, x=15, w=180)
    pdf.ln(10)

    # Historical data table
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 8, "Historical Deforestation Data:", ln=True)
    for idx, row in df.iterrows():
        pdf.cell(0, 8, f"{row['Year']}: {row['Deforestation_Rate_%']}%", ln=True)
    pdf.ln(5)

    # Prediction plot
    prediction_path = plot_prediction(future_data, location)
    pdf.image(prediction_path, x=15, w=180)
    pdf.ln(10)

    # Summary
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 8, "Summary:", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 8, f"Average Rate: {summary['average_rate']}%", ln=True)
    pdf.cell(0, 8, f"Minimum Rate: {summary['min_rate']}%", ln=True)
    pdf.cell(0, 8, f"Maximum Rate: {summary['max_rate']}%", ln=True)
    pdf.ln(5)

    pdf.cell(0, 8, "Causes of Deforestation:", ln=True)
    for cause in summary['causes']:
        pdf.cell(0, 8, f"- {cause}", ln=True)

    pdf.ln(5)
    pdf.cell(0, 8, "Methods to Reduce Deforestation:", ln=True)
    for method in summary['reduction_methods']:
        pdf.cell(0, 8, f"- {method}", ln=True)

    pdf_path = os.path.join(os.path.dirname(__file__), f'report_{location}.pdf')
    pdf.output(pdf_path)
    return pdf_path
