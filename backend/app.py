from flask import Flask, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import os

# Import helper functions from utils.py
from utils import (
    filter_location_data,
    summarize_deforestation,
    summarize_environment,
    predict_future_10_years,
    generate_pdf_report
)

# ---------------- App Setup ----------------
app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data.csv")

# Load dataset once at startup
try:
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()  # remove accidental spaces in headers
except Exception as e:
    raise RuntimeError(f"Error loading dataset: {e}")

# ---------------- Routes ----------------

@app.route("/api/data/<location>")
def get_data(location):
    """Return historical data for a specific district."""
    try:
        data = filter_location_data(df, location)
        history = data.to_dict(orient="records")
        return jsonify({"history": history})
    except ValueError as e:
        return jsonify({"error": str(e)}), 404


@app.route("/api/summary/<location>")
def get_summary(location):
    """Return summary, environment, and future prediction for a district."""
    try:
        data = filter_location_data(df, location)
        defo_summary = summarize_deforestation(data)
        future = predict_future_10_years(data)
        env_summary = summarize_environment(data)

        return jsonify({
            "summary": defo_summary,
            "future": future,
            "environment": env_summary
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 404


@app.route("/api/report/<location>")
def download_report(location):
    """Generate and download a full PDF report for a district."""
    try:
        data = filter_location_data(df, location)
        summary = summarize_deforestation(data)
        future = predict_future_10_years(data)
        path = generate_pdf_report(location, data, summary, future)
        return send_file(path, as_attachment=True)
    except ValueError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        return jsonify({"error": f"Report generation failed: {e}"}), 500


# ---------------- Main Entry ----------------
if __name__ == "__main__":
    app.run(port=8000, debug=True)
