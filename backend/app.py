from flask import Flask, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import os
from utils import filter_location_data, summarize_deforestation, predict_future_10_years, generate_pdf_report

app = Flask(__name__)
CORS(app)

DATA_PATH = os.path.join(os.path.dirname(__file__), 'data.csv')
df = pd.read_csv(DATA_PATH)

@app.route('/api/data/<location>', methods=['GET'])
def get_data(location):
    location_data = filter_location_data(df, location)
    if location_data.empty:
        return jsonify({"error": "Location not found"}), 404
    history = location_data.to_dict(orient='records')
    return jsonify({"location": location, "history": history})

@app.route('/api/summary/<location>', methods=['GET'])
def get_summary(location):
    location_data = filter_location_data(df, location)
    if location_data.empty:
        return jsonify({"error": "Location not found"}), 404

    summary = summarize_deforestation(location_data)
    future = predict_future_10_years(location_data)

    # Ensure JSON serializable
    future_json = {
        "years": [int(y) for y in future['years']],
        "rates": [float(r) for r in future['rates']]
    }

    return jsonify({
        "location": location,
        "summary": summary,
        "future": future_json
    })

@app.route('/api/report/<location>', methods=['GET'])
def download_report(location):
    location_data = filter_location_data(df, location)
    if location_data.empty:
        return jsonify({"error": "Location not found"}), 404

    summary = summarize_deforestation(location_data)
    future = predict_future_10_years(location_data)
    pdf_path = generate_pdf_report(location, location_data, summary, future)

    if os.path.exists(pdf_path):
        return send_file(pdf_path, as_attachment=True)
    else:
        return jsonify({"error": "Report not generated"}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)
