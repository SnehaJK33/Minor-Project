
from flask import Flask, jsonify, send_file, send_from_directory
from flask_cors import CORS
import pandas as pd
import os
from io import BytesIO
from werkzeug.exceptions import NotFound

from utils import (
    _district_history_frame,
    _district_prediction_trend,
    _normalize_district_name,
    _safe_float,
    _to_num,
    build_combined_prediction,
    build_district_risk_clusters,
    clean_json,
)

BUILD_VERSION = "2026-03-21-05"

app = Flask(__name__)
# Allow frontend dev server (Live Server) to call the API.
CORS(
    app,
    resources={
        r"/api/*": {
            "origins": [
                "http://127.0.0.1:5500",
                "http://localhost:5500",
                "http://127.0.0.1:8000",
                "http://localhost:8000",
            ]
        }
    },
)

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend")
DATA_DIR = os.path.join(BASE_DIR, "data")
FOREST_CSV = os.path.join(DATA_DIR, "MAIN DATA - Sheet1.csv")
DRIVERS_CSV = os.path.join(DATA_DIR, "MAIN DATA - Sheet2.csv")
GAIN_LOSS_CSV = os.path.join(DATA_DIR, "MAIN DATA - overall gain and lose data.csv")
GAS_CSV = os.path.join(DATA_DIR, "MAIN DATA - gas.csv")
DISTRICT_CSV = os.path.join(DATA_DIR, "MAIN DATA - district_data.csv")
DISTRICT_GAIN_CSV = os.path.join(DATA_DIR, "ALL DATASETS - overall gain forest .csv")

def load_wb_data():
    try:
        forest = pd.read_csv(FOREST_CSV)
        drivers = pd.read_csv(DRIVERS_CSV)
        gain_loss = pd.read_csv(GAIN_LOSS_CSV)
        gas = pd.read_csv(GAS_CSV)
        district = pd.read_csv(DISTRICT_CSV)
        district_gain = pd.read_csv(DISTRICT_GAIN_CSV)

        forest.columns = forest.columns.str.strip()
        drivers.columns = drivers.columns.str.strip()
        gain_loss.columns = gain_loss.columns.str.strip()
        gas.columns = gas.columns.str.strip()
        district.columns = district.columns.str.strip()
        district_gain.columns = district_gain.columns.str.strip()

        forest = _to_num(
            forest,
            [
                "Year",
                "Tree_Cover_Lose_ha",
                "Green-House_Gases",
                "Tree_cover_loss_from_fires",
                "Tree_cover_loss_rainforest",
            ],
        )
        drivers = _to_num(drivers, ["Year", "Loss_area_ha"])
        gain_loss = _to_num(
            gain_loss,
            ["stable", "loss", "gain", "disturb", "net", "change", "gfw_area__ha", "adm1"],
        )
        gas = _to_num(
            gas,
            [
                "adm1",
                "umd_tree_cover_loss__year",
                "gfw_gross_emissions_co2e_all_gases__Mg",
                "gfw_gross_emissions_co2e_non_co2__Mg",
                "gfw_gross_emissions_co2e_co2_only__Mg",
            ],
        )
        district = _to_num(
            district,
            [
                "adm1",
                "umd_tree_cover_loss__year",
                "umd_tree_cover_loss__ha",
                "gfw_gross_emissions_co2e_all_gases__Mg",
            ],
        )
        # Winsorize extreme loss outliers to stabilize district models.
        loss_col = "umd_tree_cover_loss__ha"
        loss_vals = pd.to_numeric(district[loss_col], errors="coerce")
        if not loss_vals.dropna().empty:
            cap = float(loss_vals.dropna().quantile(0.95))
            district[loss_col] = loss_vals.clip(upper=cap)
        district_gain = _to_num(
            district_gain,
            [
                "adm1",
                "umd_tree_cover_gain__ha",
            ],
        )

        forest = forest.sort_values("Year")
        drivers = drivers.sort_values(["Year", "Drivers_type"])
        gas = gas.sort_values("umd_tree_cover_loss__year")
        district = district.sort_values(["adm2", "umd_tree_cover_loss__year"])
        district_gain = district_gain.sort_values(["adm2"])

        print("West Bengal CSV datasets loaded.")
        return forest, drivers, gain_loss, gas, district, district_gain
    except Exception as e:
        raise RuntimeError(f"Error loading West Bengal CSV files: {e}")


df_forest, df_drivers, df_gain_loss, df_gas, df_district, df_district_gain = load_wb_data()


@app.route("/", methods=["GET"], strict_slashes=False)
def home():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/westbengal", methods=["GET"], strict_slashes=False)
def westbengal_page():
    return send_from_directory(FRONTEND_DIR, "westbengal.html")


@app.route("/WestBengal", methods=["GET"], strict_slashes=False)
@app.route("/westBengal", methods=["GET"], strict_slashes=False)
def westbengal_page_alias():
    return send_from_directory(FRONTEND_DIR, "westbengal.html")


@app.route("/westbengal.html", methods=["GET"])
@app.route("/WestBengal.html", methods=["GET"])
@app.route("/westBengal.html", methods=["GET"])
def westbengal_page_html():
    return send_from_directory(FRONTEND_DIR, "westbengal.html")


@app.route("/district", methods=["GET"], strict_slashes=False)
def district_page():
    return send_from_directory(FRONTEND_DIR, "District.html")


@app.route("/District", methods=["GET"], strict_slashes=False)
def district_page_alias():
    return send_from_directory(FRONTEND_DIR, "District.html")


@app.route("/district.html", methods=["GET"])
@app.route("/District.html", methods=["GET"])
def district_page_html():
    return send_from_directory(FRONTEND_DIR, "District.html")


@app.route("/index.html", methods=["GET"])
def index_page_html():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/frontend", methods=["GET"], strict_slashes=False)
def frontend_root():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/frontend/<path:filename>", methods=["GET"])
def frontend_prefixed_assets(filename):
    if filename.startswith("api/"):
        return jsonify({"status": "error", "message": "Not found"}), 404
    if filename.lower() == "district.html":
        return send_from_directory(FRONTEND_DIR, "District.html")
    return send_from_directory(FRONTEND_DIR, filename)


@app.route("/<path:filename>", methods=["GET"])
def frontend_assets(filename):
    # Serve frontend files (script.js/style.css/html). API routes continue using /api/*.
    if filename.startswith("api/"):
        return jsonify({"status": "error", "message": "Not found"}), 404
    try:
        return send_from_directory(FRONTEND_DIR, filename)
    except NotFound:
        return jsonify({"status": "error", "message": "Not found"}), 404


@app.route("/api/wb/forest_overall", methods=["GET"])
def wb_forest_overall():
    records = df_forest.to_dict(orient="records")
    return jsonify(
        clean_json(
            {
                "status": "success",
                "state": "West Bengal",
                "total_records": len(records),
                "history": records,
            }
        )
    )


@app.route("/api/wb/drivers", methods=["GET"])
def wb_drivers():
    years = sorted(df_drivers["Year"].dropna().astype(int).unique().tolist())
    drivers = sorted(df_drivers["Drivers_type"].dropna().astype(str).unique().tolist())

    series = {d: [0.0 for _ in years] for d in drivers}
    grouped = (
        df_drivers.groupby(["Drivers_type", "Year"], as_index=False)["Loss_area_ha"]
        .sum()
    )
    y_index = {y: i for i, y in enumerate(years)}
    for row in grouped.to_dict(orient="records"):
        d = str(row.get("Drivers_type", ""))
        y = int(row.get("Year", 0))
        v = float(row.get("Loss_area_ha", 0) or 0)
        if d in series and y in y_index:
            series[d][y_index[y]] = v

    return jsonify(
        clean_json(
            {
                "status": "success",
                "state": "West Bengal",
                "total_records": len(df_drivers),
                "years": years,
                "drivers": drivers,
                "series": series,
                "records": df_drivers.to_dict(orient="records"),
            }
        )
    )


@app.route("/api/wb/gain_loss", methods=["GET"])
def wb_gain_loss():
    if df_gain_loss.empty:
        return jsonify({"status": "error", "message": "No gain/loss data found"}), 404

    row = df_gain_loss.iloc[0].to_dict()
    return jsonify(
        clean_json(
            {
                "status": "success",
                "state": row.get("adm1_name", "West Bengal"),
                "summary": row,
            }
        )
    )


@app.route("/api/wb/prediction", methods=["GET"])
def wb_prediction():
    try:
        pred = build_combined_prediction(df_forest, df_drivers, n_years=5)
        return jsonify(
            clean_json(
                {
                    "status": "success",
                    "state": "West Bengal",
                "future_prediction_5_years": pred,
                "build_version": BUILD_VERSION,
            }
        )
    )
    except Exception as exc:
        import traceback

        return (
            jsonify(
                {
                    "status": "error",
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                    "build_version": BUILD_VERSION,
                }
            ),
            500,
        )


@app.route("/api/wb/gas", methods=["GET"])
def wb_gas():
    if df_gas.empty:
        return jsonify({"status": "error", "message": "No gas data found"}), 404

    grouped = (
        df_gas.groupby("umd_tree_cover_loss__year", as_index=False)[
            [
                "gfw_gross_emissions_co2e_all_gases__Mg",
                "gfw_gross_emissions_co2e_non_co2__Mg",
                "gfw_gross_emissions_co2e_co2_only__Mg",
            ]
        ]
        .sum()
        .sort_values("umd_tree_cover_loss__year")
    )

    return jsonify(
        clean_json(
            {
                "status": "success",
                "state": "West Bengal",
                "years": grouped["umd_tree_cover_loss__year"].astype(int).tolist(),
                "all_gases_mg": grouped["gfw_gross_emissions_co2e_all_gases__Mg"].tolist(),
                "non_co2_mg": grouped["gfw_gross_emissions_co2e_non_co2__Mg"].tolist(),
                "co2_only_mg": grouped["gfw_gross_emissions_co2e_co2_only__Mg"].tolist(),
                "records": grouped.to_dict(orient="records"),
            }
        )
    )


@app.route("/api/wb/district_risk_clusters", methods=["GET"])
def wb_district_risk_clusters():
    rows = build_district_risk_clusters(df_district)
    return jsonify(
        clean_json(
            {
                "status": "success",
                "state": "West Bengal",
                "clusters": rows,
            }
        )
    )


@app.route("/api/wb/district_deforestation", methods=["GET"])
def wb_district_deforestation():
    if df_district.empty:
        return jsonify({"status": "error", "message": "No district data found"}), 404

    d = df_district.copy()
    d["umd_tree_cover_loss__ha"] = pd.to_numeric(d["umd_tree_cover_loss__ha"], errors="coerce")
    d = d.dropna(subset=["adm2", "umd_tree_cover_loss__ha"])

    grouped = (
        d.groupby("adm2", as_index=False)["umd_tree_cover_loss__ha"]
        .sum()
        .sort_values("umd_tree_cover_loss__ha", ascending=False)
    )

    return jsonify(
        clean_json(
            {
                "status": "success",
                "state": "West Bengal",
                "districts": grouped["adm2"].astype(str).tolist(),
                "loss_ha": grouped["umd_tree_cover_loss__ha"].astype(float).tolist(),
                "records": grouped.to_dict(orient="records"),
            }
        )
    )


@app.route("/api/data/<location>", methods=["GET"])
def district_data(location):
    district_name, hist = _district_history_frame(df_district, location, df_forest)
    if hist.empty:
        return jsonify({"status": "error", "message": f"District '{location}' not found."}), 404

    cols = [
        "Year",
        "Deforestation_Rate_%",
        "Tree_Cover_Loss_ha",
        "Tree_cover_loss_from_fires",
        "Green-House_Gases",
        "Pollution Index (AQI)",
        "Rainfall_mm",
        "Temperature_C",
        "Urbanization Rate (%)",
        "Population (Est.)",
    ]
    out = hist[cols].to_dict(orient="records")
    return jsonify(
        clean_json(
            {
                "status": "success",
                "location": district_name,
                "total_records": len(out),
                "history": out,
            }
        )
    )


@app.route("/api/summary/<location>", methods=["GET"])
def district_summary(location):
    district_name, hist = _district_history_frame(df_district, location, df_forest)
    if hist.empty:
        return jsonify({"status": "error", "message": f"District '{location}' not found."}), 404

    rate = pd.to_numeric(hist["Deforestation_Rate_%"], errors="coerce")
    pred = _district_prediction_trend(hist, n_years=5)
    cluster_rows = build_district_risk_clusters(df_district)
    risk = "Unknown"
    district_key = district_name.lower().strip()
    for row in cluster_rows:
        if str(row.get("district", "")).lower().strip() == district_key:
            risk = str(row.get("risk_zone", "Unknown"))
            break
    if risk == "Unknown":
        latest = float(rate.dropna().iloc[-1]) if not rate.dropna().empty else 0.0
        risk = "Low"
        if latest >= 60:
            risk = "High"
        elif latest >= 20:
            risk = "Medium"

    summary = {
        "average_deforestation_rate": _safe_float(rate.mean()),
        "max_deforestation_rate": _safe_float(rate.max()),
        "min_deforestation_rate": _safe_float(rate.min()),
        "total_records": int(len(hist)),
    }
    env = {
        "average_rainfall_mm": None,
        "average_temperature_c": None,
        "avg_pollution": _safe_float(pd.to_numeric(hist["Pollution Index (AQI)"], errors="coerce").mean()),
    }
    return jsonify(
        clean_json(
            {
                "status": "success",
                "build_version": BUILD_VERSION,
                "location": district_name,
                "summary": summary,
                "environment": env,
                "future_prediction_5_years": pred,
                "prediction_source": "district_elasticnet_lags",
                "risk_assessment": {"risk_level": risk},
            }
        )
    )


@app.route("/api/report/<location>", methods=["GET"])
def district_report(location):
    district_name, hist = _district_history_frame(df_district, location, df_forest)
    if hist.empty:
        return jsonify({"status": "error", "message": f"District '{location}' not found."}), 404

    report_cols = [
        "Year",
        "Tree_Cover_Loss_ha",
        "Green-House_Gases",
        "Pollution Index (AQI)",
    ]
    csv_data = hist[report_cols].to_csv(index=False)
    buffer = BytesIO(csv_data.encode("utf-8"))
    buffer.seek(0)
    return send_file(
        buffer,
        mimetype="text/csv",
        as_attachment=True,
        download_name=f"{district_name}_district_report.csv",
    )


@app.route("/api/district/gain/<location>", methods=["GET"])
def district_gain(location):
    district_name = _normalize_district_name(location)
    d = df_district_gain[df_district_gain["adm2"].astype(str).str.lower() == district_name.lower()].copy()
    if d.empty:
        return jsonify({"status": "error", "message": f"District '{location}' gain data not found."}), 404

    gain_val = _safe_float(pd.to_numeric(d["umd_tree_cover_gain__ha"], errors="coerce").sum())
    return jsonify(
        clean_json(
            {
                "status": "success",
                "location": district_name,
                "gain_ha": gain_val if gain_val is not None else 0.0,
            }
        )
    )


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=8000,
        debug=True
    )

