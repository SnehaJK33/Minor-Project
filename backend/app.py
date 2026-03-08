from flask import Flask, jsonify, send_file, send_from_directory
from flask_cors import CORS
import pandas as pd
import os
import math
import numpy as np
from io import BytesIO
from werkzeug.exceptions import NotFound
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend")
DATA_DIR = os.path.join(BASE_DIR, "data")
FOREST_CSV = os.path.join(DATA_DIR, "MAIN DATA - Sheet1.csv")
DRIVERS_CSV = os.path.join(DATA_DIR, "MAIN DATA - Sheet2.csv")
GAIN_LOSS_CSV = os.path.join(DATA_DIR, "MAIN DATA - overall gain and lose data.csv")
GAS_CSV = os.path.join(DATA_DIR, "MAIN DATA - gas.csv")
DISTRICT_CSV = os.path.join(DATA_DIR, "ALL DATASETS - district_data.csv")
DISTRICT_GAIN_CSV = os.path.join(DATA_DIR, "ALL DATASETS - overall gain forest .csv")

DISTRICT_ALIASES = {
    "paschim bardhaman": "Bardhaman",
    "purba bardhaman": "Bardhaman",
    "jhargram": "Paschim Medinipur",
    "kalimpong": "Darjeeling",
}


def clean_json(obj):
    if isinstance(obj, dict):
        return {k: clean_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_json(v) for v in obj]
    if hasattr(obj, "item"):
        obj = obj.item()
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def _to_num(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _linear_future(values, n_years):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return np.zeros(n_years, dtype=float)
    if arr.size == 1:
        return np.repeat(arr[-1], n_years)

    x = np.arange(arr.size, dtype=float)
    slope, intercept = np.polyfit(x, arr, 1)
    fx = np.arange(arr.size, arr.size + n_years, dtype=float)
    return intercept + slope * fx


def _safe_float(v):
    try:
        if v is None:
            return None
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return None
        return float(v)
    except Exception:
        return None


def _normalize_district_name(name):
    raw = (name or "").strip()
    if not raw:
        return raw
    return DISTRICT_ALIASES.get(raw.lower(), raw)


def _district_history_frame(df_district, location):
    district_name = _normalize_district_name(location)
    d = df_district[df_district["adm2"].astype(str).str.lower() == district_name.lower()].copy()
    if d.empty:
        return district_name, d

    d = d.sort_values("umd_tree_cover_loss__year").copy()
    d["Year"] = pd.to_numeric(d["umd_tree_cover_loss__year"], errors="coerce")
    d["Tree_Cover_Loss_ha"] = pd.to_numeric(d["umd_tree_cover_loss__ha"], errors="coerce")
    d["Deforestation_Rate_%"] = d["Tree_Cover_Loss_ha"]  # Keep frontend compatibility.
    d["Green-House_Gases"] = pd.to_numeric(d["gfw_gross_emissions_co2e_all_gases__Mg"], errors="coerce")
    d["Pollution Index (AQI)"] = d["Green-House_Gases"] / 1000.0

    # Estimate district fire-related loss from state-level yearly fire/loss ratio (Sheet1).
    fire_ratio_by_year = {}
    if "Year" in df_forest.columns and "Tree_cover_loss_from_fires" in df_forest.columns and "Tree_Cover_Lose_ha" in df_forest.columns:
        wb_tmp = df_forest[["Year", "Tree_cover_loss_from_fires", "Tree_Cover_Lose_ha"]].dropna().copy()
        for _, r in wb_tmp.iterrows():
            y = int(r["Year"])
            total = float(r["Tree_Cover_Lose_ha"]) if r["Tree_Cover_Lose_ha"] is not None else 0.0
            fire = float(r["Tree_cover_loss_from_fires"]) if r["Tree_cover_loss_from_fires"] is not None else 0.0
            fire_ratio_by_year[y] = (fire / total) if total > 0 else 0.0
    default_fire_ratio = float(np.mean(list(fire_ratio_by_year.values()))) if fire_ratio_by_year else 0.0
    d["Tree_cover_loss_from_fires"] = d.apply(
        lambda r: (r["Tree_Cover_Loss_ha"] or 0.0) * fire_ratio_by_year.get(int(r["Year"]), default_fire_ratio),
        axis=1,
    )

    # Not present in district CSV, keep as missing for charts/tables.
    d["Rainfall_mm"] = np.nan
    d["Temperature_C"] = np.nan
    d["Urbanization Rate (%)"] = np.nan
    d["Population (Est.)"] = np.nan
    return district_name, d


def _district_prediction(history_df, n_years=5):
    rows = history_df.dropna(subset=["Year", "Tree_Cover_Loss_ha"]).copy()
    if len(rows) < 3:
        vals = rows["Tree_Cover_Loss_ha"].tolist()
        years = rows["Year"].astype(int).tolist()
        fut = _linear_future(vals, n_years) if vals else []
        future_years = [int(years[-1] + i) for i in range(1, n_years + 1)] if years else []
        return {
            "years": future_years,
            "rates": [round(float(max(v, 0.0)), 2) for v in fut],
            "model_name": "Fallback linear trend (district CSV)",
            "training_samples": int(len(rows)),
            "used_fallback": True,
        }

    rows = rows.sort_values("Year").copy()
    X = rows[["Year", "Green-House_Gases"]].fillna(0.0).astype(float)
    y = rows["Tree_Cover_Loss_ha"].astype(float).values

    model = RandomForestRegressor(n_estimators=500, random_state=42)
    n_splits = max(2, min(5, len(X) - 1))
    tscv = TimeSeriesSplit(n_splits=n_splits)

    fold_mae, fold_rmse, fold_r2 = [], [], []
    for tr, te in tscv.split(X):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr, yte = y[tr], y[te]
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        fold_mae.append(mean_absolute_error(yte, pred))
        fold_rmse.append(math.sqrt(mean_squared_error(yte, pred)))
        if len(yte) >= 2:
            fold_r2.append(r2_score(yte, pred))

    model.fit(X, y)
    train_pred = model.predict(X)
    train_r2 = r2_score(y, train_pred) if len(y) >= 2 else None

    last_year = int(rows["Year"].max())
    future_years = [last_year + i for i in range(1, n_years + 1)]
    future_ghg = _linear_future(rows["Green-House_Gases"].fillna(0.0).values, n_years)
    Xf = pd.DataFrame({"Year": future_years, "Green-House_Gases": future_ghg}).astype(float)
    fut_pred = model.predict(Xf)

    return {
        "years": future_years,
        "rates": [round(float(max(v, 0.0)), 2) for v in fut_pred.tolist()],
        "model_name": "RandomForestRegressor (district CSV)",
        "training_samples": int(len(rows)),
        "model_accuracy_r2": _safe_float(round(train_r2, 4)) if train_r2 is not None else None,
        "cv_mae": _safe_float(round(float(np.mean(fold_mae)), 4)) if fold_mae else None,
        "cv_rmse": _safe_float(round(float(np.mean(fold_rmse)), 4)) if fold_rmse else None,
        "cv_r2": _safe_float(round(float(np.mean(fold_r2)), 4)) if fold_r2 else None,
        "used_fallback": False,
    }


def build_combined_prediction(forest_df, drivers_df, n_years=5):
    required_sheet1 = [
        "Year",
        "Tree_Cover_Lose_ha",
        "Green-House_Gases",
        "Tree_cover_loss_from_fires",
        "Tree_cover_loss_rainforest",
    ]
    forest = forest_df[required_sheet1].dropna(subset=["Year", "Tree_Cover_Lose_ha"]).copy()
    if forest.empty:
        return {
            "years": [],
            "rates": [],
            "model_name": "Ensemble model (Sheet1 + Sheet2)",
            "training_samples": 0,
            "used_fallback": True,
        }

    # Sheet2 contributes yearly Loss_area_ha (summed across all driver types).
    drivers = drivers_df[["Year", "Loss_area_ha"]].dropna(subset=["Year"]).copy()
    driver_total = (
        drivers.groupby("Year", as_index=False)["Loss_area_ha"]
        .sum()
        .rename(columns={"Loss_area_ha": "Loss_area_ha"})
    )

    merged = forest.merge(driver_total, on="Year", how="left")
    merged = merged.fillna(0.0).sort_values("Year").reset_index(drop=True)
    merged["Year"] = merged["Year"].astype(int)

    if len(merged) < 3:
        years = merged["Year"].astype(int).tolist()
        vals = merged["Tree_Cover_Lose_ha"].astype(float).tolist()
        fut_vals = _linear_future(vals, n_years)
        future_years = [int(years[-1] + i) for i in range(1, n_years + 1)] if years else []
        return {
            "years": future_years,
            "rates": [round(float(max(v, 0.0)), 2) for v in fut_vals],
            "model_name": "Fallback linear trend (limited samples)",
            "training_samples": int(len(merged)),
            "used_fallback": True,
        }

    feature_cols = [
        "Year",
        "Green-House_Gases",
        "Tree_cover_loss_from_fires",
        "Tree_cover_loss_rainforest",
        "Loss_area_ha",
    ]
    X = merged[feature_cols].astype(float).copy()
    y = merged["Tree_Cover_Lose_ha"].astype(float).values

    # Use Random Forest as requested.
    model_name = "RandomForestRegressor (features: Sheet1 + Sheet2)"
    model = RandomForestRegressor(n_estimators=500, random_state=42)
    n_splits = max(2, min(5, len(X) - 1))
    tscv = TimeSeriesSplit(n_splits=n_splits)

    fold_mae = []
    fold_r2 = []
    fold_rmse = []
    for train_idx, test_idx in tscv.split(X):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        fold_mae.append(mean_absolute_error(y_test, pred))
        fold_rmse.append(math.sqrt(mean_squared_error(y_test, pred)))
        if len(y_test) >= 2:
            fold_r2.append(r2_score(y_test, pred))

    cv_mae = float(np.mean(fold_mae)) if fold_mae else None
    cv_rmse = float(np.mean(fold_rmse)) if fold_rmse else None
    cv_r2 = float(np.mean(fold_r2)) if fold_r2 else None

    # Fit model on all historical data.
    model.fit(X, y)
    train_pred = model.predict(X)
    train_r2 = r2_score(y, train_pred) if len(y) >= 2 else None

    future_years = [int(merged["Year"].iloc[-1] + i) for i in range(1, n_years + 1)]

    # Forecast future exogenous features with trend, then predict target.
    future_features = pd.DataFrame({"Year": future_years})
    future_features["Green-House_Gases"] = _linear_future(merged["Green-House_Gases"].values, n_years)
    future_features["Tree_cover_loss_from_fires"] = _linear_future(merged["Tree_cover_loss_from_fires"].values, n_years)
    future_features["Tree_cover_loss_rainforest"] = _linear_future(merged["Tree_cover_loss_rainforest"].values, n_years)
    future_features["Loss_area_ha"] = _linear_future(merged["Loss_area_ha"].values, n_years)
    future_X = future_features[feature_cols].astype(float).copy()
    fut_pred = model.predict(future_X)

    return {
        "years": future_years,
        "rates": [round(float(max(v, 0.0)), 2) for v in fut_pred.tolist()],
        "model_name": model_name,
        "training_samples": int(len(merged)),
        "model_accuracy_r2": None if train_r2 is None else round(float(train_r2), 4),
        "cv_mae": None if cv_mae is None else round(float(cv_mae), 4),
        "cv_rmse": None if cv_rmse is None else round(float(cv_rmse), 4),
        "cv_r2": None if cv_r2 is None else round(float(cv_r2), 4),
        "feature_columns": feature_cols,
        "used_fallback": False,
    }


def build_district_risk_clusters(df_district):
    if df_district.empty:
        return []

    d = df_district.copy()
    d["umd_tree_cover_loss__ha"] = pd.to_numeric(d["umd_tree_cover_loss__ha"], errors="coerce")
    d["gfw_gross_emissions_co2e_all_gases__Mg"] = pd.to_numeric(
        d["gfw_gross_emissions_co2e_all_gases__Mg"], errors="coerce"
    )
    d = d.dropna(subset=["adm2", "umd_tree_cover_loss__ha"])
    if d.empty:
        return []

    grouped = d.groupby("adm2")
    rows = []
    for district, g in grouped:
        g = g.sort_values("umd_tree_cover_loss__year")
        losses = g["umd_tree_cover_loss__ha"].dropna().values
        gases = g["gfw_gross_emissions_co2e_all_gases__Mg"].dropna().values
        if len(losses) == 0:
            continue
        avg_loss = float(np.mean(losses))
        recent_loss = float(np.mean(losses[-3:])) if len(losses) >= 3 else float(np.mean(losses))
        avg_gas = float(np.mean(gases)) if len(gases) else 0.0
        loss_trend = 0.0
        if len(losses) >= 2:
            x = np.arange(len(losses), dtype=float)
            slope, _ = np.polyfit(x, losses.astype(float), 1)
            loss_trend = float(slope)
        rows.append(
            {
                "district": district,
                "avg_loss_ha": avg_loss,
                "recent_loss_ha": recent_loss,
                "avg_gas_mg": avg_gas,
                "loss_trend": loss_trend,
            }
        )

    if len(rows) < 3:
        # Not enough districts for 3 clusters; mark by quantiles.
        sorted_rows = sorted(rows, key=lambda r: r["avg_loss_ha"])
        for i, r in enumerate(sorted_rows):
            if i < len(sorted_rows) / 3:
                r["risk_zone"] = "Low"
            elif i < 2 * len(sorted_rows) / 3:
                r["risk_zone"] = "Medium"
            else:
                r["risk_zone"] = "High"
        return sorted_rows

    feat = np.array(
        [[r["avg_loss_ha"], r["recent_loss_ha"], r["avg_gas_mg"], r["loss_trend"]] for r in rows],
        dtype=float,
    )
    scaler = StandardScaler()
    X = scaler.fit_transform(feat)

    model = KMeans(n_clusters=3, random_state=42, n_init=20)
    labels = model.fit_predict(X)
    centers = scaler.inverse_transform(model.cluster_centers_)

    # Rank clusters by risk score based on loss and gas.
    scores = []
    for i, c in enumerate(centers):
        # c: [avg_loss, recent_loss, avg_gas, trend]
        score = float(0.5 * c[0] + 0.3 * c[1] + 0.2 * c[2] + 0.1 * c[3])
        scores.append((i, score))
    scores = sorted(scores, key=lambda x: x[1])
    risk_map = {scores[0][0]: "Low", scores[1][0]: "Medium", scores[2][0]: "High"}

    for i, r in enumerate(rows):
        r["cluster_id"] = int(labels[i])
        r["risk_zone"] = risk_map[int(labels[i])]
    return rows


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


@app.route("/westbengal.html", methods=["GET"])
def westbengal_page_html():
    return send_from_directory(FRONTEND_DIR, "westbengal.html")


@app.route("/district", methods=["GET"], strict_slashes=False)
def district_page():
    return send_from_directory(FRONTEND_DIR, "District.html")


@app.route("/district.html", methods=["GET"])
def district_page_html():
    return send_from_directory(FRONTEND_DIR, "District.html")


@app.route("/index.html", methods=["GET"])
def index_page_html():
    return send_from_directory(FRONTEND_DIR, "index.html")


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
    pred = build_combined_prediction(df_forest, df_drivers, n_years=5)
    return jsonify(
        clean_json(
            {
                "status": "success",
                "state": "West Bengal",
                "future_prediction_5_years": pred,
            }
        )
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
    district_name, hist = _district_history_frame(df_district, location)
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
    district_name, hist = _district_history_frame(df_district, location)
    if hist.empty:
        return jsonify({"status": "error", "message": f"District '{location}' not found."}), 404

    rate = pd.to_numeric(hist["Deforestation_Rate_%"], errors="coerce")
    pred = _district_prediction(hist, n_years=5)
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
                "location": district_name,
                "summary": summary,
                "environment": env,
                "future_prediction_5_years": pred,
                "risk_assessment": {"risk_level": risk},
            }
        )
    )


@app.route("/api/report/<location>", methods=["GET"])
def district_report(location):
    district_name, hist = _district_history_frame(df_district, location)
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
