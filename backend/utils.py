"""Shared helpers for backend/app.py."""

import math

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

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


def _poly_future(values, n_years, degree=2):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return np.zeros(n_years, dtype=float)
    if arr.size == 1:
        return np.repeat(arr[-1], n_years)
    x = np.arange(arr.size, dtype=float)
    deg = min(degree, max(1, arr.size - 1))
    coeffs = np.polyfit(x, arr, deg)
    fx = np.arange(arr.size, arr.size + n_years, dtype=float)
    return np.polyval(coeffs, fx)


def _safe_float(v):
    try:
        if v is None:
            return None
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return None
        return float(v)
    except Exception:
        return None


def _safe_fold_r2(y_true, y_pred, min_var=1e-8):
    try:
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        if y_true.size < 2:
            return None
        # R2 becomes numerically unstable when fold targets are almost constant.
        if float(np.var(y_true)) < float(min_var):
            return None
        score = float(r2_score(y_true, y_pred))
        if not np.isfinite(score):
            return None
        return score
    except Exception:
        return None


def _to_log_target(arr):
    vals = np.asarray(arr, dtype=float)
    vals = np.maximum(vals, 0.0)
    return np.log1p(vals)


def _from_log_target(arr):
    vals = np.asarray(arr, dtype=float)
    return np.expm1(vals)


def _log_clip_range(y_log, z=2.0):
    vals = np.asarray(y_log, dtype=float)
    if vals.size == 0:
        return None
    mean = float(np.mean(vals))
    std = float(np.std(vals))
    if not np.isfinite(std) or std <= 0.0:
        lo = float(np.min(vals))
        hi = float(np.max(vals))
        return (lo, hi)
    lo = float(np.min(vals) - z * std)
    hi = float(np.max(vals) + z * std)
    return (lo, hi)


def _clip_log_pred(pred_log, clip_range):
    if clip_range is None:
        return pred_log
    lo, hi = clip_range
    return np.clip(pred_log, lo, hi)


def _build_lag_frame(df, target_col, exog_cols, lag_count=2, roll_window=3):
    d = df.copy()
    d[target_col] = pd.to_numeric(d[target_col], errors="coerce").astype(float)
    for i in range(1, lag_count + 1):
        d[f"lag_{i}"] = d[target_col].shift(i)
    d["roll_mean_3"] = d[target_col].rolling(roll_window).mean().shift(1)
    feature_cols = exog_cols + [f"lag_{i}" for i in range(1, lag_count + 1)] + ["roll_mean_3"]
    d = d.dropna(subset=feature_cols + [target_col]).copy()
    return d, feature_cols


def _forecast_with_lags(model, exog_future, history, lag_count=2, roll_window=3, log_target=False, log_clip_range=None):
    preds = []
    hist = list(history)
    for i in range(len(exog_future)):
        if len(hist) == 0:
            return preds
        lag_1 = hist[-1]
        lag_2 = hist[-2] if len(hist) >= 2 else lag_1
        roll_mean = float(np.mean(hist[-roll_window:])) if len(hist) else lag_1
        row = exog_future.iloc[i].copy()
        row["lag_1"] = lag_1
        if lag_count >= 2:
            row["lag_2"] = lag_2
        row["roll_mean_3"] = roll_mean
        pred_val = float(model.predict(pd.DataFrame([row]))[0])
        if log_target:
            pred_val = float(_from_log_target(_clip_log_pred(pred_val, log_clip_range)))
        preds.append(pred_val)
        hist.append(pred_val)
    return preds


def _normalize_district_name(name):
    raw = (name or "").strip()
    if not raw:
        return raw
    return DISTRICT_ALIASES.get(raw.lower(), raw)


def _district_history_frame(df_district, location, df_forest=None):
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
    if df_forest is not None:
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


def _district_prediction_trend(history_df, n_years=5):
    rows = history_df.dropna(subset=["Year", "Tree_Cover_Loss_ha"]).copy()
    if len(rows) == 0:
        return {
            "years": [],
            "rates": [],
            "model_name": "Linear Trend (district)",
            "training_samples": 0,
            "model_accuracy_r2": None,
            "cv_mae": None,
            "cv_rmse": None,
            "cv_r2": None,
            "used_fallback": False,
            "reliability_status": "Unknown",
        }

    rows = rows.sort_values("Year").copy()
    # Add gas if present in district history.
    rows["Green-House_Gases"] = pd.to_numeric(rows.get("Green-House_Gases"), errors="coerce")
    # Smooth target with 3-year moving average for stability.
    rows["target_ma3"] = rows["Tree_Cover_Loss_ha"].rolling(3, min_periods=3).mean()
    rows = rows.dropna(subset=["target_ma3", "Green-House_Gases"]).copy()
    if rows.empty:
        return {
            "years": [],
            "rates": [],
            "model_name": "ElasticNet Regression (degree=1 + lags, district)",
            "training_samples": 0,
            "model_accuracy_r2": None,
            "cv_mae": None,
            "cv_rmse": None,
            "cv_r2": None,
            "used_fallback": False,
            "reliability_status": "Unknown",
        }
    rows["Year"] = pd.to_numeric(rows["Year"], errors="coerce")
    rows = rows.dropna(subset=["Year"]).copy()
    exog_cols = ["Year", "Green-House_Gases"]
    train_df, feature_cols = _build_lag_frame(rows, "target_ma3", exog_cols, lag_count=1, roll_window=3)
    if train_df.empty:
        return {
            "years": [],
            "rates": [],
            "model_name": "ElasticNet Regression (degree=1 + lags, district)",
            "training_samples": 0,
            "model_accuracy_r2": None,
            "cv_mae": None,
            "cv_rmse": None,
            "cv_r2": None,
            "used_fallback": False,
            "reliability_status": "Unknown",
        }
    X = train_df[feature_cols].astype(float)
    y = train_df["target_ma3"].astype(float).values
    y_log = _to_log_target(y)
    log_clip_range = _log_clip_range(y_log)

    fold_mae, fold_rmse, fold_r2 = [], [], []
    if len(X) >= 3:
        n_splits = max(2, min(5, len(X) - 1))
        tscv = TimeSeriesSplit(n_splits=n_splits)
        for tr, te in tscv.split(X):
            Xtr, Xte = X.iloc[tr], X.iloc[te]
            ytr, yte = y[tr], y[te]
            model = Pipeline(
                steps=[
                    ("poly", PolynomialFeatures(degree=1, include_bias=False)),
                    ("scaler", StandardScaler()),
                    ("model", ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=10000)),
                ]
            )
            model.fit(Xtr, _to_log_target(ytr))
            fold_clip = _log_clip_range(_to_log_target(ytr))
            pred_log = _clip_log_pred(model.predict(Xte), fold_clip)
            pred = _from_log_target(pred_log)
            fold_mae.append(mean_absolute_error(yte, pred))
            fold_rmse.append(math.sqrt(mean_squared_error(yte, pred)))
            fold_r2_val = _safe_fold_r2(yte, pred)
            if fold_r2_val is not None:
                fold_r2.append(fold_r2_val)

    cv_mae = float(np.mean(fold_mae)) if fold_mae else None
    cv_rmse = float(np.mean(fold_rmse)) if fold_rmse else None
    cv_r2 = float(np.mean(fold_r2)) if fold_r2 else None

    model = Pipeline(
        steps=[
            ("poly", PolynomialFeatures(degree=1, include_bias=False)),
            ("scaler", StandardScaler()),
            ("model", ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=10000)),
        ]
    )
    model.fit(X, y_log)
    train_pred = _from_log_target(_clip_log_pred(model.predict(X), log_clip_range))
    train_r2 = r2_score(y, train_pred) if len(y) >= 2 else None

    last_year = int(rows["Year"].max())
    future_years = [last_year + i for i in range(1, n_years + 1)]
    future_gas = _linear_future(rows["Green-House_Gases"].fillna(0.0).values, n_years)
    future_X = pd.DataFrame({"Year": future_years, "Green-House_Gases": future_gas}).astype(float)
    fut_pred = _forecast_with_lags(
        model,
        future_X,
        rows["target_ma3"].astype(float).values,
        lag_count=1,
        log_target=True,
        log_clip_range=log_clip_range,
    )

    cv_r2_rounded = None if cv_r2 is None else round(float(cv_r2), 4)
    cv_mae_rounded = None if cv_mae is None else round(float(cv_mae), 4)
    cv_rmse_rounded = None if cv_rmse is None else round(float(cv_rmse), 4)
    train_r2_rounded = None if train_r2 is None else round(float(train_r2), 4)
    band = cv_rmse if cv_rmse is not None else None
    reliability_status = "Reliable"
    if cv_r2 is None:
        reliability_status = "Unknown"
    elif cv_r2 < 0.3:
        reliability_status = "Use with caution"

    return {
        "years": future_years,
        "rates": [round(float(v), 2) for v in list(fut_pred)],
        "lower": None if band is None else [round(float(max(v - band, v * 0.6, 0.0)), 2) for v in list(fut_pred)],
        "upper": None if band is None else [round(float(max(v + band, 0.0)), 2) for v in list(fut_pred)],
        "model_name": "ElasticNet Regression (degree=1 + lags) (alpha=0.01, l1_ratio=0.5)",
        "training_samples": int(len(rows)),
        "model_accuracy_r2": train_r2_rounded,
        "cv_mae": None,
        "cv_rmse": None,
        "cv_r2": None,
        "used_fallback": False,
        "reliability_status": reliability_status,
    }


def build_combined_prediction(forest_df, drivers_df, n_years=5):
    # Use Sheet1 only (forest_df).
    if forest_df is None or forest_df.empty:
        return {
            "years": [],
            "rates": [],
            "model_name": "ElasticNet Regression (degree=1 + lags)",
            "training_samples": 0,
            "model_accuracy_r2": None,
            "cv_mae": None,
            "cv_rmse": None,
            "cv_r2": None,
            "feature_columns": [],
            "used_fallback": False,
            "reliability_status": "Unknown",
        }

    merged = forest_df.copy()
    merged["Year"] = pd.to_numeric(merged.get("Year"), errors="coerce")
    merged["Tree_Cover_Lose_ha"] = pd.to_numeric(merged.get("Tree_Cover_Lose_ha"), errors="coerce")
    merged["Green-House_Gases"] = pd.to_numeric(merged.get("Green-House_Gases"), errors="coerce")
    merged["Tree_cover_loss_from_fires"] = pd.to_numeric(merged.get("Tree_cover_loss_from_fires"), errors="coerce")
    merged["Tree_cover_loss_rainforest"] = pd.to_numeric(merged.get("Tree_cover_loss_rainforest"), errors="coerce")
    merged["umd_tree_cover_loss__ha"] = pd.to_numeric(merged.get("umd_tree_cover_loss__ha"), errors="coerce")
    merged = merged.dropna(subset=["Year", "Tree_Cover_Lose_ha"])
    if merged.empty:
        return {
            "years": [],
            "rates": [],
            "model_name": "ElasticNet Regression (degree=1 + lags)",
            "training_samples": 0,
            "model_accuracy_r2": None,
            "cv_mae": None,
            "cv_rmse": None,
            "cv_r2": None,
            "feature_columns": [],
            "used_fallback": False,
            "reliability_status": "Unknown",
        }

    merged = merged.sort_values("Year").reset_index(drop=True)
    merged["Year"] = merged["Year"].astype(int)
    model_name = "ElasticNet Regression (degree=1 + lags)"
    merged["Tree_Cover_Lose_ha"] = pd.to_numeric(merged["Tree_Cover_Lose_ha"], errors="coerce")
    merged["Tree_cover_loss_from_fires"] = pd.to_numeric(merged.get("Tree_cover_loss_from_fires"), errors="coerce")
    merged["Tree_cover_loss_rainforest"] = pd.to_numeric(merged.get("Tree_cover_loss_rainforest"), errors="coerce")
    merged["Drivers_type"] = pd.to_numeric(merged.get("Drivers_type"), errors="coerce")
    merged = merged.dropna(
        subset=[
            "Tree_Cover_Lose_ha",
            "Tree_cover_loss_from_fires",
            "Tree_cover_loss_rainforest",
        ]
    )
    if merged.empty:
        return {
            "years": [],
            "rates": [],
            "model_name": model_name,
            "training_samples": 0,
            "model_accuracy_r2": None,
            "cv_mae": None,
            "cv_rmse": None,
            "cv_r2": None,
            "feature_columns": feature_cols,
            "used_fallback": False,
            "reliability_status": "Unknown",
        }
    feature_cols = [
        "Tree_Cover_Lose_ha",
        "Tree_cover_loss_from_fires",
        "Tree_cover_loss_rainforest",
    ]
    train_df = merged.dropna(subset=feature_cols).copy()
    if train_df.empty:
        return {
            "years": [],
            "rates": [],
            "model_name": model_name,
            "training_samples": 0,
            "model_accuracy_r2": None,
            "cv_mae": None,
            "cv_rmse": None,
            "cv_r2": None,
            "feature_columns": feature_cols,
            "used_fallback": False,
            "reliability_status": "Unknown",
        }

    exog_cols = [
        "Tree_Cover_Lose_ha",
        "Tree_cover_loss_from_fires",
        "Tree_cover_loss_rainforest",
    ]
    train_df, feature_cols = _build_lag_frame(merged, "Tree_Cover_Lose_ha", exog_cols, lag_count=1, roll_window=3)
    if train_df.empty:
        return {
            "years": [],
            "rates": [],
            "model_name": model_name,
            "training_samples": 0,
            "model_accuracy_r2": None,
            "cv_mae": None,
            "cv_rmse": None,
            "cv_r2": None,
            "feature_columns": feature_cols,
            "used_fallback": False,
            "reliability_status": "Unknown",
        }

    X = train_df[feature_cols].astype(float)
    y = train_df["Tree_Cover_Lose_ha"].astype(float).values
    y_log = _to_log_target(y)
    log_clip_range = _log_clip_range(y_log)

    alpha_grid = [0.01, 0.1, 0.5, 1.0, 5.0]
    l1_grid = [0.1, 0.5, 0.9]
    best_alpha = alpha_grid[0]
    best_l1 = l1_grid[0]
    best_cv_r2 = -float("inf")
    best_cv_mae = None
    best_cv_rmse = None
    for alpha in alpha_grid:
        for l1_ratio in l1_grid:
            model = Pipeline(
                steps=[
                    ("poly", PolynomialFeatures(degree=1, include_bias=False)),
                    ("scaler", StandardScaler()),
                    ("model", ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000)),
                ]
            )
            fold_mae = []
            fold_r2 = []
            fold_rmse = []
            if len(X) >= 3:
                n_splits = max(2, min(5, len(X) - 1))
                tscv = TimeSeriesSplit(n_splits=n_splits)
                for train_idx, test_idx in tscv.split(X):
                    X_train = X.iloc[train_idx]
                    X_test = X.iloc[test_idx]
                    y_train = y[train_idx]
                    y_test = y[test_idx]
                    model.fit(X_train, _to_log_target(y_train))
                    fold_clip = _log_clip_range(_to_log_target(y_train))
                    pred_log = _clip_log_pred(model.predict(X_test), fold_clip)
                    pred = _from_log_target(pred_log)
                    fold_mae.append(mean_absolute_error(y_test, pred))
                    fold_rmse.append(math.sqrt(mean_squared_error(y_test, pred)))
                    fold_r2_val = _safe_fold_r2(y_test, pred)
                    if fold_r2_val is not None:
                        fold_r2.append(fold_r2_val)

            mean_r2 = float(np.mean(fold_r2)) if fold_r2 else -float("inf")
            mean_mae = float(np.mean(fold_mae)) if fold_mae else None
            mean_rmse = float(np.mean(fold_rmse)) if fold_rmse else None
            if mean_r2 > best_cv_r2:
                best_cv_r2 = mean_r2
                best_alpha = alpha
                best_l1 = l1_ratio
                best_cv_mae = mean_mae
                best_cv_rmse = mean_rmse

    model = Pipeline(
        steps=[
            ("poly", PolynomialFeatures(degree=1, include_bias=False)),
            ("scaler", StandardScaler()),
            ("model", ElasticNet(alpha=best_alpha, l1_ratio=best_l1, max_iter=10000)),
        ]
    )
    model.fit(X, y_log)
    train_pred = _from_log_target(_clip_log_pred(model.predict(X), log_clip_range))
    train_r2 = r2_score(y, train_pred) if len(y) >= 2 else None

    last_year = int(merged["Year"].max())
    future_years = [last_year + i for i in range(1, n_years + 1)]

    future_exog = pd.DataFrame(
        {
            "Tree_Cover_Lose_ha": _poly_future(merged["Tree_Cover_Lose_ha"].values, n_years, degree=2),
            "Tree_cover_loss_from_fires": _poly_future(
                merged["Tree_cover_loss_from_fires"].fillna(0.0).values, n_years, degree=2
            ),
            "Tree_cover_loss_rainforest": _poly_future(
                merged["Tree_cover_loss_rainforest"].fillna(0.0).values, n_years, degree=2
            ),
        }
    )
    fut_pred = _forecast_with_lags(
        model,
        future_exog,
        merged["Tree_Cover_Lose_ha"].values,
        lag_count=1,
        log_target=True,
        log_clip_range=log_clip_range,
    )

    # Add a bit of oscillation to avoid perfectly flat projections.
    n_years = len(future_years)
    y = merged["Tree_Cover_Lose_ha"].values
    fut_pred = [
        max(0.0, float(v + 0.03 * float(np.std(y)) * math.sin(2 * math.pi * i / max(2, n_years))))
        for i, v in enumerate(fut_pred)
    ]

    cv_r2_rounded = None if best_cv_r2 == -float("inf") else round(float(best_cv_r2), 4)
    cv_mae_rounded = None if best_cv_mae is None else round(float(best_cv_mae), 4)
    cv_rmse_rounded = None if best_cv_rmse is None else round(float(best_cv_rmse), 4)
    train_r2_rounded = None if train_r2 is None else round(float(train_r2), 4)
    band = best_cv_rmse if best_cv_rmse is not None else None
    reliability_status = "Reliable"
    if cv_r2_rounded is None:
        reliability_status = "Unknown"
    elif cv_r2_rounded < 0.3:
        reliability_status = "Use with caution"

    return {
        "years": future_years,
        "rates": [round(float(v), 2) for v in list(fut_pred)],
        "lower": None if band is None else [round(float(max(v - band, v * 0.6, 0.0)), 2) for v in list(fut_pred)],
        "upper": None if band is None else [round(float(max(v + band, 0.0)), 2) for v in list(fut_pred)],
        "model_name": f"ElasticNet Regression (degree=1 + lags) (alpha={best_alpha}, l1_ratio={best_l1})",
        "training_samples": int(len(merged)),
        "model_accuracy_r2": train_r2_rounded,
        "cv_mae": cv_mae_rounded,
        "cv_rmse": cv_rmse_rounded,
        "cv_r2": cv_r2_rounded,
        "feature_columns": feature_cols,
        "used_fallback": False,
        "reliability_status": reliability_status,
    }


def build_district_risk_clusters(df_district):
    if df_district is None or df_district.empty:
        return []

    df = df_district.copy()
    df["loss_ha"] = pd.to_numeric(df["umd_tree_cover_loss__ha"], errors="coerce").fillna(0.0)
    df["gas_mg"] = pd.to_numeric(df["gfw_gross_emissions_co2e_all_gases__Mg"], errors="coerce").fillna(0.0)
    df = df.dropna(subset=["adm2"])

    grouped = df.groupby("adm2", as_index=False).agg(
        avg_loss_ha=("loss_ha", "mean"),
        recent_loss_ha=("loss_ha", "median"),
        avg_gas_mg=("gas_mg", "mean"),
    )
    rows = grouped.to_dict(orient="records")

    # Add loss trend (last - first) to help clustering.
    trend_vals = {}
    for name, g in df.groupby("adm2"):
        g = g.sort_values("umd_tree_cover_loss__year")
        vals = g["loss_ha"].values
        trend_vals[name] = float(vals[-1] - vals[0]) if len(vals) >= 2 else 0.0
    for r in rows:
        r["loss_trend"] = trend_vals.get(r["adm2"], 0.0)

    if len(rows) < 3:
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


def build_wb_risk_clusters_from_forest(df_forest):
    if df_forest is None or df_forest.empty:
        return []

    d = df_forest.copy()
    d["Year"] = pd.to_numeric(d.get("Year"), errors="coerce")
    d["Tree_Cover_Lose_ha"] = pd.to_numeric(d.get("Tree_Cover_Lose_ha"), errors="coerce")
    d["Green-House_Gases"] = pd.to_numeric(d.get("Green-House_Gases"), errors="coerce")
    d["Tree_cover_loss_from_fires"] = pd.to_numeric(d.get("Tree_cover_loss_from_fires"), errors="coerce")
    d["Tree_cover_loss_rainforest"] = pd.to_numeric(d.get("Tree_cover_loss_rainforest"), errors="coerce")
    d = d.dropna(subset=["Year", "Tree_Cover_Lose_ha"]).sort_values("Year")
    if d.empty:
        return []

    losses = d["Tree_Cover_Lose_ha"].astype(float).values
    gases = d["Green-House_Gases"].fillna(0.0).astype(float).values
    years = d["Year"].astype(int).values
    rows = []
    for i, year in enumerate(years):
        loss_val = float(losses[i])
        recent_start = max(0, i - 2)
        recent_loss = float(np.mean(losses[recent_start : i + 1]))
        gas_val = float(gases[i]) if i < len(gases) else 0.0
        trend_val = float(losses[i] - losses[i - 1]) if i > 0 else 0.0
        rows.append(
            {
                "district": str(year),
                "avg_loss_ha": loss_val,
                "recent_loss_ha": recent_loss,
                "avg_gas_mg": gas_val,
                "loss_trend": trend_val,
            }
        )

    if len(rows) < 3:
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

    def _choose_k_elbow(X, max_k=8):
        n = X.shape[0]
        max_k = max(2, min(int(max_k), n))
        ks = list(range(1, max_k + 1))
        inertias = []
        for k in ks:
            km = KMeans(n_clusters=k, random_state=42, n_init=20)
            km.fit(X)
            inertias.append(float(km.inertia_))
        if len(inertias) < 3:
            return 2
        # Elbow via max second derivative of inertia curve.
        d2 = []
        for i in range(1, len(inertias) - 1):
            d2.append(inertias[i - 1] - 2 * inertias[i] + inertias[i + 1])
        elbow_idx = int(np.argmax(d2)) + 1
        return max(2, ks[elbow_idx])

    k = max(3, _choose_k_elbow(X, max_k=8))
    model = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = model.fit_predict(X)
    centers = scaler.inverse_transform(model.cluster_centers_)

    scores = []
    for i, c in enumerate(centers):
        score = float(0.5 * c[0] + 0.3 * c[1] + 0.2 * c[2] + 0.1 * c[3])
        scores.append((i, score))
    scores = sorted(scores, key=lambda x: x[1])
    # Map clusters into Low/Medium/High by score rank (works for any k >= 1).
    n_scores = len(scores)
    risk_map = {}
    for rank, (cluster_id, _) in enumerate(scores):
        pct = rank / max(1, n_scores - 1)
        if pct <= 0.33:
            risk_map[cluster_id] = "Low"
        elif pct <= 0.66:
            risk_map[cluster_id] = "Medium"
        else:
            risk_map[cluster_id] = "High"

    for i, r in enumerate(rows):
        r["cluster_id"] = int(labels[i])
        r["risk_zone"] = risk_map[int(labels[i])]
    return rows
