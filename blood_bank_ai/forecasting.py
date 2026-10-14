"""
Time-Series Feature Engineering and Multi-Model Forecasting Engine.

Trains and benchmarks Machine Learning regressors (Random Forest,
Gradient Boosting, Ridge Regression) against a 7-day seasonal naive baseline.
Provides recursive multi-step forecasting with uncertainty intervals and
feature importance interpretability.
"""

from typing import Dict, List, Tuple, Any
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore", category=RuntimeWarning)


FEATURE_COLUMNS = [
    "dow",
    "month",
    "dayofyear",
    "lag_1",
    "lag_7",
    "lag_14",
    "rolling_mean_7",
    "rolling_mean_14",
]


def make_features(series_df: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs time-based and lag features for chronological modeling.
    Excludes the target day from rolling windows to prevent data leakage.
    """
    data = series_df.sort_values("date").copy()
    data["dow"] = data["date"].dt.dayofweek
    data["month"] = data["date"].dt.month
    data["dayofyear"] = data["date"].dt.dayofyear
    data["lag_1"] = data["demand_units"].shift(1)
    data["lag_7"] = data["demand_units"].shift(7)
    data["lag_14"] = data["demand_units"].shift(14)
    data["rolling_mean_7"] = data["demand_units"].shift(1).rolling(7).mean()
    data["rolling_mean_14"] = data["demand_units"].shift(1).rolling(14).mean()
    return data.dropna().reset_index(drop=True)


def calculate_mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Computes Mean Absolute Percentage Error, ignoring zero-actual records."""
    mask = actual > 0
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100.0)


def train_and_evaluate(group_df: pd.DataFrame) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame, Dict[str, np.ndarray], pd.DataFrame]:
    """
    Trains multiple models on chronological split (first 80% train, last 20% test).
    Evaluates MAE, RMSE, MAPE, and R2 on holdout set.
    Refits models on full feature history for future forecasting.
    """
    features = make_features(group_df)
    if len(features) < 40:
        raise ValueError("At least 55 daily records are recommended for this demonstration.")

    split = int(len(features) * 0.8)
    train, test = features.iloc[:split], features.iloc[split:]

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=250, random_state=42, min_samples_leaf=2
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=150, learning_rate=0.05, max_depth=2, random_state=42
        ),
        "Ridge Regression": Ridge(alpha=1.0, random_state=42),
    }

    predictions = {
        "7-day seasonal naive": np.maximum(0, test["lag_7"].to_numpy())
    }

    for name, model in models.items():
        model.fit(train[FEATURE_COLUMNS], train["demand_units"])
        predictions[name] = np.maximum(
            0, model.predict(test[FEATURE_COLUMNS])
        )

    y_test = test["demand_units"].to_numpy()
    metrics = []
    for name, pred in predictions.items():
        mae = mean_absolute_error(y_test, pred)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        mape = calculate_mape(y_test, pred)
        r2 = r2_score(y_test, pred) if len(y_test) > 1 else 0.0
        metrics.append(
            {
                "model": name,
                "MAE (units/day)": round(mae, 3),
                "RMSE (units/day)": round(rmse, 3),
                "MAPE (%)": round(mape, 2),
                "R² Score": round(r2, 3),
            }
        )

    # Refit models on full features for forward recursive forecasting
    for model in models.values():
        model.fit(features[FEATURE_COLUMNS], features["demand_units"])

    return models, features, test, predictions, pd.DataFrame(metrics)


def get_feature_importances(model: Any) -> pd.DataFrame:
    """Extracts normalized feature importance weights from fitted tree models."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        df = pd.DataFrame(
            {"Feature": FEATURE_COLUMNS, "Importance": importances}
        ).sort_values("Importance", ascending=False)
        df["Importance (%)"] = (df["Importance"] * 100).round(2)
        return df
    elif hasattr(model, "coef_"):
        coefs = np.abs(model.coef_)
        total = np.sum(coefs) if np.sum(coefs) > 0 else 1.0
        df = pd.DataFrame(
            {"Feature": FEATURE_COLUMNS, "Importance": coefs / total}
        ).sort_values("Importance", ascending=False)
        df["Importance (%)"] = (df["Importance"] * 100).round(2)
        return df
    return pd.DataFrame()


def forecast_next_days(group_df: pd.DataFrame, model: Any, horizon: int = 7) -> pd.DataFrame:
    """
    Generates multi-step recursive forecasts into the future.
    Each predicted value is iteratively appended to simulate lags for the next step.
    """
    history = group_df.sort_values("date")[["date", "demand_units"]].copy()
    output = []
    for _ in range(horizon):
        next_date = history["date"].max() + pd.Timedelta(days=1)
        values = history["demand_units"].astype(float).to_numpy()
        row = {
            "dow": next_date.dayofweek,
            "month": next_date.month,
            "dayofyear": next_date.dayofyear,
            "lag_1": values[-1],
            "lag_7": values[-7] if len(values) >= 7 else values.mean(),
            "lag_14": values[-14] if len(values) >= 14 else values.mean(),
            "rolling_mean_7": values[-7:].mean(),
            "rolling_mean_14": values[-14:].mean(),
        }
        pred_val = float(model.predict(pd.DataFrame([row])[FEATURE_COLUMNS])[0])
        predicted = max(0, int(round(pred_val)))
        
        output.append(
            {"date": next_date, "predicted_demand_units": predicted}
        )
        history = pd.concat(
            [
                history,
                pd.DataFrame([{"date": next_date, "demand_units": predicted}]),
            ],
            ignore_index=True,
        )
    return pd.DataFrame(output)


def forecast_with_confidence_bounds(
    group_df: pd.DataFrame,
    model: Any,
    rmse: float,
    horizon: int = 7,
    confidence_level: float = 0.95,
) -> pd.DataFrame:
    """
    Generates multi-step recursive forecast with analytical confidence intervals
    scaling with sqrt(step) as recursive uncertainty compounds over time.
    """
    base_forecast = forecast_next_days(group_df, model, horizon)
    confidence_z = {
        0.90: 1.645,
        0.95: 1.960,
        0.98: 2.326,
        0.99: 2.576,
    }
    z = confidence_z.get(confidence_level, 1.96)
    
    lowers = []
    uppers = []
    for step, row in enumerate(base_forecast.itertuples(), start=1):
        # Compound uncertainty over steps
        step_uncertainty = z * rmse * np.sqrt(step * 0.35 + 0.65)
        pred = row.predicted_demand_units
        lowers.append(max(0, int(round(pred - step_uncertainty))))
        uppers.append(int(round(pred + step_uncertainty)))
        
    base_forecast["lower_bound"] = lowers
    base_forecast["upper_bound"] = uppers
    return base_forecast


def evaluate_all_blood_groups(demand_df: pd.DataFrame) -> pd.DataFrame:
    """
    Evaluates and compares models across all 8 blood groups simultaneously.
    Returns a unified summary dataframe suitable for project reports and viva defense.
    """
    groups = sorted(demand_df["blood_group"].unique())
    records = []
    
    for bg in groups:
        gdf = demand_df[demand_df["blood_group"] == bg].copy()
        if len(gdf) < 55:
            continue
        _, _, _, _, metrics = train_and_evaluate(gdf)
        
        # Extract MAE and RMSE for each model
        row = {"Blood Group": bg}
        for _, mrow in metrics.iterrows():
            mname = mrow["model"]
            row[f"{mname} MAE"] = mrow["MAE (units/day)"]
            row[f"{mname} RMSE"] = mrow["RMSE (units/day)"]
            row[f"{mname} R²"] = mrow["R² Score"]
            
        # Determine winning model
        ml_models = [m for m in metrics["model"] if m != "7-day seasonal naive"]
        ml_metrics = metrics[metrics["model"].isin(ml_models)]
        best_model = ml_metrics.sort_values("MAE (units/day)").iloc[0]["model"]
        row["Best Model"] = best_model
        records.append(row)
        
    return pd.DataFrame(records)


def compute_stock_risk_status(
    current_stock: int, safety_stock: int, reorder_point: int, forecast_demand: int
) -> Dict[str, str]:
    """
    Computes 3-tier stock risk classification matching Review 0 Slides 9 & 12:
    - 'Critical': current_stock <= safety_stock or current_stock < forecast_demand
    - 'Low': current_stock <= reorder_point
    - 'Normal': current_stock > reorder_point
    """
    if current_stock <= 0 or current_stock <= safety_stock or current_stock < forecast_demand:
        return {
            "level": "Critical",
            "badge": "🚨 Critical Deficit",
            "css_class": "badge-danger",
            "description": "Immediate stockout risk. Urgent donation campaign required.",
        }
    elif current_stock <= reorder_point:
        return {
            "level": "Low",
            "badge": "⚠️ Low Stock Warning",
            "css_class": "badge-warning",
            "description": "Below Reorder Point. Schedule donor blood drive.",
        }
    else:
        return {
            "level": "Normal",
            "badge": "✅ Normal Buffer",
            "css_class": "badge-healthy",
            "description": "Stock adequately meets demand and buffer targets.",
        }


def compute_key_group_forecasts(
    demand_df: pd.DataFrame, horizon: int = 7, model_name: str = "Gradient Boosting"
) -> Dict[str, int]:
    """
    Computes forecasted demand for the 4 key representative blood groups
    (A+, B+, O-, AB+) directly matching Review 0 Slide 13 Expected Results banner.
    """
    key_groups = ["A+", "B+", "O-", "AB+"]
    results = {}
    for bg in key_groups:
        gdf = demand_df[demand_df["blood_group"] == bg].copy()
        if len(gdf) >= 55:
            try:
                models, _, _, _, _ = train_and_evaluate(gdf)
                target_model = models.get(model_name) or models.get("Gradient Boosting") or models.get("Random Forest")
                if target_model:
                    fc = forecast_next_days(gdf, target_model, horizon=horizon)
                    results[bg] = int(fc["predicted_demand_units"].sum())
                else:
                    results[bg] = int(gdf["demand_units"].tail(horizon).sum())
            except Exception:
                results[bg] = int(gdf["demand_units"].tail(horizon).sum())
        else:
            results[bg] = int(gdf["demand_units"].tail(horizon).sum()) if not gdf.empty else 0
    return results

