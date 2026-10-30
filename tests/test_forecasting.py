import unittest

import pandas as pd

from blood_bank_ai.forecasting import FEATURE_COLUMNS, forecast_next_days, make_features


class ConstantModel:
    def predict(self, features):
        return [4] * len(features)


class ForecastingTests(unittest.TestCase):
    def setUp(self):
        self.history = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=30, freq="D"),
                "demand_units": range(1, 31),
            }
        )

    def test_features_use_prior_history_and_include_expected_columns(self):
        features = make_features(self.history)

        self.assertEqual(len(features), 16)
        self.assertEqual(features.loc[0, "lag_1"], 14)
        self.assertEqual(features.loc[0, "lag_14"], 1)
        self.assertEqual(features.loc[0, "rolling_mean_7"], 11)
        self.assertTrue(set(FEATURE_COLUMNS).issubset(features.columns))

    def test_forecast_returns_requested_horizon_and_future_dates(self):
        forecast = forecast_next_days(self.history, ConstantModel(), horizon=3)

        self.assertEqual(len(forecast), 3)
        self.assertEqual(
            forecast["date"].tolist(),
            list(pd.date_range("2024-01-31", periods=3, freq="D")),
        )
        self.assertEqual(forecast["predicted_demand_units"].tolist(), [4, 4, 4])

    def test_forecast_with_confidence_bounds(self):
        from blood_bank_ai.forecasting import forecast_with_confidence_bounds
        forecast = forecast_with_confidence_bounds(
            self.history, ConstantModel(), rmse=1.5, horizon=4, confidence_level=0.95
        )
        self.assertEqual(len(forecast), 4)
        self.assertIn("lower_bound", forecast.columns)
        self.assertIn("upper_bound", forecast.columns)
        for _, row in forecast.iterrows():
            self.assertLessEqual(row["lower_bound"], row["predicted_demand_units"])
            self.assertGreaterEqual(row["upper_bound"], row["predicted_demand_units"])

    def test_calculate_mape(self):
        import numpy as np
        from blood_bank_ai.forecasting import calculate_mape
        actual = np.array([10.0, 20.0, 0.0])
        pred = np.array([11.0, 18.0, 5.0])
        mape = calculate_mape(actual, pred)
        # only first two non-zero items considered: |10-11|/10 = 0.1, |20-18|/20 = 0.1 -> mean = 0.1 * 100 = 10.0%
        self.assertAlmostEqual(mape, 10.0, places=1)

    def test_train_and_evaluate(self):
        from blood_bank_ai.forecasting import train_and_evaluate, get_feature_importances
        long_history = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=60, freq="D"),
                "demand_units": [5 + (i % 7) for i in range(60)],
            }
        )
        models, features, test, predictions, metrics_df = train_and_evaluate(long_history)
        self.assertIn("Linear Regression", models)
        self.assertIn("Random Forest", models)
        self.assertIn("Gradient Boosting", models)
        self.assertIn("Ridge Regression", models)
        self.assertGreater(len(metrics_df), 0)

        rf_importances = get_feature_importances(models["Random Forest"])
        self.assertEqual(len(rf_importances), len(FEATURE_COLUMNS))
        self.assertIn("Importance (%)", rf_importances.columns)

        lr_importances = get_feature_importances(models["Linear Regression"])
        self.assertEqual(len(lr_importances), len(FEATURE_COLUMNS))

    def test_compute_stock_risk_status(self):
        from blood_bank_ai.forecasting import compute_stock_risk_status
        # Critical if stock <= safety stock or < forecast demand
        crit = compute_stock_risk_status(current_stock=5, safety_stock=10, reorder_point=25, forecast_demand=15)
        self.assertEqual(crit["level"], "Critical")

        # Low if stock <= reorder point
        low = compute_stock_risk_status(current_stock=20, safety_stock=10, reorder_point=25, forecast_demand=15)
        self.assertEqual(low["level"], "Low")

        # Normal if stock > reorder point and >= forecast demand
        norm = compute_stock_risk_status(current_stock=40, safety_stock=10, reorder_point=25, forecast_demand=15)
        self.assertEqual(norm["level"], "Normal")

    def test_compute_key_group_forecasts(self):
        from blood_bank_ai.forecasting import compute_key_group_forecasts
        dummy_df = pd.DataFrame(
            {
                "date": list(pd.date_range("2024-01-01", periods=60, freq="D")) * 4,
                "blood_group": ["A+"] * 60 + ["B+"] * 60 + ["O-"] * 60 + ["AB+"] * 60,
                "demand_units": [5] * 240,
            }
        )
        res = compute_key_group_forecasts(dummy_df, horizon=5)
        self.assertIn("A+", res)
        self.assertIn("B+", res)
        self.assertIn("O-", res)
        self.assertIn("AB+", res)
        self.assertGreater(res["A+"], 0)

    def test_weighted_ensemble_and_recommendation(self):
        from blood_bank_ai.forecasting import train_and_evaluate, recommend_best_model
        history = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=60, freq="D"),
                "demand_units": [8 + (i % 6) for i in range(60)],
            }
        )
        models, _, _, predictions, metrics_df = train_and_evaluate(history)
        self.assertIn("Weighted Ensemble (AI Stacking)", models)
        self.assertIn("Weighted Ensemble (AI Stacking)", predictions)
        self.assertIn("Weighted Ensemble (AI Stacking)", metrics_df["model"].values)

        best_info = recommend_best_model(metrics_df)
        self.assertIn("best_model", best_info)
        self.assertIn("rationale", best_info)

    def test_detect_demand_anomalies(self):
        from blood_bank_ai.forecasting import detect_demand_anomalies
        vals = [10] * 30
        vals[15] = 45  # spike anomaly
        vals[25] = 1   # dip anomaly
        df = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=30, freq="D"),
                "blood_group": ["O-"] * 30,
                "demand_units": vals,
            }
        )
        anomalies = detect_demand_anomalies(df, threshold_z=2.0)
        self.assertGreater(len(anomalies), 0)
        self.assertIn("z_score", anomalies.columns)
        self.assertIn("Anomaly Type", anomalies.columns)


if __name__ == "__main__":
    unittest.main()

