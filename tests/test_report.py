import unittest
import pandas as pd
from blood_bank_ai.report import generate_audit_report_markdown, calculate_facility_readiness_score


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.inventory_df = pd.DataFrame(
            [
                {"blood_group": "A+", "available_units": 30, "low_stock_threshold": 10, "units_expiring_soon": 2},
                {"blood_group": "O-", "available_units": 12, "low_stock_threshold": 10, "units_expiring_soon": 0},
            ]
        )
        self.demand_df = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=20, freq="D"),
                "blood_group": ["A+"] * 20,
                "demand_units": [12] * 20,
            }
        )
        self.metrics_df = pd.DataFrame(
            [
                {"model": "Gradient Boosting", "MAE (units/day)": 8.94, "RMSE (units/day)": 11.58, "MAPE (%)": 9.2, "R² Score": 0.51}
            ]
        )
        self.key_forecasts = {"A+": 85, "B+": 70, "O-": 42, "AB+": 25}

    def test_readiness_score_calculation(self):
        score, rating = calculate_facility_readiness_score(self.inventory_df)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 100)
        self.assertIsInstance(rating, str)

    def test_generate_audit_report_markdown(self):
        md = generate_audit_report_markdown(
            inventory_df=self.inventory_df,
            demand_df=self.demand_df,
            metrics_df=self.metrics_df,
            key_forecasts=self.key_forecasts,
            facility_name="District Hospital",
        )
        self.assertIn("Clinical Blood Bank Operational Audit", md)
        self.assertIn("District Hospital", md)
        self.assertIn("Gradient Boosting", md)
        self.assertIn("https://venkateshmr29-coder-blood-bank-ai-app-98ipak.streamlit.app", md)


if __name__ == "__main__":
    unittest.main()
