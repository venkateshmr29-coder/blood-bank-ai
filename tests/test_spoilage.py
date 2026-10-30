import unittest
import pandas as pd
from blood_bank_ai.spoilage import (
    generate_component_batch_breakdown,
    compute_spoilage_risk_matrix,
    generate_fefo_dispatch_queue,
)


class SpoilageTests(unittest.TestCase):
    def setUp(self):
        self.inventory_df = pd.DataFrame(
            [
                {"blood_group": "A+", "available_units": 40, "low_stock_threshold": 10, "units_expiring_soon": 5},
                {"blood_group": "O-", "available_units": 10, "low_stock_threshold": 8, "units_expiring_soon": 3},
            ]
        )
        self.demand_df = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=20, freq="D"),
                "blood_group": ["A+"] * 10 + ["O-"] * 10,
                "demand_units": [4] * 20,
            }
        )

    def test_component_batch_breakdown(self):
        batch_df = generate_component_batch_breakdown(self.inventory_df)
        self.assertEqual(len(batch_df), 2)
        self.assertIn("Critical (<3 Days)", batch_df.columns)
        self.assertIn("Imminent Waste Risk (%)", batch_df.columns)

    def test_compute_spoilage_risk_matrix(self):
        risk_df = compute_spoilage_risk_matrix(self.inventory_df, self.demand_df)
        self.assertEqual(len(risk_df), 2)
        self.assertIn("Financial Risk (₹)", risk_df.columns)
        self.assertIn("Turnover Health", risk_df.columns)

    def test_generate_fefo_dispatch_queue(self):
        batch_df = generate_component_batch_breakdown(self.inventory_df)
        fefo = generate_fefo_dispatch_queue(batch_df, requested_group="A+", units_needed=8)
        self.assertTrue(fefo["fulfilled"])
        self.assertEqual(fefo["units_allocated"], 8)
        # Check critical units were taken first (there were 5 critical units)
        self.assertEqual(fefo["cohort_breakdown"]["Critical (<3 Days)"], 5)
        self.assertEqual(fefo["waste_prevented_units"], 5)


if __name__ == "__main__":
    unittest.main()
