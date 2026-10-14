"""Unit tests for the inventory optimization and safety stock module."""

import unittest
import pandas as pd
from blood_bank_ai.inventory import compute_inventory_metrics, generate_inventory_health_summary


class InventoryTests(unittest.TestCase):
    def setUp(self):
        self.history_demand = pd.Series([10, 12, 10, 8, 11, 9, 10, 10, 11, 9])
        self.inventory_df = pd.DataFrame(
            [
                {
                    "blood_group": "A+",
                    "available_units": 45,
                    "low_stock_threshold": 20,
                    "units_expiring_soon": 2,
                },
                {
                    "blood_group": "O-",
                    "available_units": 8,
                    "low_stock_threshold": 15,
                    "units_expiring_soon": 1,
                },
            ]
        )
        self.demand_df = pd.DataFrame(
            {
                "blood_group": ["A+"] * 10 + ["O-"] * 10,
                "demand_units": [10, 12, 10, 8, 11, 9, 10, 10, 11, 9] * 2,
            }
        )

    def test_compute_inventory_metrics_calculates_sensible_values(self):
        metrics = compute_inventory_metrics(
            current_stock=25,
            history_demand=self.history_demand,
            lead_time_days=2,
            service_level=0.95,
        )

        self.assertAlmostEqual(metrics["avg_daily_demand"], 10.0, places=1)
        self.assertGreater(metrics["safety_stock"], 0)
        self.assertGreater(metrics["reorder_point"], metrics["safety_stock"])
        self.assertEqual(metrics["stock_status"], "HEALTHY")

    def test_compute_inventory_metrics_triggers_reorder_when_low(self):
        metrics = compute_inventory_metrics(
            current_stock=5,
            history_demand=self.history_demand,
            lead_time_days=2,
            service_level=0.95,
        )
        self.assertIn(metrics["stock_status"], ["REORDER_TRIGGERED", "CRITICAL_DEFICIT"])
        self.assertGreater(metrics["recommended_order"], 0)

    def test_generate_inventory_health_summary(self):
        summary = generate_inventory_health_summary(
            self.inventory_df, self.demand_df, lead_time_days=2, service_level=0.95
        )
        self.assertEqual(len(summary), 2)
        self.assertIn("Safety Stock", summary.columns)
        self.assertIn("Reorder Point (ROP)", summary.columns)
        self.assertIn("Risk Category", summary.columns)


if __name__ == "__main__":
    unittest.main()
