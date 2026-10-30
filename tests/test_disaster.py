import unittest
import pandas as pd
from blood_bank_ai.disaster import (
    get_preset_disaster_scenarios,
    simulate_disaster_surge,
)


class DisasterTests(unittest.TestCase):
    def setUp(self):
        self.inventory_df = pd.DataFrame(
            [
                {"blood_group": "A+", "available_units": 30, "low_stock_threshold": 10, "units_expiring_soon": 2},
                {"blood_group": "O+", "available_units": 25, "low_stock_threshold": 10, "units_expiring_soon": 1},
                {"blood_group": "O-", "available_units": 8, "low_stock_threshold": 10, "units_expiring_soon": 0},
                {"blood_group": "B+", "available_units": 20, "low_stock_threshold": 10, "units_expiring_soon": 0},
                {"blood_group": "A-", "available_units": 5, "low_stock_threshold": 5, "units_expiring_soon": 0},
                {"blood_group": "B-", "available_units": 4, "low_stock_threshold": 5, "units_expiring_soon": 0},
                {"blood_group": "AB+", "available_units": 15, "low_stock_threshold": 5, "units_expiring_soon": 0},
                {"blood_group": "AB-", "available_units": 3, "low_stock_threshold": 3, "units_expiring_soon": 0},
            ]
        )

    def test_preset_scenarios_available(self):
        scenarios = get_preset_disaster_scenarios()
        self.assertGreater(len(scenarios), 2)
        self.assertIn("Level-1 Highway Multi-Vehicle Collision (Trauma Surge)", scenarios)

    def test_simulate_disaster_surge(self):
        scenario_key = "Level-1 Highway Multi-Vehicle Collision (Trauma Surge)"
        res = simulate_disaster_surge(self.inventory_df, scenario_name=scenario_key)

        self.assertGreater(res["total_requested"], 0)
        self.assertGreater(res["total_allocated"], 0)
        self.assertLess(res["post_stock_total"], res["pre_stock_total"])
        self.assertIsInstance(res["impact_df"], pd.DataFrame)
        self.assertTrue(len(res["emergency_checklist"]) > 0)
        self.assertGreater(res["hours_of_runway"], 0)


if __name__ == "__main__":
    unittest.main()
