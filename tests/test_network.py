import unittest
import pandas as pd
from blood_bank_ai.network import (
    generate_facility_comparison_matrix,
    optimize_inter_hospital_transfers,
    execute_network_transfer,
)


class NetworkTests(unittest.TestCase):
    def setUp(self):
        self.inventories = {
            "National Blood Centre": pd.DataFrame(
                [
                    {"blood_group": "A+", "available_units": 80, "low_stock_threshold": 20, "units_expiring_soon": 5},
                    {"blood_group": "O-", "available_units": 35, "low_stock_threshold": 10, "units_expiring_soon": 2},
                ]
            ),
            "District Hospital": pd.DataFrame(
                [
                    {"blood_group": "A+", "available_units": 15, "low_stock_threshold": 15, "units_expiring_soon": 1},
                    {"blood_group": "O-", "available_units": 4, "low_stock_threshold": 10, "units_expiring_soon": 0},
                ]
            ),
            "Rural Hospital": pd.DataFrame(
                [
                    {"blood_group": "A+", "available_units": 2, "low_stock_threshold": 8, "units_expiring_soon": 0},
                    {"blood_group": "O-", "available_units": 1, "low_stock_threshold": 5, "units_expiring_soon": 0},
                ]
            ),
        }

    def test_facility_comparison_matrix(self):
        matrix = generate_facility_comparison_matrix(self.inventories)
        self.assertIn("Blood Group", matrix.columns)
        self.assertIn("Regional Total Stock", matrix.columns)
        self.assertEqual(len(matrix), 8)

    def test_optimize_inter_hospital_transfers(self):
        opt = optimize_inter_hospital_transfers(self.inventories, safety_buffer=10)
        transfers_df = opt["transfers_df"]
        self.assertIsInstance(transfers_df, pd.DataFrame)
        self.assertGreater(opt["total_units_rebalanced"], 0)
        self.assertIn("Origin Facility", transfers_df.columns)
        self.assertIn("Destination Facility", transfers_df.columns)

    def test_execute_network_transfer(self):
        updated = execute_network_transfer(
            self.inventories,
            source_fac="National Blood Centre",
            dest_fac="Rural Hospital",
            blood_group="A+",
            units=10,
        )
        src_stock = updated["National Blood Centre"].set_index("blood_group").loc["A+", "available_units"]
        dest_stock = updated["Rural Hospital"].set_index("blood_group").loc["A+", "available_units"]
        self.assertEqual(src_stock, 70)
        self.assertEqual(dest_stock, 12)


if __name__ == "__main__":
    unittest.main()
