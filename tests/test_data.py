"""Unit tests for the data loading and validation module."""

import unittest
from io import StringIO
import pandas as pd
from blood_bank_ai.data import (
    generate_sample_donor_registry,
    get_dataset_summary,
    load_demand_data,
    load_inventory_data,
)


class DataTests(unittest.TestCase):
    def test_load_demand_data_valid(self):
        csv_data = "date,blood_group,demand_units\n2024-01-01,A+,5\n2024-01-02,A+,8\n"
        df = load_demand_data(StringIO(csv_data))
        self.assertEqual(len(df), 2)
        self.assertIn("demand_units", df.columns)
        self.assertEqual(df["demand_units"].tolist(), [5, 8])

    def test_load_demand_data_missing_column_raises_error(self):
        csv_data = "date,blood_group\n2024-01-01,A+\n"
        with self.assertRaises(ValueError):
            load_demand_data(StringIO(csv_data))

    def test_dataset_summary(self):
        csv_data = (
            "date,blood_group,demand_units\n"
            "2024-01-01,A+,10\n"
            "2024-01-02,A+,12\n"
            "2024-01-01,B+,6\n"
        )
        df = load_demand_data(StringIO(csv_data))
        summary = get_dataset_summary(df)
        self.assertEqual(summary["total_records"], 3)
        self.assertEqual(summary["total_demand_units"], 28)
        self.assertEqual(len(summary["groups_present"]), 2)

    def test_generate_sample_donor_registry(self):
        registry = generate_sample_donor_registry()
        self.assertGreater(len(registry), 10)
        self.assertIn("Eligibility Status", registry.columns)
        self.assertIn("Blood Group", registry.columns)

    def test_load_huggingface_facility_data(self):
        from blood_bank_ai.data import load_huggingface_facility_data
        demand, inv = load_huggingface_facility_data("District Hospital")
        self.assertGreater(len(demand), 1000)
        self.assertEqual(len(inv), 8)
        self.assertIn("blood_group", demand.columns)
        self.assertIn("demand_units", demand.columns)

    def test_generate_sample_hospital_requests(self):
        from blood_bank_ai.data import generate_sample_hospital_requests
        reqs = generate_sample_hospital_requests()
        self.assertGreaterEqual(len(reqs), 5)
        self.assertIn("Request ID", reqs.columns)
        self.assertIn("Urgency Level", reqs.columns)


if __name__ == "__main__":
    unittest.main()
