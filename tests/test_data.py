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

    def test_insert_or_update_demand_record(self):
        from blood_bank_ai.data import insert_or_update_demand_record
        df = pd.DataFrame([{"date": pd.Timestamp("2026-01-01"), "blood_group": "A+", "demand_units": 10.0}])
        # Insert new date
        df2 = insert_or_update_demand_record(df, "2026-01-02", "A+", 15.0)
        self.assertEqual(len(df2), 2)
        # Update existing date (overwrite mode)
        df3 = insert_or_update_demand_record(df2, "2026-01-02", "A+", 25.0, mode="overwrite")
        self.assertEqual(len(df3), 2)
        self.assertEqual(float(df3[df3["date"] == pd.Timestamp("2026-01-02")]["demand_units"].iloc[0]), 25.0)
        # Update existing date (add mode)
        df4 = insert_or_update_demand_record(df3, "2026-01-02", "A+", 5.0, mode="add")
        self.assertEqual(float(df4[df4["date"] == pd.Timestamp("2026-01-02")]["demand_units"].iloc[0]), 30.0)

    def test_update_inventory_units(self):
        from blood_bank_ai.data import update_inventory_units
        inv = pd.DataFrame([{"blood_group": "A+", "available_units": 10, "low_stock_threshold": 5, "units_expiring_soon": 1}])
        updated = update_inventory_units(inv, "A+", available_units=20, low_stock_threshold=8)
        self.assertEqual(int(updated[updated["blood_group"] == "A+"]["available_units"].iloc[0]), 20)
        self.assertEqual(int(updated[updated["blood_group"] == "A+"]["low_stock_threshold"].iloc[0]), 8)

    def test_add_hospital_request(self):
        from blood_bank_ai.data import generate_sample_hospital_requests, add_hospital_request
        reqs = generate_sample_hospital_requests()
        initial_len = len(reqs)
        updated, req_id = add_hospital_request(reqs, "Metro Hospital", "O-", 5, "🚨 Emergency")
        self.assertEqual(len(updated), initial_len + 1)
        self.assertTrue(req_id.startswith("REQ-"))
        self.assertEqual(updated.iloc[0]["Blood Group"], "O-")

    def test_add_donor_to_registry(self):
        from blood_bank_ai.data import generate_sample_donor_registry, add_donor_to_registry
        donors = generate_sample_donor_registry()
        initial_len = len(donors)
        updated, donor_id = add_donor_to_registry(donors, "John Doe", "B+", days_since_last_donation=100)
        self.assertEqual(len(updated), initial_len + 1)
        self.assertEqual(updated.iloc[0]["Eligibility Status"], "Eligible")


    def test_insert_batch_demand_records(self):
        from blood_bank_ai.data import insert_batch_demand_records
        df = pd.DataFrame([{"date": pd.Timestamp("2026-01-01"), "blood_group": "A+", "demand_units": 10.0}])
        updated = insert_batch_demand_records(
            df,
            entry_date="2026-01-02",
            records_dict={"A+": 15.0, "B+": 12.0, "O-": 8.0},
            mode="overwrite",
        )
        self.assertEqual(len(updated), 4)
        self.assertEqual(float(updated[updated["blood_group"] == "O-"]["demand_units"].iloc[0]), 8.0)

    def test_parse_pasted_demand_text(self):
        from blood_bank_ai.data import parse_pasted_demand_text
        text = """
        2026-10-29, A+, 15
        2026-10-29 | B+ | 12
        2026-10-29\tO-\t8
        """
        parsed = parse_pasted_demand_text(text)
        self.assertEqual(len(parsed), 3)
        self.assertListEqual(list(parsed["blood_group"]), ["A+", "B+", "O-"])
        self.assertListEqual(list(parsed["demand_units"]), [15.0, 12.0, 8.0])


if __name__ == "__main__":
    unittest.main()
