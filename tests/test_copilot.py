import unittest
import pandas as pd
from blood_bank_ai.copilot import ask_blood_bank_copilot, PRESET_QUICK_QUESTIONS


class CopilotTests(unittest.TestCase):
    def setUp(self):
        self.inventory_df = pd.DataFrame(
            [
                {"blood_group": "A+", "available_units": 25, "low_stock_threshold": 10, "units_expiring_soon": 2},
                {"blood_group": "O-", "available_units": 4, "low_stock_threshold": 10, "units_expiring_soon": 1},
                {"blood_group": "B+", "available_units": 15, "low_stock_threshold": 8, "units_expiring_soon": 0},
            ]
        )
        self.demand_df = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=30, freq="D"),
                "blood_group": ["A+"] * 30,
                "demand_units": [10 + (i % 5) for i in range(30)],
            }
        )
        self.donor_df = pd.DataFrame(
            [
                {"Donor ID": "DON-001", "Full Name": "Test Donor", "Blood Group": "O-", "Eligibility Status": "Eligible"},
                {"Donor ID": "DON-002", "Full Name": "Test Donor 2", "Blood Group": "A+", "Eligibility Status": "Cooldown Active"},
            ]
        )
        self.hospital_reqs_df = pd.DataFrame(
            [
                {"Request ID": "REQ-001", "Hospital": "City Trauma", "Blood Group": "O-", "Units": 3, "Status": "Fulfilled & Dispatched"},
                {"Request ID": "REQ-002", "Hospital": "Rural Clinic", "Blood Group": "A+", "Units": 2, "Status": "Pending Review"},
            ]
        )

    def test_shortage_triage_detects_low_groups(self):
        res = ask_blood_bank_copilot(
            query="Which blood groups are at critical shortage risk?",
            inventory_df=self.inventory_df,
            demand_df=self.demand_df,
            donor_df=self.donor_df,
            hospital_reqs_df=self.hospital_reqs_df,
        )
        self.assertEqual(res["category"], "Risk Management")
        self.assertIn("O-", res["answer_markdown"])
        self.assertTrue(len(res["action_items"]) > 0)

    def test_emergency_crossmatch_advisory(self):
        res = ask_blood_bank_copilot(
            query="Emergency request for 10 units of O-",
            inventory_df=self.inventory_df,
            demand_df=self.demand_df,
            donor_df=self.donor_df,
            hospital_reqs_df=self.hospital_reqs_df,
            selected_bg="O-",
        )
        self.assertEqual(res["category"], "Immunohematology")
        self.assertIn("Universal", res["answer_markdown"])

    def test_donor_mobilization_generates_sms(self):
        res = ask_blood_bank_copilot(
            query="Generate donor mobilization campaign and SMS alert",
            inventory_df=self.inventory_df,
            demand_df=self.demand_df,
            donor_df=self.donor_df,
            hospital_reqs_df=self.hospital_reqs_df,
        )
        self.assertEqual(res["category"], "Operations")
        self.assertIn("SMS", res["answer_markdown"])

    def test_preset_questions_exist(self):
        self.assertGreater(len(PRESET_QUICK_QUESTIONS), 3)


if __name__ == "__main__":
    unittest.main()
