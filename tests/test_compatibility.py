"""Unit tests for the blood compatibility and emergency allocation module."""

import unittest
import pandas as pd
from blood_bank_ai.compatibility import (
    get_compatibility_matrix_df,
    get_compatible_donors,
    get_compatible_recipients,
    simulate_emergency_allocation,
)


class CompatibilityTests(unittest.TestCase):
    def test_universal_donor_and_recipient(self):
        # O- can donate to all 8 blood groups
        o_neg_recipients = get_compatible_recipients("O-")
        self.assertEqual(len(o_neg_recipients), 8)

        # AB+ can receive from all 8 blood groups
        ab_pos_donors = get_compatible_donors("AB+")
        self.assertEqual(len(ab_pos_donors), 8)

        # O- can only receive from O-
        o_neg_donors = get_compatible_donors("O-")
        self.assertEqual(o_neg_donors, ["O-"])

    def test_compatibility_matrix_df(self):
        matrix = get_compatibility_matrix_df()
        self.assertEqual(matrix.shape, (8, 8))
        self.assertEqual(matrix.loc["AB+", "O-"], "✅ Match")
        self.assertEqual(matrix.loc["O-", "A+"], "❌")

    def test_emergency_allocation_simulation_exact_match(self):
        inv = pd.DataFrame(
            [
                {"blood_group": "A+", "available_units": 10},
                {"blood_group": "O+", "available_units": 5},
                {"blood_group": "O-", "available_units": 2},
            ]
        )
        res = simulate_emergency_allocation(inv, requested_group="A+", units_needed=8)
        self.assertTrue(res["success"])
        self.assertEqual(res["units_allocated"], 8)
        self.assertEqual(res["allocation_plan"].get("A+"), 8)
        self.assertEqual(len(res["substitutions_used"]), 0)

    def test_emergency_allocation_simulation_fallback_substitution(self):
        inv = pd.DataFrame(
            [
                {"blood_group": "A+", "available_units": 3},
                {"blood_group": "A-", "available_units": 4},
                {"blood_group": "O+", "available_units": 5},
                {"blood_group": "O-", "available_units": 2},
            ]
        )
        # Needs 8 units of A+, has only 3 A+, falls back to compatible donors
        res = simulate_emergency_allocation(inv, requested_group="A+", units_needed=8)
        self.assertTrue(res["success"])
        self.assertEqual(res["units_allocated"], 8)
        self.assertEqual(res["allocation_plan"].get("A+"), 3)
        self.assertGreater(len(res["substitutions_used"]), 0)


if __name__ == "__main__":
    unittest.main()
