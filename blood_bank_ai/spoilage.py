"""
Blood Component Shelf-Life & FEFO (First-Expired, First-Out) Spoilage Mitigation Engine.

Models component-level perishability dynamics (Platelets 5d, PRBC 42d, Whole Blood 35d, FFP 365d),
evaluates outdating risk velocity against forecasted demand, and optimizes FEFO dispatch queues
to minimize biological discard waste and healthcare costs.
"""

from typing import Any, Dict, List
import numpy as np
import pandas as pd

from blood_bank_ai.inventory import SHELF_LIFE_DAYS


COMPONENT_VALUATION = {
    "Packed Red Blood Cells (PRBC)": {"cost_inr": 1600, "shelf_days": 42, "storage": "2°C to 6°C"},
    "Platelets": {"cost_inr": 1800, "shelf_days": 5, "storage": "20°C to 24°C (Agitated)"},
    "Whole Blood": {"cost_inr": 1400, "shelf_days": 35, "storage": "2°C to 6°C"},
    "Fresh Frozen Plasma (FFP)": {"cost_inr": 1200, "shelf_days": 365, "storage": "≤ -18°C"},
}


def generate_component_batch_breakdown(inventory_df: pd.DataFrame) -> pd.DataFrame:
    """
    Simulates a granular batch age distribution across all blood groups in the bank.
    Partitions total units into 4 distinct age cohorts for operational dispatch:
    - Critical Near-Expiry (<= 3 days remaining)
    - Aging Reserve (4 - 10 days remaining)
    - Prime Middle-Age (11 - 25 days remaining)
    - Fresh Inflow (> 25 days remaining)
    """
    batches = []

    for _, row in inventory_df.iterrows():
        bg = row["blood_group"]
        avail = int(row["available_units"])
        exp_soon = int(row["units_expiring_soon"])

        # Distribute available units across realistic age cohorts
        crit_units = min(avail, max(0, exp_soon))
        remaining = max(0, avail - crit_units)

        aging_units = int(round(remaining * 0.35))
        mid_units = int(round(remaining * 0.45))
        fresh_units = max(0, remaining - aging_units - mid_units)

        batches.append(
            {
                "Blood Group": bg,
                "Total Available (Units)": avail,
                "Critical (<3 Days)": crit_units,
                "Aging (4-10 Days)": aging_units,
                "Prime (11-25 Days)": mid_units,
                "Fresh (>25 Days)": fresh_units,
                "Imminent Waste Risk (%)": round((crit_units / avail * 100.0) if avail > 0 else 0.0, 1),
            }
        )

    return pd.DataFrame(batches)


def compute_spoilage_risk_matrix(
    inventory_df: pd.DataFrame, demand_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Evaluates outdating velocity by comparing units expiring within 3 days against
    expected 3-day demand for each blood group.
    Flags Outdating Risk: 'Severe Discard Alert', 'Moderate Watch', or 'Healthy Turnover'.
    """
    records = []

    for _, row in inventory_df.iterrows():
        bg = row["blood_group"]
        avail = int(row["available_units"])
        expiring = int(row["units_expiring_soon"])

        bg_demand = demand_df[demand_df["blood_group"] == bg]["demand_units"]
        daily_mean = float(bg_demand.mean()) if not bg_demand.empty else 5.0
        expected_3d_consumption = daily_mean * 3.0

        # Discard risk occurs if units expiring exceed anticipated 3-day clinical demand
        projected_waste = max(0, int(round(expiring - expected_3d_consumption)))
        waste_cost = projected_waste * COMPONENT_VALUATION["Packed Red Blood Cells (PRBC)"]["cost_inr"]

        if projected_waste > 2:
            status = "🚨 High Discard Alert"
            rec_action = f"Auto-transfer {projected_waste} units to Regional Trauma Center immediately"
        elif expiring > 0:
            status = "⚠️ Moderate Watch"
            rec_action = "Prioritize for incoming elective surgical orders under FEFO protocol"
        else:
            status = "✅ Healthy Turnover"
            rec_action = "Normal FIFO inventory rotation"

        records.append(
            {
                "Blood Group": bg,
                "On-Hand Stock": avail,
                "Expiring (<3d)": expiring,
                "Expected 3d Demand": round(expected_3d_consumption, 1),
                "Projected Waste (Units)": projected_waste,
                "Financial Risk (₹)": waste_cost,
                "Turnover Health": status,
                "Prescriptive Action": rec_action,
            }
        )

    return pd.DataFrame(records)


def generate_fefo_dispatch_queue(
    batch_df: pd.DataFrame, requested_group: str, units_needed: int
) -> Dict[str, Any]:
    """
    Implements First-Expired, First-Out (FEFO) order allocation.
    Drains from oldest viable batches before touching fresher stock.
    """
    bg_row = batch_df[batch_df["Blood Group"] == requested_group]
    if bg_row.empty:
        return {"fulfilled": False, "allocated_units": 0, "cohort_breakdown": {}, "message": "Group not found"}

    row = bg_row.iloc[0]
    total_avail = int(row["Total Available (Units)"])
    crit = int(row["Critical (<3 Days)"])
    aging = int(row["Aging (4-10 Days)"])
    prime = int(row["Prime (11-25 Days)"])
    fresh = int(row["Fresh (>25 Days)"])

    alloc = {"Critical (<3 Days)": 0, "Aging (4-10 Days)": 0, "Prime (11-25 Days)": 0, "Fresh (>25 Days)": 0}
    remaining = min(units_needed, total_avail)
    actual_allocated = remaining

    # 1. Take Critical first
    take_crit = min(remaining, crit)
    alloc["Critical (<3 Days)"] = take_crit
    remaining -= take_crit

    # 2. Take Aging next
    take_aging = min(remaining, aging)
    alloc["Aging (4-10 Days)"] = take_aging
    remaining -= take_aging

    # 3. Take Prime
    take_prime = min(remaining, prime)
    alloc["Prime (11-25 Days)"] = take_prime
    remaining -= take_prime

    # 4. Take Fresh
    take_fresh = min(remaining, fresh)
    alloc["Fresh (>25 Days)"] = take_fresh
    remaining -= take_fresh

    waste_prevented = take_crit
    savings_inr = waste_prevented * COMPONENT_VALUATION["Packed Red Blood Cells (PRBC)"]["cost_inr"]

    return {
        "fulfilled": actual_allocated >= units_needed,
        "units_needed": units_needed,
        "units_allocated": actual_allocated,
        "shortfall": max(0, units_needed - actual_allocated),
        "cohort_breakdown": alloc,
        "waste_prevented_units": waste_prevented,
        "savings_inr": savings_inr,
        "message": f"Successfully allocated {actual_allocated} units via FEFO (prevented outdating of {waste_prevented} near-expiry units).",
    }
