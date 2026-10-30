"""
Disaster Surge & Mass-Casualty Incident (MCI) Triage Simulator.

Simulates acute disaster surges (multi-vehicle highway collisions, industrial explosions,
severe dengue outbreaks, code black incidents) to stress-test blood bank resilience,
evaluate universal donor exhaustion, and formulate emergency mobilization checklists.
"""

from typing import Any, Dict, List, Optional
import pandas as pd

from blood_bank_ai.compatibility import get_compatible_donors, simulate_emergency_allocation


DISASTER_SCENARIOS: Dict[str, Dict[str, Any]] = {
    "Level-1 Highway Multi-Vehicle Collision (Trauma Surge)": {
        "description": "Massive highway pileup involving multi-passenger buses. High acute hemorrhage demand.",
        "casualties": 22,
        "severity": "Level-1 Critical Trauma",
        "demand": {"O-": 8, "O+": 12, "A+": 8, "B+": 4, "A-": 2, "B-": 1, "AB+": 1, "AB-": 0},
    },
    "Regional Severe Dengue Outbreak (Thrombocytopenia Surge)": {
        "description": "Endemic seasonal outbreak causing acute thrombocytopenia and microvascular bleeding.",
        "casualties": 35,
        "severity": "Epidemic Inpatient Surge",
        "demand": {"O+": 10, "A+": 10, "B+": 12, "AB+": 4, "O-": 4, "A-": 2, "B-": 2, "AB-": 1},
    },
    "Obstetric Massive Hemorrhage Emergency (Code Stork)": {
        "description": "Clustered acute postpartum hemorrhage emergencies in regional maternity ICU.",
        "casualties": 6,
        "severity": "Acute Critical Obstetrics",
        "demand": {"O-": 10, "O+": 6, "A+": 4, "A-": 2, "B+": 2, "B-": 1, "AB+": 0, "AB-": 0},
    },
    "Industrial Plant Explosion (Code Black Incident)": {
        "description": "Severe blast trauma with complex polytrauma, penetrating wounds, and extensive burns.",
        "casualties": 48,
        "severity": "Code Black Disaster Surge",
        "demand": {"O-": 14, "O+": 18, "A+": 14, "B+": 10, "A-": 4, "B-": 3, "AB+": 3, "AB-": 2},
    },
}


def get_preset_disaster_scenarios() -> Dict[str, Dict[str, Any]]:
    """Returns catalog of standard disaster triage scenarios."""
    return DISASTER_SCENARIOS


def simulate_disaster_surge(
    inventory_df: pd.DataFrame,
    scenario_name: str,
    custom_demand: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """
    Stress-tests blood bank inventory against mass-casualty demand.
    Sequential allocation with biological cross-match substitution when
    exact blood groups are exhausted.
    """
    scenario_meta = DISASTER_SCENARIOS.get(
        scenario_name,
        {
            "description": "Custom Clinical Disaster Scenario",
            "casualties": 15,
            "severity": "Custom Trauma Surge",
            "demand": {"O-": 5, "O+": 5, "A+": 5, "B+": 5, "A-": 2, "B-": 2, "AB+": 2, "AB-": 1},
        },
    )

    demand_plan = custom_demand if custom_demand is not None else scenario_meta["demand"]

    inv_state = inventory_df.set_index("blood_group")["available_units"].to_dict()
    pre_stock_total = sum(inv_state.values())

    impact_records: List[Dict[str, Any]] = []
    total_requested = sum(demand_plan.values())
    total_allocated = 0
    substitutions_used: Dict[str, List[str]] = {}
    exhausted_groups: List[str] = []

    # Process each demanded blood group
    for bg, units_needed in demand_plan.items():
        if units_needed <= 0:
            continue

        # Simulate emergency allocation using compatibility module
        temp_inv_df = pd.DataFrame(
            [{"blood_group": k, "available_units": v} for k, v in inv_state.items()]
        )
        alloc_res = simulate_emergency_allocation(temp_inv_df, bg, units_needed)

        allocated_qty = int(alloc_res["units_allocated"])
        total_allocated += allocated_qty

        # Deduct from working inventory
        for donor_g, taken_u in alloc_res["allocation_plan"].items():
            inv_state[donor_g] = max(0, inv_state.get(donor_g, 0) - taken_u)

        if alloc_res["substitutions_used"]:
            substitutions_used[bg] = alloc_res["substitutions_used"]

    post_stock_total = sum(inv_state.values())
    shortfall = max(0, total_requested - total_allocated)

    # Compile detailed impact breakdown table
    orig_inv = inventory_df.set_index("blood_group")["available_units"].to_dict()
    for bg in sorted(orig_inv.keys()):
        pre = orig_inv.get(bg, 0)
        post = inv_state.get(bg, 0)
        dem = demand_plan.get(bg, 0)
        used = pre - post

        if post == 0 and pre > 0:
            status = "🚨 Completely Depleted"
            exhausted_groups.append(bg)
        elif post <= 5:
            status = "⚠️ Near Exhaustion"
        else:
            status = "✅ Stable Runway"

        impact_records.append(
            {
                "Blood Group": bg,
                "Pre-Surge Stock": pre,
                "Direct Demand": dem,
                "Units Consumed": used,
                "Post-Surge Balance": post,
                "Surge Status": status,
            }
        )

    # Emergency Action Directives based on simulation outcome
    checklist = [
        "Activate Code Black Emergency Response Protocol.",
        "Lock down blood bank dispensaries to authorized trauma resuscitation personnel only.",
        "Notify state regional blood coordination bureau for urgent inter-hospital transport.",
    ]
    if "O-" in exhausted_groups or inv_state.get("O-", 0) <= 2:
        checklist.append("CRITICAL: Universal O- supply exhausted! Enforce emergency Rh+ switching for non-reproductive age casualties.")
    if shortfall > 0:
        checklist.append(f"SHORTFALL ALERT: Immediate shortfall of {shortfall} units across critical groups. Mobilize urgent military/police donor corps.")
    else:
        checklist.append("All trauma casualties successfully sustained by existing inventory and biological substitutes.")

    # Calculate estimated runway in hours based on surge consumption
    burn_rate_per_hour = total_allocated / 4.0  # Assumes 4-hour acute surgical window
    hours_runway = round(post_stock_total / max(burn_rate_per_hour, 1.0), 1)

    return {
        "scenario_name": scenario_name,
        "scenario_meta": scenario_meta,
        "total_requested": total_requested,
        "total_allocated": total_allocated,
        "shortfall": shortfall,
        "pre_stock_total": pre_stock_total,
        "post_stock_total": post_stock_total,
        "exhausted_groups": exhausted_groups,
        "substitutions_used": substitutions_used,
        "impact_df": pd.DataFrame(impact_records),
        "emergency_checklist": checklist,
        "hours_of_runway": hours_runway,
    }
