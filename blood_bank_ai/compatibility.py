"""
Blood Compatibility and Emergency Allocation Module.

Provides standard ABO and Rh D compatibility rules for Red Blood Cells (RBC)
and Whole Blood, along with emergency shortage allocation fallbacks.
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd


# Red Blood Cell (RBC) / Whole Blood transfusion compatibility matrix
# Format: Recipient blood group -> List of acceptable donor blood groups in priority order
RBC_COMPATIBILITY: Dict[str, List[str]] = {
    "A+": ["A+", "A-", "O+", "O-"],
    "A-": ["A-", "O-"],
    "B+": ["B+", "B-", "O+", "O-"],
    "B-": ["B-", "O-"],
    "AB+": ["AB+", "AB-", "A+", "A-", "B+", "B-", "O+", "O-"],  # Universal recipient
    "AB-": ["AB-", "A-", "B-", "O-"],
    "O+": ["O+", "O-"],
    "O-": ["O-"],  # Universal donor
}

# Plasma transfusion compatibility matrix
PLASMA_COMPATIBILITY: Dict[str, List[str]] = {
    "O": ["O", "A", "B", "AB"],
    "A": ["A", "AB"],
    "B": ["B", "AB"],
    "AB": ["AB"],  # Universal plasma donor
}


def get_compatible_donors(recipient_group: str) -> List[str]:
    """Return list of blood groups that can safely donate RBCs to recipient_group."""
    return RBC_COMPATIBILITY.get(recipient_group, [])


def get_compatible_recipients(donor_group: str) -> List[str]:
    """Return list of blood groups that can safely receive RBCs from donor_group."""
    recipients = []
    for rec, donors in RBC_COMPATIBILITY.items():
        if donor_group in donors:
            recipients.append(rec)
    return recipients


def get_compatibility_matrix_df() -> pd.DataFrame:
    """Generate a grid dataframe of donor vs recipient compatibility."""
    groups = ["O-", "O+", "B-", "B+", "A-", "A+", "AB-", "AB+"]
    matrix = {}
    for recipient in groups:
        row = {}
        compatible_donors = set(get_compatible_donors(recipient))
        for donor in groups:
            row[donor] = "✅ Match" if donor in compatible_donors else "❌"
        matrix[recipient] = row
    df = pd.DataFrame(matrix).T
    df.index.name = "Recipient ↓ / Donor →"
    return df


def simulate_emergency_allocation(
    inventory_df: pd.DataFrame,
    requested_group: str,
    units_needed: int,
) -> Dict[str, object]:
    """
    Simulates allocation for an emergency demand request.
    If the requested group does not have enough units, falls back to
    medically compatible donor groups in order of optimal conservation.
    """
    inv = inventory_df.set_index("blood_group")["available_units"].to_dict()
    donors = get_compatible_donors(requested_group)
    
    allocated: Dict[str, int] = {}
    remaining_needed = units_needed

    for donor in donors:
        if remaining_needed <= 0:
            break
        available = inv.get(donor, 0)
        if available > 0:
            take = min(available, remaining_needed)
            allocated[donor] = take
            remaining_needed -= take

    success = remaining_needed == 0
    total_allocated = units_needed - remaining_needed

    return {
        "requested_group": requested_group,
        "units_needed": units_needed,
        "units_allocated": total_allocated,
        "shortfall": remaining_needed,
        "success": success,
        "allocation_plan": allocated,
        "substitutions_used": [g for g in allocated if g != requested_group],
    }
