"""
Regional Healthcare Network Blood Supply & Inter-Hospital Transfer Optimizer.

Simulates and coordinates blood stock rebalancing across multiple healthcare facility tiers
(National Blood Centre, District Hospital, and Rural Hospital) to avert localized stockouts
and minimize wastage through automated inter-facility logistics.
"""

from typing import Any, Dict, List, Tuple
import pandas as pd

from blood_bank_ai.data import load_huggingface_facility_data


FACILITY_TIERS = ["District Hospital", "National Blood Centre", "Rural Hospital"]

# Transport matrix between facilities (transit time in minutes and distance in km)
TRANSIT_MATRIX: Dict[Tuple[str, str], Dict[str, Any]] = {
    ("National Blood Centre", "District Hospital"): {"time_min": 45, "dist_km": 28, "vehicle": "Dedicated Blood Courier Van"},
    ("National Blood Centre", "Rural Hospital"): {"time_min": 90, "dist_km": 65, "vehicle": "Cold-Chain Transport Unit"},
    ("District Hospital", "Rural Hospital"): {"time_min": 50, "dist_km": 35, "vehicle": "Emergency Ambulance Transfer"},
    ("District Hospital", "National Blood Centre"): {"time_min": 45, "dist_km": 28, "vehicle": "Dedicated Blood Courier Van"},
    ("Rural Hospital", "District Hospital"): {"time_min": 50, "dist_km": 35, "vehicle": "Emergency Ambulance Transfer"},
    ("Rural Hospital", "National Blood Centre"): {"time_min": 90, "dist_km": 65, "vehicle": "Cold-Chain Transport Unit"},
}


def load_all_facility_snapshots() -> Dict[str, pd.DataFrame]:
    """Loads inventory snapshots for all facility tiers from the Hugging Face dataset."""
    snapshots = {}
    for fac in FACILITY_TIERS:
        try:
            _, inv = load_huggingface_facility_data(fac)
            snapshots[fac] = inv.copy()
        except Exception:
            # Fallback synthetic frame if offline or missing
            snapshots[fac] = pd.DataFrame(
                [
                    {"blood_group": bg, "available_units": 20, "low_stock_threshold": 10, "units_expiring_soon": 1}
                    for bg in ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]
                ]
            )
    return snapshots


def generate_facility_comparison_matrix(inventories: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Creates a cross-facility comparison table showing available stock,
    safety threshold, and status for each blood group side by side.
    """
    blood_groups = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]
    rows = []

    for bg in blood_groups:
        row = {"Blood Group": bg}
        total_net = 0
        for fac_name, df in inventories.items():
            match = df[df["blood_group"] == bg]
            if not match.empty:
                units = int(match.iloc[0]["available_units"])
                thresh = int(match.iloc[0]["low_stock_threshold"])
                row[f"{fac_name} (Stock)"] = units
                row[f"{fac_name} (Thresh)"] = thresh
                total_net += units
            else:
                row[f"{fac_name} (Stock)"] = 0
                row[f"{fac_name} (Thresh)"] = 10
        row["Regional Total Stock"] = total_net
        rows.append(row)

    return pd.DataFrame(rows)


def optimize_inter_hospital_transfers(
    inventories: Dict[str, pd.DataFrame],
    safety_buffer: int = 12,
) -> Dict[str, Any]:
    """
    Computes optimal inter-facility blood transfers.
    Identifies facilities with surplus (available > threshold + safety_buffer)
    and routes units to facilities facing critical deficits (available <= threshold).
    """
    blood_groups = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]
    transfers: List[Dict[str, Any]] = []

    # Map stock status by facility and blood group
    facility_stocks: Dict[str, Dict[str, int]] = {}
    facility_thresholds: Dict[str, Dict[str, int]] = {}

    for fac, df in inventories.items():
        facility_stocks[fac] = df.set_index("blood_group")["available_units"].to_dict()
        facility_thresholds[fac] = df.set_index("blood_group")["low_stock_threshold"].to_dict()

    for bg in blood_groups:
        deficits: List[Tuple[str, int]] = []
        surpluses: List[Tuple[str, int]] = []

        for fac in FACILITY_TIERS:
            stock = facility_stocks.get(fac, {}).get(bg, 0)
            thresh = facility_thresholds.get(fac, {}).get(bg, 10)

            if stock < thresh:
                deficits.append((fac, thresh - stock))
            elif stock > thresh + safety_buffer:
                excess = stock - (thresh + safety_buffer)
                surpluses.append((fac, excess))

        # Sort deficits descending (most severe first) and surpluses descending (largest buffer first)
        deficits.sort(key=lambda x: x[1], reverse=True)
        surpluses.sort(key=lambda x: x[1], reverse=True)

        for dest_fac, needed in deficits:
            for i, (src_fac, available_excess) in enumerate(surpluses):
                if available_excess <= 0 or needed <= 0 or src_fac == dest_fac:
                    continue

                transfer_qty = min(needed, available_excess)
                surpluses[i] = (src_fac, available_excess - transfer_qty)
                needed -= transfer_qty

                transit_info = TRANSIT_MATRIX.get(
                    (src_fac, dest_fac),
                    {"time_min": 60, "dist_km": 40, "vehicle": "Standard Courier"},
                )

                urgency = "🚨 Critical Transfer" if facility_stocks[dest_fac][bg] <= 3 else "⚠️ Preventive Rebalance"
                benefit = f"Eliminates stockout at {dest_fac}; restores buffer by +{transfer_qty} units"

                transfers.append(
                    {
                        "Transfer ID": f"TRF-{len(transfers) + 1:03d}",
                        "Blood Group": bg,
                        "Origin Facility": src_fac,
                        "Destination Facility": dest_fac,
                        "Units Dispatched": transfer_qty,
                        "Urgency Level": urgency,
                        "Transit Time": f"{transit_info['time_min']} mins ({transit_info['dist_km']} km)",
                        "Cold-Chain Protocol": "2°C to 6°C Insulated Courier Box",
                        "Clinical Impact": benefit,
                    }
                )

    transfers_df = pd.DataFrame(transfers) if transfers else pd.DataFrame(
        columns=[
            "Transfer ID",
            "Blood Group",
            "Origin Facility",
            "Destination Facility",
            "Units Dispatched",
            "Urgency Level",
            "Transit Time",
            "Cold-Chain Protocol",
            "Clinical Impact",
        ]
    )

    total_transferred = int(transfers_df["Units Dispatched"].sum()) if not transfers_df.empty else 0
    stockouts_prevented = len(transfers_df[transfers_df["Urgency Level"].str.contains("Critical")]) if not transfers_df.empty else 0

    return {
        "transfers_df": transfers_df,
        "total_transfers_count": len(transfers_df),
        "total_units_rebalanced": total_transferred,
        "critical_stockouts_prevented": stockouts_prevented,
        "network_status": "Rebalanced & Optimized" if total_transferred > 0 else "Network Balanced (No Transfers Required)",
    }


def execute_network_transfer(
    inventories: Dict[str, pd.DataFrame],
    source_fac: str,
    dest_fac: str,
    blood_group: str,
    units: int,
) -> Dict[str, pd.DataFrame]:
    """
    Applies an inter-facility transfer by deducting units from source
    and adding units to destination facility inventory.
    """
    updated = {k: v.copy() for k, v in inventories.items()}

    if source_fac in updated and dest_fac in updated:
        # Deduct from source
        src_df = updated[source_fac]
        idx_src = src_df[src_df["blood_group"] == blood_group].index
        if not idx_src.empty:
            cur_src = int(src_df.loc[idx_src[0], "available_units"])
            new_src = max(0, cur_src - units)
            src_df.loc[idx_src[0], "available_units"] = new_src

        # Add to destination
        dest_df = updated[dest_fac]
        idx_dest = dest_df[dest_df["blood_group"] == blood_group].index
        if not idx_dest.empty:
            cur_dest = int(dest_df.loc[idx_dest[0], "available_units"])
            dest_df.loc[idx_dest[0], "available_units"] = cur_dest + units

    return updated
