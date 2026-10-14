"""
Data Ingestion, Schema Validation, and Dataset Profiling Module.

Handles loading, validating, and cleaning of historical demand records and
inventory snapshots from CSV files or uploaded file streams.
"""

from io import StringIO
from pathlib import Path
from typing import Any, Dict, Tuple, Union
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
DEMAND_FILE = DATA_DIR / "sample_demand_synthetic.csv"
INVENTORY_FILE = DATA_DIR / "sample_inventory_synthetic.csv"

# Hugging Face Real Multi-Facility Dataset Paths (electricsheepafrica/blood-bank-supply-management)
HF_DATASETS = {
    "District Hospital": {
        "demand": DATA_DIR / "hf_district_daily_demand.csv",
        "inventory": DATA_DIR / "hf_district_inventory.csv",
        "source_url": "https://huggingface.co/datasets/electricsheepafrica/blood-bank-supply-management/raw/main/data/blood_district_hospital_bb.csv",
    },
    "National Blood Centre": {
        "demand": DATA_DIR / "hf_national_daily_demand.csv",
        "inventory": DATA_DIR / "hf_national_inventory.csv",
        "source_url": "https://huggingface.co/datasets/electricsheepafrica/blood-bank-supply-management/raw/main/data/blood_national_blood_centre.csv",
    },
    "Rural Hospital": {
        "demand": DATA_DIR / "hf_rural_daily_demand.csv",
        "inventory": DATA_DIR / "hf_rural_inventory.csv",
        "source_url": "https://huggingface.co/datasets/electricsheepafrica/blood-bank-supply-management/raw/main/data/blood_rural_hospital_bb.csv",
    },
}


def load_demand_data(source: Union[str, Path, StringIO, bytes] = DEMAND_FILE) -> pd.DataFrame:
    """
    Loads and sanitizes daily blood demand data.
    Validates required columns, parses dates, removes nulls/negatives, and sorts.
    """
    if isinstance(source, bytes):
        df = pd.read_csv(StringIO(source.decode("utf-8")), parse_dates=["date"])
    elif isinstance(source, StringIO):
        df = pd.read_csv(source, parse_dates=["date"])
    else:
        df = pd.read_csv(source, parse_dates=["date"])

    required = {"date", "blood_group", "demand_units"}
    if not required.issubset(df.columns):
        raise ValueError(f"Demand data must contain columns: {sorted(required)}. Found: {list(df.columns)}")

    df = df.dropna(subset=["date", "blood_group", "demand_units"]).copy()
    df["blood_group"] = df["blood_group"].astype(str).str.strip().str.upper()
    df["demand_units"] = pd.to_numeric(df["demand_units"], errors="coerce")
    df = df.dropna(subset=["demand_units"])
    df = df[df["demand_units"] >= 0].sort_values(["blood_group", "date"])

    if df.empty:
        raise ValueError("Demand data contains no valid non-negative demand rows.")
    return df.reset_index(drop=True)


def load_inventory_data(source: Union[str, Path, StringIO, bytes] = INVENTORY_FILE) -> pd.DataFrame:
    """
    Loads and validates blood bank inventory status snapshot.
    """
    if isinstance(source, bytes):
        df = pd.read_csv(StringIO(source.decode("utf-8")))
    elif isinstance(source, StringIO):
        df = pd.read_csv(source)
    else:
        df = pd.read_csv(source)

    required = {
        "blood_group",
        "available_units",
        "low_stock_threshold",
        "units_expiring_soon",
    }
    if not required.issubset(df.columns):
        raise ValueError(f"Inventory data must contain columns: {sorted(required)}. Found: {list(df.columns)}")

    df["blood_group"] = df["blood_group"].astype(str).str.strip().str.upper()
    for col in ["available_units", "low_stock_threshold", "units_expiring_soon"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    return df.reset_index(drop=True)


def get_dataset_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Generates an academic statistical summary of the demand dataset."""
    min_date = df["date"].min().strftime("%Y-%m-%d")
    max_date = df["date"].max().strftime("%Y-%m-%d")
    total_days = (df["date"].max() - df["date"].min()).days + 1
    groups = sorted(df["blood_group"].unique())
    total_units = int(df["demand_units"].sum())
    avg_daily_all = round(float(df["demand_units"].mean()), 2)

    group_stats = []
    for g in groups:
        gdf = df[df["blood_group"] == g]
        group_stats.append(
            {
                "Blood Group": g,
                "Observations": len(gdf),
                "Total Demand": int(gdf["demand_units"].sum()),
                "Daily Mean": round(float(gdf["demand_units"].mean()), 2),
                "Std Dev": round(float(gdf["demand_units"].std()), 2),
                "Max Single-Day": int(gdf["demand_units"].max()),
            }
        )

    return {
        "start_date": min_date,
        "end_date": max_date,
        "total_days_span": total_days,
        "total_records": len(df),
        "total_demand_units": total_units,
        "overall_mean_daily_demand": avg_daily_all,
        "groups_present": groups,
        "group_stats_df": pd.DataFrame(group_stats),
    }


def generate_sample_donor_registry() -> pd.DataFrame:
    """Generates synthetic donor contact records for camp planning simulation."""
    import random
    random.seed(42)
    names = [
        "Aarav Sharma", "Priya Patel", "Rahul Verma", "Sneha Nair", "Vikram Reddy",
        "Ananya Rao", "Rohan Iyer", "Kavya Menon", "Aditya Kulkarni", "Neha Gupta",
        "Deepak Joshi", "Pooja Hegde", "Manoj Pillai", "Divya Deshmukh", "Suresh Kumar"
    ]
    groups = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]
    donors = []
    for i, name in enumerate(names, 1):
        bg = groups[i % len(groups)]
        last_donation_days = random.randint(30, 180)
        eligible = "Eligible" if last_donation_days >= 90 else "Cooldown"
        donors.append(
            {
                "Donor ID": f"DNR-{1000 + i}",
                "Name": name,
                "Blood Group": bg,
                "Days Since Last Donation": last_donation_days,
                "Eligibility Status": eligible,
                "Contact Status": "Verified",
            }
        )
    return pd.DataFrame(donors)


def load_huggingface_facility_data(facility_name: str = "District Hospital") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads daily demand and inventory snapshot for a chosen Hugging Face healthcare facility.
    Supports 'District Hospital', 'National Blood Centre', and 'Rural Hospital'.
    """
    if facility_name not in HF_DATASETS:
        facility_name = "District Hospital"

    info = HF_DATASETS[facility_name]
    demand_path = info["demand"]
    inventory_path = info["inventory"]

    if not demand_path.exists() or not inventory_path.exists():
        # Fallback to default files if pre-generated files are missing
        demand_path = DEMAND_FILE
        inventory_path = INVENTORY_FILE

    demand_df = load_demand_data(demand_path)
    inventory_df = load_inventory_data(inventory_path)
    return demand_df, inventory_df


def generate_sample_hospital_requests() -> pd.DataFrame:
    """
    Generates demonstration clinical hospital requests for the Hospital Request Portal.
    Implements the request tracking specified in Review 0 Slide 6 & 10.
    """
    return pd.DataFrame(
        [
            {
                "Request ID": "REQ-7801",
                "Hospital / Clinic": "City General Hospital",
                "Ward": "Emergency Trauma ICU",
                "Blood Group": "O-",
                "Units Requested": 6,
                "Urgency Level": "🚨 Emergency",
                "Requested Date": "2026-10-14 18:30",
                "Status": "Pending Review",
            },
            {
                "Request ID": "REQ-7802",
                "Hospital / Clinic": "Vemana Memorial Surgical Center",
                "Ward": "Operation Theatre 2",
                "Blood Group": "A+",
                "Units Requested": 4,
                "Urgency Level": "⚠️ Urgent",
                "Requested Date": "2026-10-14 17:15",
                "Status": "Approved",
            },
            {
                "Request ID": "REQ-7803",
                "Hospital / Clinic": "District Maternal Hospital",
                "Ward": "Maternity & Obstetrics",
                "Blood Group": "B+",
                "Units Requested": 3,
                "Urgency Level": "⚠️ Urgent",
                "Requested Date": "2026-10-14 16:40",
                "Status": "Dispatched",
            },
            {
                "Request ID": "REQ-7804",
                "Hospital / Clinic": "St. John Oncology Center",
                "Ward": "Hematology Daycare",
                "Blood Group": "AB+",
                "Units Requested": 2,
                "Urgency Level": "Routine",
                "Requested Date": "2026-10-14 14:00",
                "Status": "Fulfilled",
            },
            {
                "Request ID": "REQ-7805",
                "Hospital / Clinic": "Apex Children's Specialty",
                "Ward": "Pediatric Surgery",
                "Blood Group": "O+",
                "Units Requested": 5,
                "Urgency Level": "🚨 Emergency",
                "Requested Date": "2026-10-14 18:50",
                "Status": "Pending Review",
            },
        ]
    )

