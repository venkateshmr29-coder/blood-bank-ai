"""
Inventory Optimization and Risk Management Module for Blood Banks.

Implements operational research formulas for Safety Stock (SS),
Reorder Point (ROP), and Expiration Risk assessment for blood components.
"""

from typing import Dict, Any
import numpy as np
import pandas as pd


# Standard Z-scores for desired service levels
SERVICE_LEVEL_Z: Dict[float, float] = {
    0.90: 1.282,
    0.95: 1.645,
    0.98: 2.054,
    0.99: 2.326,
}

# Standard blood product shelf life in days
SHELF_LIFE_DAYS: Dict[str, int] = {
    "Whole Blood": 35,
    "Packed Red Blood Cells (PRBC)": 42,
    "Platelets": 5,
    "Fresh Frozen Plasma (FFP)": 365,
}


def compute_inventory_metrics(
    current_stock: int,
    history_demand: pd.Series,
    lead_time_days: int = 2,
    service_level: float = 0.95,
) -> Dict[str, Any]:
    """
    Computes Safety Stock (SS) and Reorder Point (ROP) using standard
    probabilistic inventory theory.

    Formula:
        Average Daily Demand = D_avg
        Demand Standard Deviation = sigma_D
        Lead Time = L (days)
        Safety Stock (SS) = Z * sigma_D * sqrt(L)
        Reorder Point (ROP) = (D_avg * L) + SS
    """
    if len(history_demand) == 0:
        return {
            "avg_daily_demand": 0.0,
            "std_daily_demand": 0.0,
            "safety_stock": 0,
            "reorder_point": 0,
            "current_stock": current_stock,
            "stock_status": "No data",
            "recommended_order": 0,
        }

    d_avg = float(history_demand.mean())
    sigma_d = float(history_demand.std(ddof=1)) if len(history_demand) > 1 else 0.0
    z = SERVICE_LEVEL_Z.get(service_level, 1.645)

    # Lead time demand + safety stock
    lead_time_demand = d_avg * lead_time_days
    safety_stock = int(np.ceil(z * sigma_d * np.sqrt(lead_time_days)))
    reorder_point = int(np.ceil(lead_time_demand + safety_stock))

    # Evaluate current stock status and risk category (Review 0 Slide 9 & 12)
    if current_stock <= 0:
        status = "STOCKOUT_CRITICAL"
        risk_level = "Critical"
        risk_badge = "🚨 Critical"
    elif current_stock < safety_stock:
        status = "CRITICAL_DEFICIT"
        risk_level = "Critical"
        risk_badge = "🚨 Critical"
    elif current_stock <= reorder_point:
        status = "REORDER_TRIGGERED"
        risk_level = "Low"
        risk_badge = "⚠️ Low"
    elif current_stock > reorder_point * 2.5:
        status = "OVERSTOCKED_WARNING"
        risk_level = "Normal"
        risk_badge = "✅ Normal (High)"
    else:
        status = "HEALTHY"
        risk_level = "Normal"
        risk_badge = "✅ Normal"

    # Recommended replenishment order to return to Target Max Stock
    # Target Max Stock = ROP + Safety Stock
    target_max = reorder_point + safety_stock
    recommended_order = max(0, target_max - current_stock) if current_stock <= reorder_point else 0

    return {
        "avg_daily_demand": round(d_avg, 2),
        "std_daily_demand": round(sigma_d, 2),
        "lead_time_days": lead_time_days,
        "lead_time_demand": round(lead_time_demand, 1),
        "service_level": service_level,
        "z_score": z,
        "safety_stock": safety_stock,
        "reorder_point": reorder_point,
        "current_stock": current_stock,
        "stock_status": status,
        "risk_level": risk_level,
        "risk_badge": risk_badge,
        "recommended_order": recommended_order,
        "days_of_supply": round(current_stock / d_avg, 1) if d_avg > 0 else 999.0,
    }


def generate_inventory_health_summary(
    inventory_df: pd.DataFrame,
    demand_df: pd.DataFrame,
    lead_time_days: int = 2,
    service_level: float = 0.95,
) -> pd.DataFrame:
    """
    Computes complete inventory health analysis across all blood groups.
    Includes Risk Category (Critical, Low, Normal) from Review 0 Slide 12.
    """
    results = []
    for _, row in inventory_df.iterrows():
        bg = row["blood_group"]
        stock = int(row["available_units"])
        expiring = int(row.get("units_expiring_soon", 0))
        group_demand = demand_df[demand_df["blood_group"] == bg]["demand_units"]
        
        metrics = compute_inventory_metrics(
            current_stock=stock,
            history_demand=group_demand,
            lead_time_days=lead_time_days,
            service_level=service_level,
        )
        
        results.append(
            {
                "Blood Group": bg,
                "Current Units": stock,
                "Avg Daily Demand": metrics["avg_daily_demand"],
                "Safety Stock": metrics["safety_stock"],
                "Reorder Point (ROP)": metrics["reorder_point"],
                "Days of Supply": metrics["days_of_supply"],
                "Expiring Soon": expiring,
                "Risk Category": metrics["risk_badge"],
                "Health Status": metrics["stock_status"],
                "Suggested Replenish": metrics["recommended_order"],
            }
        )
    return pd.DataFrame(results)
