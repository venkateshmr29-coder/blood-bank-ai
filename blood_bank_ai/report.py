"""
Automated Clinical Executive Audit & Operational Report Generator.

Generates formal clinical audit reports, inventory safety analyses, and AI forecasting
benchmarks suitable for hospital medical boards, emergency management directors,
and academic review evaluations.
"""

from datetime import datetime
from typing import Any, Dict, Tuple
import pandas as pd


def calculate_facility_readiness_score(inventory_df: pd.DataFrame) -> Tuple[int, str]:
    """Computes a 0-100 composite preparedness index for the blood bank."""
    inv = inventory_df.set_index("blood_group")["available_units"].to_dict()
    thresh = inventory_df.set_index("blood_group")["low_stock_threshold"].to_dict()

    points = 100
    for bg, units in inv.items():
        th = thresh.get(bg, 10)
        if units == 0:
            points -= 15
        elif units < th:
            points -= 8
        elif units < th * 1.5:
            points -= 2

    score = max(20, min(100, points))
    if score >= 85:
        rating = "Tier 1 — High Emergency Readiness"
    elif score >= 65:
        rating = "Tier 2 — Moderate Buffer (Active Monitoring)"
    else:
        rating = "Tier 3 — Critical Surge Vulnerability"
    return score, rating


def generate_audit_report_markdown(
    inventory_df: pd.DataFrame,
    demand_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    key_forecasts: Dict[str, int],
    facility_name: str = "District Hospital",
) -> str:
    """
    Assembles a comprehensive clinical audit report in GitHub-Flavored Markdown.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    score, rating = calculate_facility_readiness_score(inventory_df)

    inv_dict = inventory_df.set_index("blood_group")["available_units"].to_dict()
    thresh_dict = inventory_df.set_index("blood_group")["low_stock_threshold"].to_dict()
    exp_dict = inventory_df.set_index("blood_group")["units_expiring_soon"].to_dict()
    total_units = sum(inv_dict.values())
    total_exp = sum(exp_dict.values())

    # Format inventory table
    inv_rows = []
    for bg in sorted(inv_dict.keys()):
        stk = inv_dict.get(bg, 0)
        th = thresh_dict.get(bg, 10)
        ex = exp_dict.get(bg, 0)
        status = "🚨 Deficit" if stk <= th else "✅ Normal"
        inv_rows.append(f"| **{bg}** | {stk} | {th} | {ex} | {status} |")
    inv_table_md = "\n".join(inv_rows)

    # Format benchmark table
    bench_rows = []
    if not metrics_df.empty:
        for _, r in metrics_df.iterrows():
            bench_rows.append(
                f"| {r.get('model', 'N/A')} | {r.get('MAE (units/day)', 'N/A')} | {r.get('RMSE (units/day)', 'N/A')} | {r.get('MAPE (%)', 'N/A')}% | {r.get('R² Score', 'N/A')} |"
            )
    bench_table_md = "\n".join(bench_rows)

    report_md = f"""# 🩸 Clinical Blood Bank Operational Audit & AI Forecasting Report

**Healthcare Facility:** {facility_name}  
**Audit Generated:** {timestamp}  
**Academic Alignment:** Vemana Institute of Technology — Department of CSE (Data Science)  
**Project Team:** Venkatesh M.R (1VI24CD045) • Tejas S (1VI24CD043) • Karthik S (1VI24CD018)  
**System Deployment:** [Live Web Application](https://venkateshmr29-coder-blood-bank-ai-app-98ipak.streamlit.app)

---

## 1. 📊 Executive Summary & Readiness Assessment

- **Overall Facility Readiness Score:** **{score}/100** (`{rating}`)
- **Total Blood Units In-Bank:** **{total_units} units**
- **Units Expiring in $\le 72$ Hours:** **{total_exp} units**
- **7-Day Projected Demand for Core Groups:**
  - **A+:** `{key_forecasts.get('A+', 'N/A')} units`
  - **B+:** `{key_forecasts.get('B+', 'N/A')} units`
  - **O- (Universal Donor):** `{key_forecasts.get('O-', 'N/A')} units`
  - **AB+ (Universal Recipient):** `{key_forecasts.get('AB+', 'N/A')} units`

---

## 2. 📋 Real-Time Inventory & Safety Threshold Matrix

| Blood Group | Available (Units) | Reorder Threshold | Expiring Soon (<3d) | Status |
| :--- | :---: | :---: | :---: | :---: |
{inv_table_md}

---

## 3. 🔮 AI Multi-Model Forecasting & Holdout Validation Benchmark

Evaluated on chronological holdout cross-validation:

| Regressor Architecture | Holdout MAE | Holdout RMSE | MAPE (%) | Coefficient of Determination ($R^2$) |
| :--- | :---: | :---: | :---: | :---: |
{bench_table_md}

---

## 4. 🏥 Clinical Directives & Risk Mitigation Checklist

1. **Universal Donor Preservation:** Prioritize type-specific cross-matching before dispensing universal O- units.
2. **First-Expired, First-Out (FEFO) Protocol:** Issue units marked for $\le 72$-hour expiry to scheduled elective surgical wards immediately.
3. **Emergency Regional Rebalancing:** Connect with inter-hospital supply chain network if any single group drops below 5 units.
4. **Donor Drive Recruitment:** Mobilize eligible registered donors who have surpassed the mandatory 90-day cooldown period.

---

### ✍️ Clinical Sign-Off Block
- **Supervising Transfusion Medicine Specialist:** __________________________________
- **Lead Data Science Researcher:** Venkatesh M.R (1VI24CD045)
- **Status:** **APPROVED FOR CLINICAL DECISION SUPPORT & OPERATIONAL SIMULATION**
"""
    return report_md.strip()
