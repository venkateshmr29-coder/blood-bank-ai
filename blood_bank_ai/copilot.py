"""
AI Clinical Decision Support Copilot for Blood Bank Operations.

Provides interactive operational intelligence, emergency cross-matching guidance,
donor recall mobilization strategy, stockout risk triage, and executive shift briefings.
Works 100% offline with zero external API dependencies.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from blood_bank_ai.compatibility import RBC_COMPATIBILITY, get_compatible_donors
from blood_bank_ai.inventory import SHELF_LIFE_DAYS, compute_inventory_metrics


PRESET_QUICK_QUESTIONS: List[str] = [
    "🚨 Which blood groups are at critical stockout risk right now?",
    "🩸 How should we handle an emergency trauma request for 10 units of O-?",
    "📢 Generate an urgent donor mobilization alert & SMS template",
    "⏳ How can we minimize platelet and PRBC expiration wastage?",
    "📋 Generate morning executive rounds briefing & shift handover",
    "🏥 What is the current hospital request fulfillment rate?",
]


def ask_blood_bank_copilot(
    query: str,
    inventory_df: pd.DataFrame,
    demand_df: pd.DataFrame,
    donor_df: pd.DataFrame,
    hospital_reqs_df: pd.DataFrame,
    selected_bg: str = "A+",
    lead_time: int = 2,
    service_level: float = 0.95,
) -> Dict[str, Any]:
    """
    Synthesizes real-time blood bank metrics with clinical immunohematology rules
    to deliver actionable clinical decision support.
    """
    q = query.lower().strip()

    # Pre-calculate inventory summary
    inv_dict = inventory_df.set_index("blood_group")["available_units"].to_dict()
    thresh_dict = inventory_df.set_index("blood_group")["low_stock_threshold"].to_dict()
    exp_dict = inventory_df.set_index("blood_group")["units_expiring_soon"].to_dict()
    total_stock = sum(inv_dict.values())
    total_expiring = sum(exp_dict.values())

    critical_groups = [
        bg for bg, units in inv_dict.items() if units <= thresh_dict.get(bg, 10) or units == 0
    ]

    # Pre-calculate donor stats
    eligible_donors = donor_df[donor_df["Eligibility Status"] == "Eligible"] if not donor_df.empty else pd.DataFrame()
    total_donors = len(donor_df)
    eligible_count = len(eligible_donors)

    # Pre-calculate hospital request stats
    pending_reqs = (
        hospital_reqs_df[hospital_reqs_df["Status"].str.contains("Pending|Shortage", case=False, na=False)]
        if not hospital_reqs_df.empty
        else pd.DataFrame()
    )
    fulfilled_reqs = (
        hospital_reqs_df[hospital_reqs_df["Status"].str.contains("Fulfilled", case=False, na=False)]
        if not hospital_reqs_df.empty
        else pd.DataFrame()
    )

    # 1. CRITICAL DEFICIT & SHORTAGE TRIAGE
    if any(k in q for k in ["shortage", "deficit", "critical", "risk", "stockout", "out of stock", "which blood"]):
        if critical_groups:
            crit_md = "\n".join(
                [
                    f"- **{bg}**: `{inv_dict.get(bg, 0)} units` available (Threshold: `{thresh_dict.get(bg, 10)} units`, Deficit: `{max(0, thresh_dict.get(bg, 10) - inv_dict.get(bg, 0))} units`)"
                    for bg in critical_groups
                ]
            )
            ans = f"""
### 🚨 Critical Inventory Stockout Triage Report

Our real-time diagnostic scan identifies **{len(critical_groups)} blood groups** currently below safety reorder thresholds:

{crit_md}

#### 🎯 Root Cause & Operational Assessment:
- **Immediate Clinical Hazard:** Depletion of O- or A+ restricts emergency trauma resuscitations and scheduled cardiovascular procedures.
- **Recommended Actions:**
  1. Trigger immediate **Priority 1 Donor Mobilization** for: `{' • '.join(critical_groups)}`.
  2. Reserve remaining O- stocks strictly for neonatal resuscitation and Rh-negative women of childbearing potential.
  3. Notify regional trauma network to initiate inter-hospital emergency rebalancing.
"""
            actions = [
                f"Mobilize targeted blood drives for {', '.join(critical_groups)} immediately.",
                "Review pending non-urgent surgeries requiring these groups for potential 24-hour deferral.",
                "Verify automated cross-match substitutions with attending hematologist.",
            ]
        else:
            ans = f"""
### ✅ All Blood Groups Operating Within Safe Buffers

- **Total Bank Units:** `{total_stock} units`
- **Zero Critical Deficits Detected:** All 8 blood groups exceed their minimum safety reorder thresholds.
- **Buffer Stability:** Standard operations may proceed for all elective surgical wards and emergency departments.
"""
            actions = [
                "Maintain standard replenishment donor intake.",
                "Monitor FIFO/FEFO expiration rotation for platelet and PRBC stocks.",
            ]

        return {
            "title": "Clinical Stockout Triage",
            "category": "Risk Management",
            "answer_markdown": ans.strip(),
            "action_items": actions,
            "key_metrics": {
                "Total Units in Bank": f"{total_stock} units",
                "Groups in Critical Deficit": f"{len(critical_groups)} groups",
                "Expiring Soon (<3d)": f"{total_expiring} units",
            },
        }

    # 2. EMERGENCY CROSS-MATCHING & TRAUMA ALLOCATION
    if any(k in q for k in ["crossmatch", "substitut", "trauma", "emergency", "o-", "universal", "transfusion", "10 units"]):
        o_neg_stock = inv_dict.get("O-", 0)
        o_pos_stock = inv_dict.get("O+", 0)
        a_pos_stock = inv_dict.get("A+", 0)

        compat_o_neg = get_compatible_donors("O-")
        ans = f"""
### 🩸 Emergency Trauma Surge & Cross-Match Protocol

**Acute Request Scenario:** Emergency Room demands immediate uncrossmatched or compatible RBCs.

#### 📊 Current Available Universal & Emergency Reserve:
- **O- (Universal RBC Donor):** `{o_neg_stock} units` available in blood bank
- **O+ (Alternative for Rh+ adult males / non-childbearing females):** `{o_pos_stock} units` available
- **A+ (Common emergency recipient buffer):** `{a_pos_stock} units` available

#### 🧬 Immunohematology Protocol & Conservation Rules:
1. **O- Negative Conservation Rule:**
   - O- units MUST be conserved strictly for:
     - Patients with confirmed O- blood group
     - Female trauma patients of reproductive age ($\le 50$ years) whose Rh status is unknown
     - Pediatric / neonatal transfusions
2. **Rh D Switching in Severe Trauma (Code Black):**
   - For adult male trauma victims or females $>50$ years with unknown blood type, administer **O+ PRBC** to prevent depleting scarce O- stocks.
3. **Biological Compatibility Matrix for Selected Group ({selected_bg}):**
   - Valid donor groups: `{', '.join(get_compatible_donors(selected_bg))}`
"""
        actions = [
            f"Audit current O- supply ({o_neg_stock} units) against active emergency surgeries.",
            "Enforce Rh D switching protocol for adult male trauma cases to conserve O-.",
            "Initiate immediate cross-matching to switch from uncrossmatched O- to type-specific PRBC as soon as patient blood sample arrives.",
        ]
        return {
            "title": "Emergency Cross-Matching Guidance",
            "category": "Immunohematology",
            "answer_markdown": ans.strip(),
            "action_items": actions,
            "key_metrics": {
                "O- (Universal) Units": f"{o_neg_stock} units",
                "O+ Units": f"{o_pos_stock} units",
                "Universal Readiness": "Sufficient" if o_neg_stock >= 10 else "Scarce Alert",
            },
        }

    # 3. DONOR RECALL & MOBILIZATION CAMPAIGN
    if any(k in q for k in ["donor", "mobiliz", "campaign", "sms", "alert", "blood drive", "recruit"]):
        # Identify eligible donors
        target_groups = critical_groups if critical_groups else ["O-", "O+", "A+", "B+"]
        eligible_targets = eligible_donors[eligible_donors["Blood Group"].isin(target_groups)] if not eligible_donors.empty else pd.DataFrame()

        sample_sms = (
            "URGENT: Blood Bank Alert! Our regional trauma center is experiencing a critical shortage of "
            f"{', '.join(target_groups[:3])} blood. You are eligible to donate today and save lives. "
            "Please walk in to City Blood Bank (8am-8pm) or reply YES to book a slot. Ph: +91-80-2553-2000."
        )

        ans = f"""
### 📢 Automated Donor Mobilization & Targeted Campaign Strategy

#### 👥 Donor Pool Health:
- **Total Registered Donors:** `{total_donors}`
- **Immediately Eligible Donors (Past 90-day cooldown):** `{eligible_count}`
- **Eligible Donors in High-Need Groups ({', '.join(target_groups)}):** `{len(eligible_targets)}`

#### 📱 Ready-to-Send SMS / WhatsApp Emergency Broadcast Template:
```text
{sample_sms}
```

#### 🎯 Mobilization Protocol:
1. **Tier 1 (Instant):** Automated SMS & WhatsApp push to `{len(eligible_targets)}` pre-screened eligible donors.
2. **Tier 2 (Same-Day):** Outbound phone calls by donor coordinators to voluntary corporate and college partners.
3. **Tier 3 (Next 48 Hours):** Mobile blood donation bus deployment to district tech parks and residential hubs.
"""
        actions = [
            f"Broadcast emergency SMS to {len(eligible_targets)} verified eligible donors.",
            "Open extended weekend donation hours (8:00 AM to 8:00 PM).",
            "Send appreciation certificates & digital donor badges to retain repeat voluntary donors.",
        ]
        return {
            "title": "Donor Mobilization Strategy",
            "category": "Operations",
            "answer_markdown": ans.strip(),
            "action_items": actions,
            "key_metrics": {
                "Eligible Donors": f"{eligible_count} / {total_donors}",
                "Priority Need Groups": ", ".join(target_groups[:3]),
                "Mobilization Readiness": "High" if eligible_count >= 10 else "Moderate",
            },
        }

    # 4. WASTAGE, SPOILAGE & FEFO OPTIMIZATION
    if any(k in q for k in ["spoil", "waste", "expir", "shelf", "fefo", "fifo", "outdate"]):
        exp_table = "\n".join(
            [f"- **{bg}**: `{units} units` nearing expiration" for bg, units in exp_dict.items() if units > 0]
        ) or "- No units currently flagged for imminent expiration (<3 days)."

        ans = f"""
### ⏳ Blood Component Shelf-Life & FEFO Spoilage Mitigation

Blood components are perishable biological assets. Maximizing utilization requires strict **FEFO (First-Expired, First-Out)** dispatch.

#### 📊 Current Expiration Exposure (< 3 Days to Outdating):
- **Total Units at Immediate Risk:** `{total_expiring} units`
{exp_table}

#### 🧬 Standard Clinical Component Shelf Lives:
- **Platelets:** `{SHELF_LIFE_DAYS.get('Platelets', 5)} days` (Highest spoilage risk due to room-temperature 20-24°C agitated storage).
- **Whole Blood:** `{SHELF_LIFE_DAYS.get('Whole Blood', 35)} days` (2°C to 6°C).
- **Packed Red Blood Cells (PRBC):** `{SHELF_LIFE_DAYS.get('Packed Red Blood Cells (PRBC)', 42)} days` (CPDA-1 preservative, 2°C to 6°C).
- **Fresh Frozen Plasma (FFP):** `{SHELF_LIFE_DAYS.get('Fresh Frozen Plasma (FFP)', 365)} days` (Deep frozen at $\le -18$°C).

#### 🛡️ Active Mitigation Guidelines:
1. **FEFO Allocation Priority:** Incoming routine surgery requests must automatically draw from oldest batches with $>48$h remaining viability.
2. **Inter-Hospital Cross-Transfer:** If local consumption cannot utilize expiring units within 48h, transfer to high-volume metropolitan surgical trauma centers.
3. **Platelet Demand Smoothing:** Implement pathogen-reduced single-donor apheresis platelets to extend viable life.
"""
        actions = [
            f"Flag all {total_expiring} near-expiry units with fluorescent orange FEFO batch tags.",
            "Route nearest-expiry units to scheduled elective orthopedic and cardiac surgeries scheduled within 24 hours.",
            "Record batch audit in electronic hospital blood inventory register.",
        ]
        return {
            "title": "Component Spoilage & FEFO Optimization",
            "category": "Quality Assurance",
            "answer_markdown": ans.strip(),
            "action_items": actions,
            "key_metrics": {
                "Expiring Soon Units": f"{total_expiring} units",
                "Platelet Viability Limit": "5 Days",
                "PRBC Viability Limit": "42 Days",
            },
        }

    # 5. EXECUTIVE SHIFT BRIEFING / MORNING ROUNDS
    if any(k in q for k in ["briefing", "morning", "shift", "handover", "overview", "rounds", "status", "summary"]):
        pending_count = len(pending_reqs)
        fulfilled_count = len(fulfilled_reqs)

        ans = f"""
### 📋 Executive Clinical Shift Briefing & Operational Handover

**Session Facility Context:** Multi-Facility Healthcare Network Blood Operations

#### 📊 High-Level Operational Metrics:
- **Total In-Bank Inventory:** `{total_stock} units` across all 8 blood groups.
- **Stockout Alerts:** `{len(critical_groups)} groups` in critical deficit ({', '.join(critical_groups) if critical_groups else 'None'}).
- **Units Expiring in $\le 72$ Hours:** `{total_expiring} units` (FEFO protocol active).
- **Clinical Demand Queue:** `{fulfilled_count}` fulfilled orders today, `{pending_count}` pending review.
- **Donor Registry Readiness:** `{eligible_count} of {total_donors} donors` eligible for immediate recall.

#### 🏥 Shift Directives for Senior Blood Bank Officer:
1. **Trauma & ICU Coverage:** Ensure at least 6 units of O- and 12 units of O+ remain reserved in the emergency cross-match blood refrigerator.
2. **Cold-Chain Audit:** Verify temperature logs across all walk-in cold rooms (2°C to 6°C) and platelet agitators (20°C to 24°C).
3. **Donor Drive Scheduling:** Coordinate next mobilization cycle for `{critical_groups[0] if critical_groups else 'O-'}` with regional volunteer organizations.
"""
        actions = [
            "Conduct physical inventory count against electronic dashboard numbers.",
            "Inspect cold chain temperature recorder charts for zero excursions.",
            "Clear pending clinical hospital requisitions with priority to ICU/NICU wards.",
        ]
        return {
            "title": "Executive Clinical Shift Briefing",
            "category": "Executive Briefing",
            "answer_markdown": ans.strip(),
            "action_items": actions,
            "key_metrics": {
                "Total Bank Units": f"{total_stock} units",
                "Pending Requisitions": f"{pending_count} orders",
                "Eligible Donors Ready": f"{eligible_count} donors",
            },
        }

    # 6. HOSPITAL REQUEST FULFILLMENT RATE
    if any(k in q for k in ["fulfillment", "hospital request", "satisfaction", "order"]):
        total_reqs = len(hospital_reqs_df)
        pending_cnt = len(pending_reqs)
        fulfilled_cnt = len(fulfilled_reqs)
        rate = (fulfilled_cnt / total_reqs * 100.0) if total_reqs > 0 else 100.0

        ans = f"""
### 🏥 Clinical Hospital Request Fulfillment Analytics

- **Total Hospital Requisitions:** `{total_reqs}`
- **Successfully Dispatched & Fulfilled:** `{fulfilled_cnt} ({rate:.1f}%)`
- **Pending or Shortage Requisitions:** `{pending_cnt}`

#### 🎯 Key Hospital Request Observations:
- **Emergency Trauma Orders:** Fast-track protocol enables instantaneous automated inventory deduction and dispatch.
- **Cross-Match Fallbacks:** When primary requested groups face stockout, emergency biological substitution ensures continuous trauma coverage.
"""
        actions = [
            f"Expedite resolution of {pending_cnt} pending hospital requisitions.",
            "Review stock reservation thresholds with hospital trauma surgical directors.",
        ]
        return {
            "title": "Hospital Requisition Fulfillment",
            "category": "Hospital Logistics",
            "answer_markdown": ans.strip(),
            "action_items": actions,
            "key_metrics": {
                "Fulfillment Rate": f"{rate:.1f}%",
                "Fulfilled Orders": f"{fulfilled_cnt}",
                "Pending Orders": f"{pending_cnt}",
            },
        }

    # 7. DEFAULT / CUSTOM QUERY SYNTHESIS FOR ACTIVE BLOOD GROUP
    group_inv = inv_dict.get(selected_bg, 0)
    thresh = thresh_dict.get(selected_bg, 10)
    compat = get_compatible_donors(selected_bg)

    ans = f"""
### 🤖 Clinical Copilot Advisory for Blood Group {selected_bg}

**Query Analysis:** *"{query}"*

#### 📊 Live Status for {selected_bg}:
- **Current On-Hand Inventory:** `{group_inv} units`
- **Low Stock Threshold:** `{thresh} units`
- **Status:** `{'🚨 Critical Deficit' if group_inv <= thresh else '✅ Safe Buffer'}`
- **Biological Compatible Donors:** `{', '.join(compat)}`

#### 💡 Prescriptive Recommendation:
- If demand surges for **{selected_bg}**, clinical staff can draw from compatible donors: **{', '.join(compat)}**.
- Maintain lead time replenishment orders and rotate older units first following FEFO guidelines.
"""
    actions = [
        f"Monitor daily burn rate of {selected_bg}.",
        f"Verify availability of compatible substitutes ({', '.join(compat)}).",
    ]
    return {
        "title": f"Copilot Intelligence: {selected_bg}",
        "category": "Clinical Decision Support",
        "answer_markdown": ans.strip(),
        "action_items": actions,
        "key_metrics": {
            "Group Stock": f"{group_inv} units",
            "Threshold": f"{thresh} units",
            "Compatible Donors": f"{len(compat)} groups",
        },
    }
