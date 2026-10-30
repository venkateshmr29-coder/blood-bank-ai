"""Blood bank demand forecasting and intelligent inventory management system."""

from blood_bank_ai.compatibility import (
    RBC_COMPATIBILITY,
    PLASMA_COMPATIBILITY,
    get_compatibility_matrix_df,
    get_compatible_donors,
    get_compatible_recipients,
    simulate_emergency_allocation,
)
from blood_bank_ai.copilot import ask_blood_bank_copilot
from blood_bank_ai.data import (
    load_demand_data,
    load_huggingface_facility_data,
    load_inventory_data,
)
from blood_bank_ai.disaster import (
    DISASTER_SCENARIOS,
    get_preset_disaster_scenarios,
    simulate_disaster_surge,
)
from blood_bank_ai.forecasting import (
    WeightedEnsembleRegressor,
    detect_demand_anomalies,
    forecast_next_days,
    forecast_with_confidence_bounds,
    recommend_best_model,
    train_and_evaluate,
)
from blood_bank_ai.inventory import (
    SHELF_LIFE_DAYS,
    compute_inventory_metrics,
    generate_inventory_health_summary,
)
from blood_bank_ai.network import (
    generate_facility_comparison_matrix,
    load_all_facility_snapshots,
    optimize_inter_hospital_transfers,
)
from blood_bank_ai.report import generate_audit_report_markdown
from blood_bank_ai.spoilage import (
    compute_spoilage_risk_matrix,
    generate_component_batch_breakdown,
    generate_fefo_dispatch_queue,
)

__all__ = [
    "RBC_COMPATIBILITY",
    "PLASMA_COMPATIBILITY",
    "get_compatibility_matrix_df",
    "get_compatible_donors",
    "get_compatible_recipients",
    "simulate_emergency_allocation",
    "ask_blood_bank_copilot",
    "load_demand_data",
    "load_huggingface_facility_data",
    "load_inventory_data",
    "DISASTER_SCENARIOS",
    "get_preset_disaster_scenarios",
    "simulate_disaster_surge",
    "WeightedEnsembleRegressor",
    "detect_demand_anomalies",
    "forecast_next_days",
    "forecast_with_confidence_bounds",
    "recommend_best_model",
    "train_and_evaluate",
    "SHELF_LIFE_DAYS",
    "compute_inventory_metrics",
    "generate_inventory_health_summary",
    "generate_facility_comparison_matrix",
    "load_all_facility_snapshots",
    "optimize_inter_hospital_transfers",
    "generate_audit_report_markdown",
    "compute_spoilage_risk_matrix",
    "generate_component_batch_breakdown",
    "generate_fefo_dispatch_queue",
]
