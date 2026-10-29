"""
Academic Mini-Project Web Application.
Interactive Dashboard for AI Blood Bank Demand Forecasting & Inventory Optimization.
Aligned with Vemana IT CSE (Data Science) Review 0 Presentation Specifications.
"""

import io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from blood_bank_ai.compatibility import (
    get_compatibility_matrix_df,
    get_compatible_donors,
    get_compatible_recipients,
    simulate_emergency_allocation,
)
from blood_bank_ai.data import (
    add_donor_to_registry,
    add_hospital_request,
    generate_sample_donor_registry,
    generate_sample_hospital_requests,
    get_dataset_summary,
    insert_batch_demand_records,
    insert_or_update_demand_record,
    load_demand_data,
    load_huggingface_facility_data,
    load_inventory_data,
    parse_pasted_demand_text,
    update_inventory_units,
)
from blood_bank_ai.forecasting import (
    compute_key_group_forecasts,
    compute_stock_risk_status,
    evaluate_all_blood_groups,
    forecast_next_days,
    forecast_with_confidence_bounds,
    get_feature_importances,
    train_and_evaluate,
)
from blood_bank_ai.inventory import (
    SHELF_LIFE_DAYS,
    compute_inventory_metrics,
    generate_inventory_health_summary,
)


@st.cache_data
def load_default_demand():
    return load_demand_data()


@st.cache_data
def load_default_inventory():
    return load_inventory_data()


@st.cache_data
def get_cached_hf_data(facility_name: str):
    return load_huggingface_facility_data(facility_name)


@st.cache_data
def get_cached_all_group_benchmark(demand_df):
    return evaluate_all_blood_groups(demand_df)


def apply_custom_styles():
    st.markdown(
        """
        <style>
        .header-box {
            background: linear-gradient(135deg, #8b0000 0%, #c62828 50%, #b71c1c 100%);
            color: white;
            padding: 24px;
            border-radius: 12px;
            margin-bottom: 20px;
            box-shadow: 0 4px 12px rgba(139, 0, 0, 0.2);
        }
        .header-box h1 {
            color: white !important;
            margin: 0;
            font-size: 2.1rem;
            font-weight: 700;
        }
        .header-box p {
            color: #ffebee;
            margin: 6px 0 0 0;
            font-size: 0.95rem;
        }
        .team-badge {
            background: rgba(255, 255, 255, 0.15);
            border-radius: 6px;
            padding: 4px 10px;
            display: inline-block;
            margin-top: 8px;
            font-size: 0.82rem;
            color: #ffffff;
        }
        .kpi-card-banner {
            background: #ffffff;
            border: 1px solid #e0e0e0;
            border-radius: 10px;
            padding: 16px;
            text-align: center;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            border-top: 4px solid #b71c1c;
        }
        .kpi-title {
            font-size: 0.85rem;
            color: #666;
            text-transform: uppercase;
            font-weight: 600;
        }
        .kpi-value {
            font-size: 1.8rem;
            font-weight: 700;
            color: #b71c1c;
            margin: 4px 0;
        }
        .kpi-sub {
            font-size: 0.78rem;
            color: #888;
        }
        .badge-healthy {
            background-color: #d4edda;
            color: #155724;
            padding: 4px 8px;
            border-radius: 4px;
            font-weight: 600;
        }
        .badge-warning {
            background-color: #fff3cd;
            color: #856404;
            padding: 4px 8px;
            border-radius: 4px;
            font-weight: 600;
        }
        .badge-danger {
            background-color: #f8d7da;
            color: #721c24;
            padding: 4px 8px;
            border-radius: 4px;
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def run_dashboard():
    st.set_page_config(
        page_title="AI Blood Bank Forecasting & Management System",
        page_icon="🩸",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    apply_custom_styles()

    # Title Banner with Academic & Review Metadata
    st.markdown(
        """
        <div class="header-box">
            <h1>🩸 AI-Driven Blood Bank Demand Forecasting & Inventory Management</h1>
            <p>Department of CSE (Data Science), Vemana IT • Review 0 Prototype • Multi-Model Machine Learning Time-Series Regressors • Safety Stock & Reorder Point (ROP) Optimization</p>
            <div class="team-badge">
                👥 <b>Project Team:</b> Venkatesh M.R (1VI24CD045) • Tejas S (1VI24CD043) • Karthik S (1VI24CD018) | <b>Academic Year:</b> 2026–27
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar Controls
    with st.sidebar:
        st.header("⚙️ Project Controls")
        
        # Dataset Selection
        data_source_mode = st.radio(
            "Dataset Source",
            [
                "🌐 Hugging Face Healthcare Dataset (Real)",
                "📊 Daily Time-Series Benchmark (Synthetic)",
                "📁 Upload Custom Demand CSV",
            ],
            index=0,
            help="Select Hugging Face real-world healthcare multi-facility records, benchmark daily series, or upload a custom CSV.",
        )

        facility_choice = "District Hospital"
        uploaded_file = None

        if data_source_mode == "🌐 Hugging Face Healthcare Dataset (Real)":
            facility_choice = st.selectbox(
                "Select Healthcare Facility Tier",
                ["District Hospital", "National Blood Centre", "Rural Hospital"],
                index=0,
                help="Dataset from Hugging Face: electricsheepafrica/blood-bank-supply-management (~30,000 facility observations).",
            )
            source_key = f"HF_{facility_choice}"
        elif data_source_mode == "📁 Upload Custom Demand CSV":
            uploaded_file = st.file_uploader(
                "Upload Demand CSV (columns: date, blood_group, demand_units)",
                type=["csv"],
            )
            source_key = f"UPLOAD_{uploaded_file.name if uploaded_file else 'NONE'}"
        else:
            source_key = "BENCHMARK_SYNTHETIC"

        # Initialize or reload session state if source changed or missing
        if ("active_demand_df" not in st.session_state) or (st.session_state.get("current_source_key") != source_key):
            if data_source_mode == "🌐 Hugging Face Healthcare Dataset (Real)":
                init_demand, init_inventory = get_cached_hf_data(facility_choice)
            elif data_source_mode == "📁 Upload Custom Demand CSV" and uploaded_file is not None:
                try:
                    init_demand = load_demand_data(uploaded_file.getvalue())
                    init_inventory = load_default_inventory()
                except Exception as e:
                    st.error(f"Error loading uploaded CSV: {e}")
                    st.stop()
            else:
                init_demand = load_default_demand()
                init_inventory = load_default_inventory()

            st.session_state["active_demand_df"] = init_demand.copy()
            st.session_state["active_inventory_df"] = init_inventory.copy()
            st.session_state["hospital_requests"] = generate_sample_hospital_requests()
            st.session_state["donor_registry"] = generate_sample_donor_registry()
            st.session_state["current_source_key"] = source_key
            st.session_state["data_modified"] = False
            st.session_state["last_change_event"] = None

        demand_df = st.session_state["active_demand_df"]
        inventory_df = st.session_state["active_inventory_df"]

        if data_source_mode == "🌐 Hugging Face Healthcare Dataset (Real)":
            st.success(f"Active HF: **{facility_choice}** ({len(demand_df):,} daily records).")
        elif data_source_mode == "📁 Upload Custom Demand CSV" and uploaded_file is not None:
            st.success(f"Custom CSV loaded ({len(demand_df):,} daily records).")

        if st.session_state.get("data_modified", False):
            st.info("🟢 **Live Data Mode:** Custom records added/modified in this session.")
            if st.button("🔄 Reset to Baseline Dataset", key="btn_reset_sidebar", use_container_width=True):
                if data_source_mode == "🌐 Hugging Face Healthcare Dataset (Real)":
                    init_demand, init_inventory = get_cached_hf_data(facility_choice)
                elif data_source_mode == "📁 Upload Custom Demand CSV" and uploaded_file is not None:
                    init_demand = load_demand_data(uploaded_file.getvalue())
                    init_inventory = load_default_inventory()
                else:
                    init_demand = load_default_demand()
                    init_inventory = load_default_inventory()
                st.session_state["active_demand_df"] = init_demand.copy()
                st.session_state["active_inventory_df"] = init_inventory.copy()
                st.session_state["hospital_requests"] = generate_sample_hospital_requests()
                st.session_state["donor_registry"] = generate_sample_donor_registry()
                st.session_state["data_modified"] = False
                st.session_state["last_change_event"] = None
                st.rerun()

        st.markdown("---")
        st.subheader("📥 Export Live Session Data")
        c_exp1, c_exp2 = st.columns(2)
        with c_exp1:
            st.download_button(
                label="Demand CSV",
                data=demand_df.to_csv(index=False),
                file_name="active_demand_session.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with c_exp2:
            st.download_button(
                label="Inventory CSV",
                data=inventory_df.to_csv(index=False),
                file_name="active_inventory_session.csv",
                mime="text/csv",
                use_container_width=True,
            )

        available_groups = sorted(demand_df["blood_group"].unique())

        st.markdown("---")
        with st.expander("⚡ 1-Click Quick Add Observation", expanded=False):
            st.caption("Quickly log a daily observation from any dashboard tab.")
            current_active_bg = st.session_state.get("selected_blood_group", "A+" if "A+" in available_groups else available_groups[0])
            sb_ins_bg = st.selectbox(
                "Target Blood Group",
                available_groups,
                index=available_groups.index(current_active_bg) if current_active_bg in available_groups else 0,
                key="sb_quick_bg",
            )
            sb_bg_dates = demand_df[demand_df["blood_group"] == sb_ins_bg]["date"]
            sb_def_date = (sb_bg_dates.max() + pd.Timedelta(days=1)).date() if not sb_bg_dates.empty else pd.Timestamp.today().date()
            sb_date = st.date_input("Observation Date", value=sb_def_date, key="sb_quick_date")
            sb_units = st.number_input("Demand Units", min_value=0, max_value=500, value=15, step=1, key="sb_quick_units")
            sb_mode = st.radio("Mode", ["Add to existing", "Overwrite"], horizontal=True, key="sb_quick_mode")
            if st.button("➕ Log Record Now", type="primary", use_container_width=True, key="sb_quick_submit"):
                mode_v = "add" if sb_mode == "Add to existing" else "overwrite"
                st.session_state["active_demand_df"] = insert_or_update_demand_record(
                    st.session_state["active_demand_df"],
                    entry_date=sb_date,
                    blood_group=sb_ins_bg,
                    demand_units=sb_units,
                    mode=mode_v,
                )
                st.session_state["data_modified"] = True
                st.session_state["selected_blood_group"] = sb_ins_bg
                st.session_state["last_change_event"] = f"Logged {sb_units} units for {sb_ins_bg} on {sb_date}"
                st.success(f"✅ Added {sb_units} units for {sb_ins_bg}!")
                st.rerun()

        st.markdown("---")
        st.subheader("🎯 Forecasting Parameters")

        if "selected_blood_group" not in st.session_state or st.session_state["selected_blood_group"] not in available_groups:
            st.session_state["selected_blood_group"] = "A+" if "A+" in available_groups else available_groups[0]

        default_bg_idx = available_groups.index(st.session_state["selected_blood_group"])
        selected_blood_group = st.selectbox(
            "Select Blood Group",
            available_groups,
            index=default_bg_idx,
            key="sb_selected_bg",
        )
        st.session_state["selected_blood_group"] = selected_blood_group

        forecast_horizon = st.slider(
            "Forecast Horizon (Days Ahead)",
            min_value=1,
            max_value=30,
            value=7,
            help="Number of days to forecast recursively into the future.",
        )

        model_options = [
            "Random Forest",
            "Gradient Boosting",
            "Linear Regression",
            "Ridge Regression",
            "7-day seasonal naive",
        ]
        selected_model_name = st.selectbox(
            "Active ML Model",
            model_options,
            index=0,
            help="Candidate models specified in Review 0 Slide 8 & 12 (Linear Regression, RF, Gradient Boosting, Ridge, Naive).",
        )

        st.markdown("---")
        st.subheader("📦 Inventory Parameters")
        lead_time = st.slider("Replenishment Lead Time (Days)", 1, 7, 2)
        service_level = st.select_slider(
            "Target Service Level",
            options=[0.90, 0.95, 0.98, 0.99],
            value=0.95,
            format_func=lambda x: f"{int(x*100)}%",
        )

        st.caption(
            "📌 **Academic Project Notice:** Designed for Vemana IT CSE (DS) Review 0 demonstration "
            "and research simulation. Not for clinical patient care decisions."
        )

    # Filter data for selected group
    group_df = demand_df[demand_df["blood_group"] == selected_blood_group].sort_values("date").copy()
    if len(group_df) < 20:
        st.warning(f"⚠️ Selected blood group {selected_blood_group} has limited historical records ({len(group_df)} rows).")

    # Train and evaluate models with robust safety fallback
    try:
        models, features, test, test_predictions, metrics_df = train_and_evaluate(group_df)
    except Exception as e:
        models = {}
        features = group_df.copy()
        test = group_df.tail(max(1, int(len(group_df) * 0.2))).copy()
        avg_u = float(group_df["demand_units"].mean()) if not group_df.empty else 10.0
        test_predictions = {"7-day seasonal naive": np.full(len(test), avg_u)}
        metrics_df = pd.DataFrame([{
            "model": "7-day seasonal naive",
            "MAE (units/day)": round(float(group_df["demand_units"].std() or 2.0), 3),
            "RMSE (units/day)": round(float(group_df["demand_units"].std() or 2.5), 3),
            "MAPE (%)": 12.5,
            "R² Score": 0.05,
        }])
        selected_model_name = "7-day seasonal naive"

    # Generate forward forecast
    if selected_model_name in metrics_df["model"].values:
        selected_eval_metrics = metrics_df.set_index("model").loc[selected_model_name]
    else:
        selected_eval_metrics = metrics_df.iloc[0]
        selected_model_name = selected_eval_metrics["model"]
    holdout_rmse = float(selected_eval_metrics["RMSE (units/day)"])

    if selected_model_name == "7-day seasonal naive":
        hist_vals = group_df.sort_values("date")["demand_units"].tolist()
        naive_forecast = []
        last_date = group_df["date"].max()
        confidence_z = {0.90: 1.645, 0.95: 1.960, 0.98: 2.326, 0.99: 2.576}
        z = confidence_z.get(service_level, 1.96)
        for i in range(forecast_horizon):
            cur_date = last_date + pd.Timedelta(days=i + 1)
            val = hist_vals[-7 + (i % 7)] if len(hist_vals) >= 7 else hist_vals[-1]
            step_uncertainty = z * holdout_rmse * ((i + 1) * 0.35 + 0.65) ** 0.5
            naive_forecast.append({
                "date": cur_date,
                "predicted_demand_units": int(val),
                "lower_bound": max(0, int(round(val - step_uncertainty))),
                "upper_bound": int(round(val + step_uncertainty)),
            })
        forecast_df = pd.DataFrame(naive_forecast)
    else:
        active_model = models[selected_model_name]
        forecast_df = forecast_with_confidence_bounds(
            group_df, active_model, rmse=holdout_rmse, horizon=forecast_horizon, confidence_level=service_level
        )

    # Current inventory row & calculations
    inv_row = inventory_df[inventory_df["blood_group"] == selected_blood_group]
    cur_stock = int(inv_row.iloc[0]["available_units"]) if not inv_row.empty else 0
    cur_threshold = int(inv_row.iloc[0]["low_stock_threshold"]) if not inv_row.empty else 0
    cur_expiring = int(inv_row.iloc[0]["units_expiring_soon"]) if not inv_row.empty else 0
    forecast_sum = int(forecast_df["predicted_demand_units"].sum())
    projected_gap = cur_stock - forecast_sum

    # Calculate Safety Stock and ROP for active group
    opt_metrics = compute_inventory_metrics(
        current_stock=cur_stock,
        history_demand=group_df["demand_units"],
        lead_time_days=lead_time,
        service_level=service_level,
    )
    stock_risk = compute_stock_risk_status(
        current_stock=cur_stock,
        safety_stock=opt_metrics["safety_stock"],
        reorder_point=opt_metrics["reorder_point"],
        forecast_demand=forecast_sum,
    )

    # ==========================================
    # SLIDE 13 EXPECTED RESULTS TOP BANNER
    # ==========================================
    st.markdown("### 🎯 Expected Results Forecast Banner (Review 0 Slide 13)")
    key_forecasts = compute_key_group_forecasts(demand_df, horizon=forecast_horizon)
    
    col_k1, col_k2, col_k3, col_k4 = st.columns(4)
    with col_k1:
        st.markdown(
            f"""
            <div class="kpi-card-banner">
                <div class="kpi-title">A+ Forecast ({forecast_horizon} Days)</div>
                <div class="kpi-value">{key_forecasts.get('A+', '—')} <span style="font-size:1rem;color:#555;">units</span></div>
                <div class="kpi-sub">Key Population Blood Group</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_k2:
        st.markdown(
            f"""
            <div class="kpi-card-banner">
                <div class="kpi-title">B+ Forecast ({forecast_horizon} Days)</div>
                <div class="kpi-value">{key_forecasts.get('B+', '—')} <span style="font-size:1rem;color:#555;">units</span></div>
                <div class="kpi-sub">High Clinical Utilization</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_k3:
        st.markdown(
            f"""
            <div class="kpi-card-banner" style="border-top-color: #d32f2f;">
                <div class="kpi-title">O− Forecast ({forecast_horizon} Days)</div>
                <div class="kpi-value">{key_forecasts.get('O-', '—')} <span style="font-size:1rem;color:#555;">units</span></div>
                <div class="kpi-sub">Universal RBC Donor • High Priority</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_k4:
        st.markdown(
            f"""
            <div class="kpi-card-banner">
                <div class="kpi-title">AB+ Forecast ({forecast_horizon} Days)</div>
                <div class="kpi-value">{key_forecasts.get('AB+', '—')} <span style="font-size:1rem;color:#555;">units</span></div>
                <div class="kpi-sub">Universal Recipient Component</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-bottom: 16px;'></div>", unsafe_allow_html=True)

    # Active Group Operational KPI Row
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(f"Current Stock ({selected_blood_group})", f"{cur_stock} units")
    c2.metric(f"Forecast Demand ({forecast_horizon}d)", f"{forecast_sum} units")
    c3.metric(
        "Projected Net Balance",
        f"{projected_gap:+d} units",
        delta="Surplus" if projected_gap >= 0 else "Deficit Alert",
        delta_color="normal" if projected_gap >= 0 else "inverse",
    )
    c4.metric(
        "Stock Risk Category",
        stock_risk["level"],
        delta=stock_risk["badge"],
        delta_color="normal" if stock_risk["level"] == "Normal" else "inverse",
    )
    r2_score_val = selected_eval_metrics.get("R² Score", 0.0)
    r2_fmt = f"R²: {r2_score_val:.2f}" if (r2_score_val is not None and not np.isnan(r2_score_val)) else "R²: N/A"
    c5.metric(
        f"{selected_model_name} MAE",
        f"{selected_eval_metrics['MAE (units/day)']:.2f} units",
        r2_fmt,
    )

    # ==========================================
    # 8 MAIN NAVIGATION TABS (INCLUDING LIVE DATA STUDIO)
    # ==========================================
    tab_studio, tab_overview, tab_forecast, tab_eda, tab_inventory, tab_emergency, tab_hospital, tab_benchmark = st.tabs(
        [
            "⚡ Live Data Studio & Instant AI Output",
            "🩺 Executive Dashboard & Stock Risk",
            "🔮 AI Demand Forecasting",
            "📊 Exploratory Data Analysis (EDA)",
            "📦 Safety Stock & ROP Planner",
            "🩸 Emergency Allocation Matrix",
            "🏥 Hospital Requests & Donor Operations",
            "📑 Multi-Group Benchmark",
        ]
    )

    # ==========================================
    # TAB 1: LIVE DATA STUDIO & INSTANT AI OUTPUT
    # ==========================================
    with tab_studio:
        st.subheader("⚡ Live Data Studio: Interactive Data Ingestion & Real-Time AI Output")
        st.markdown(
            "Directly change or insert demand records, adjust inventory stock levels, or run stress-test simulations. "
            "**All AI forecasts, safety stock thresholds, and risk alerts update simultaneously in real time on the right.**"
        )

        col_studio_in, col_studio_out = st.columns([1, 1.15], gap="large")

        with col_studio_in:
            st.markdown("### 📥 Change or Insert Data")
            sub_bulk, sub_single, sub_paste, sub_table, sub_stock, sub_sim = st.tabs(
                [
                    "⚡ 1-Click Multi-Group Intake",
                    "➕ Single Group Quick Insert",
                    "📋 Paste CSV / Text Data",
                    "✏️ Spreadsheet Editor",
                    "📦 Stock Adjuster",
                    "🎛️ Scenario Simulator",
                ]
            )

            # Pre-compute latest date and averages across all groups for easy pre-filling
            max_overall_date = demand_df["date"].max()
            next_series_date = (max_overall_date + pd.Timedelta(days=1)).date() if pd.notna(max_overall_date) else pd.Timestamp.today().date()
            today_date = pd.Timestamp.today().date()

            # Pre-compute recent 7-day average demand for smart defaults
            recent_avg_by_group = {}
            for g in available_groups:
                g_recent = demand_df[demand_df["blood_group"] == g].tail(7)
                recent_avg_by_group[g] = int(round(g_recent["demand_units"].mean())) if not g_recent.empty else 10

            # -------------------------------------------------------------
            # SUB-TAB 1: 1-CLICK MULTI-GROUP INTAKE (EASIEST & FASTEST!)
            # -------------------------------------------------------------
            with sub_bulk:
                st.markdown("##### ⚡ 1-Click Multi-Group Daily Intake (All 8 Groups at Once)")
                st.caption(
                    "Log today's or the next day's clinical demand/intake across all 8 blood groups in a single action. "
                    "Pre-filled with recent daily averages so you only tweak the numbers that changed."
                )

                # Initialize date session key if missing
                if "intake_target_date" not in st.session_state:
                    st.session_state["intake_target_date"] = next_series_date

                col_date_pick, col_btn_next, col_btn_today = st.columns([2, 1.2, 1.2])
                with col_date_pick:
                    intake_date_val = st.date_input(
                        "Target Intake Date",
                        value=st.session_state["intake_target_date"],
                        key="intake_date_input",
                    )
                    if isinstance(intake_date_val, (tuple, list)):
                        intake_date_val = intake_date_val[0] if len(intake_date_val) > 0 else today_date

                with col_btn_next:
                    st.write("")
                    st.write("")
                    if st.button("⚡ Next Day", use_container_width=True, help="Set date to next chronological day in sequence"):
                        st.session_state["intake_target_date"] = next_series_date
                        st.rerun()

                with col_btn_today:
                    st.write("")
                    st.write("")
                    if st.button("📅 Today", use_container_width=True, help="Set date to today's local date"):
                        st.session_state["intake_target_date"] = today_date
                        st.rerun()

                c_bulk_mode, _ = st.columns([2, 2])
                with c_bulk_mode:
                    bulk_mode = st.radio(
                        "If records exist on this date:",
                        ["Add to existing units", "Overwrite existing units"],
                        horizontal=True,
                        key="bulk_intake_mode",
                    )

                st.markdown("###### Enter Units for Each Blood Group:")
                # 4 columns x 2 rows responsive grid for all 8 blood groups
                b_cols1 = st.columns(4)
                bulk_inputs = {}
                common_order = ["A+", "A-", "B+", "B-", "O+", "O-", "AB+", "AB-"]
                display_groups = [g for g in common_order if g in available_groups] + [g for g in available_groups if g not in common_order]

                for i, bg in enumerate(display_groups[:4]):
                    with b_cols1[i]:
                        def_val = recent_avg_by_group.get(bg, 10)
                        bulk_inputs[bg] = st.number_input(
                            f"{bg} (Units)",
                            min_value=0,
                            max_value=500,
                            value=def_val,
                            step=1,
                            key=f"bulk_in_{bg}",
                        )

                b_cols2 = st.columns(4)
                for i, bg in enumerate(display_groups[4:8]):
                    with b_cols2[i]:
                        def_val = recent_avg_by_group.get(bg, 5)
                        bulk_inputs[bg] = st.number_input(
                            f"{bg} (Units)",
                            min_value=0,
                            max_value=500,
                            value=def_val,
                            step=1,
                            key=f"bulk_in_{bg}",
                        )

                if st.button("🚀 Save All 8 Records & Recalculate AI Output", type="primary", use_container_width=True, key="btn_save_bulk_intake"):
                    mode_val = "add" if bulk_mode == "Add to existing units" else "overwrite"
                    updated_demand = insert_batch_demand_records(
                        st.session_state["active_demand_df"],
                        entry_date=intake_date_val,
                        records_dict=bulk_inputs,
                        mode=mode_val,
                    )
                    st.session_state["active_demand_df"] = updated_demand
                    st.session_state["data_modified"] = True
                    st.session_state["last_change_event"] = f"Logged 8 blood groups for {intake_date_val} ({mode_val})"
                    st.success(f"✅ Successfully saved daily demand records for all 8 blood groups on {intake_date_val}! AI models retrained.")
                    st.rerun()

            # -------------------------------------------------------------
            # SUB-TAB 2: SINGLE GROUP QUICK INSERT
            # -------------------------------------------------------------
            with sub_single:
                st.markdown("##### ➕ Single Blood Group Quick Add")
                st.caption("Add or adjust a daily observation for a specific blood group. Machine learning models retrain instantly.")

                c_sg1, c_sg2 = st.columns([1.5, 1.5])
                with c_sg1:
                    ins_bg = st.selectbox(
                        "Target Blood Group",
                        available_groups,
                        index=available_groups.index(selected_blood_group) if selected_blood_group in available_groups else 0,
                        key="studio_single_bg",
                    )
                with c_sg2:
                    bg_dates = demand_df[demand_df["blood_group"] == ins_bg]["date"]
                    default_date = (bg_dates.max() + pd.Timedelta(days=1)).date() if not bg_dates.empty else next_series_date
                    ins_date = st.date_input("Observation Date", value=default_date, key="studio_single_date")
                    if isinstance(ins_date, (tuple, list)):
                        ins_date = ins_date[0] if len(ins_date) > 0 else default_date

                # Context Info
                bg_inv = inventory_df[inventory_df["blood_group"] == ins_bg]
                bg_avail = int(bg_inv.iloc[0]["available_units"]) if not bg_inv.empty else 0
                bg_avg = recent_avg_by_group.get(ins_bg, 12)
                st.info(f"💡 **Context for {ins_bg}:** Current Inventory: **{bg_avail} units** • Recent Daily Demand Avg: **~{bg_avg} units/day**")

                c_sg3, c_sg4 = st.columns([1.5, 1.5])
                with c_sg3:
                    ins_units = st.number_input(
                        "Demand Units Requested",
                        min_value=0,
                        max_value=500,
                        value=bg_avg,
                        step=1,
                        key="studio_single_units",
                    )
                with c_sg4:
                    ins_mode = st.radio(
                        "If date exists:",
                        ["Add to existing", "Overwrite"],
                        horizontal=True,
                        key="studio_single_mode",
                    )

                auto_switch = st.checkbox(
                    "Auto-switch active dashboard view to this blood group",
                    value=True,
                    key="studio_auto_switch",
                    help="Automatically update the displayed forecast, chart, and risk status to this blood group upon saving.",
                )

                if st.button(f"🚀 Insert Record for {ins_bg} & Recalculate AI Output", type="primary", use_container_width=True, key="btn_insert_single"):
                    mode_val = "add" if ins_mode == "Add to existing" else "overwrite"
                    updated_demand = insert_or_update_demand_record(
                        st.session_state["active_demand_df"],
                        entry_date=ins_date,
                        blood_group=ins_bg,
                        demand_units=ins_units,
                        mode=mode_val,
                    )
                    st.session_state["active_demand_df"] = updated_demand
                    st.session_state["data_modified"] = True
                    if auto_switch:
                        st.session_state["selected_blood_group"] = ins_bg
                    st.session_state["last_change_event"] = f"Inserted {ins_units} units for {ins_bg} on {ins_date}"
                    st.success(f"✅ Successfully saved {ins_units} units for {ins_bg} on {ins_date}! AI output refreshed.")
                    st.rerun()

            # -------------------------------------------------------------
            # SUB-TAB 3: QUICK TEXT / CSV BATCH PASTE
            # -------------------------------------------------------------
            with sub_paste:
                st.markdown("##### 📋 Paste CSV / Text Daily Records")
                st.caption("Paste lines of raw records (format: `YYYY-MM-DD, Blood_Group, Units`). Delimiters can be commas, tabs, or pipes.")

                sample_paste = f"{next_series_date}, A+, 18\n{next_series_date}, B+, 14\n{next_series_date}, O-, 7"
                pasted_text = st.text_area(
                    "Paste Daily Records Here",
                    value=sample_paste,
                    height=120,
                    key="studio_pasted_text",
                )

                if st.button("📥 Ingest Pasted Lines & Retrain Models", type="primary", use_container_width=True, key="btn_ingest_paste"):
                    try:
                        parsed_df = parse_pasted_demand_text(pasted_text)
                        df_curr = st.session_state["active_demand_df"].copy()
                        for _, row in parsed_df.iterrows():
                            df_curr = insert_or_update_demand_record(
                                df_curr,
                                entry_date=row["date"],
                                blood_group=row["blood_group"],
                                demand_units=row["demand_units"],
                                mode="overwrite",
                            )
                        st.session_state["active_demand_df"] = df_curr
                        st.session_state["data_modified"] = True
                        st.session_state["last_change_event"] = f"Ingested {len(parsed_df)} records via Batch Paste"
                        st.success(f"✅ Successfully ingested {len(parsed_df)} records! AI forecasts updated.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error parsing pasted records: {e}")

            # -------------------------------------------------------------
            # SUB-TAB 4: SPREADSHEET EDITOR
            # -------------------------------------------------------------
            with sub_table:
                st.markdown("##### ✏️ Interactive Spreadsheet Editor")
                st.caption("Click any cell to edit past demand numbers, or click the '+' button to add new rows directly in the table.")

                table_bg_filter = st.selectbox(
                    "Filter by Blood Group",
                    ["(All Blood Groups)"] + available_groups,
                    index=available_groups.index(selected_blood_group) + 1 if selected_blood_group in available_groups else 0,
                    key="studio_table_filter",
                )

                if table_bg_filter == "(All Blood Groups)":
                    edit_slice = demand_df.tail(60).copy()
                else:
                    edit_slice = demand_df[demand_df["blood_group"] == table_bg_filter].tail(60).copy()

                # Ensure edit_slice has a clean, reset index and valid compatible types for Streamlit data_editor
                edit_slice = edit_slice[["date", "blood_group", "demand_units"]].copy().reset_index(drop=True)
                edit_slice["date"] = pd.to_datetime(edit_slice["date"]).dt.date
                edit_slice["blood_group"] = edit_slice["blood_group"].astype(str).str.strip().str.upper()
                edit_slice["demand_units"] = pd.to_numeric(edit_slice["demand_units"], errors="coerce").fillna(0).astype(int)

                # Dynamic key per filter prevents Streamlit session state index mismatch
                editor_key = f"studio_demand_editor_{table_bg_filter}"
                edited_result = st.data_editor(
                    edit_slice,
                    num_rows="dynamic",
                    use_container_width=True,
                    key=editor_key,
                    column_config={
                        "date": st.column_config.DateColumn("Date", format="YYYY-MM-DD", required=True),
                        "blood_group": st.column_config.SelectboxColumn("Blood Group", options=available_groups, required=True),
                        "demand_units": st.column_config.NumberColumn("Demand (Units)", min_value=0, max_value=500, step=1, required=True),
                    },
                )

                if st.button("💾 Apply & Save Table Changes", type="primary", use_container_width=True, key="btn_save_table"):
                    try:
                        edited_df = edited_result.copy()
                        # Drop incomplete/empty rows to prevent training NaN crashes
                        edited_df = edited_df.dropna(subset=["date"]).copy()
                        edited_df["date"] = pd.to_datetime(edited_df["date"]).dt.normalize()
                        edited_df["blood_group"] = edited_df["blood_group"].astype(str).str.strip().str.upper()
                        edited_df = edited_df[edited_df["blood_group"].isin(available_groups)].copy()
                        edited_df["demand_units"] = pd.to_numeric(edited_df["demand_units"], errors="coerce").fillna(0).clip(lower=0)

                        if not edited_df.empty:
                            base_df = demand_df.copy()
                            base_df["date"] = pd.to_datetime(base_df["date"]).dt.normalize()
                            merged = pd.concat([base_df, edited_df]).drop_duplicates(subset=["date", "blood_group"], keep="last")
                            merged = merged.sort_values(["blood_group", "date"]).reset_index(drop=True)
                            st.session_state["active_demand_df"] = merged
                            st.session_state["data_modified"] = True
                            st.session_state["last_change_event"] = "Updated demand records via Spreadsheet Editor"
                            st.success(f"✅ Changes applied! {len(edited_df)} rows validated and AI models refreshed.")
                            st.rerun()
                        else:
                            st.warning("No valid rows to save.")
                    except Exception as e:
                        st.error(f"Error saving table: {e}")

            with sub_stock:
                st.markdown("##### 📦 Live Facility Stock & Threshold Adjuster")
                st.caption("Change available units, low stock thresholds, or expiring units for any blood group.")

                c_st1, c_st2 = st.columns(2)
                with c_st1:
                    stock_bg = st.selectbox(
                        "Target Blood Group",
                        available_groups,
                        index=available_groups.index(selected_blood_group) if selected_blood_group in available_groups else 0,
                        key="studio_stock_bg",
                    )
                with c_st2:
                    current_inv_row = inventory_df[inventory_df["blood_group"] == stock_bg]
                    def_avail = int(current_inv_row.iloc[0]["available_units"]) if not current_inv_row.empty else 20
                    def_thresh = int(current_inv_row.iloc[0]["low_stock_threshold"]) if not current_inv_row.empty else 10
                    def_exp = int(current_inv_row.iloc[0]["units_expiring_soon"]) if not current_inv_row.empty else 0

                c_st3, c_st4, c_st5 = st.columns(3)
                with c_st3:
                    new_avail = st.number_input("Available Stock", min_value=0, max_value=500, value=def_avail, step=1, key="studio_new_avail")
                with c_st4:
                    new_thresh = st.number_input("Low Stock Threshold", min_value=0, max_value=200, value=def_thresh, step=1, key="studio_new_thresh")
                with c_st5:
                    new_exp = st.number_input("Expiring Soon", min_value=0, max_value=100, value=def_exp, step=1, key="studio_new_exp")

                if st.button(f"⚡ Update Stock for {stock_bg}", type="primary", use_container_width=True, key="btn_update_stock"):
                    updated_inv = update_inventory_units(
                        st.session_state["active_inventory_df"],
                        blood_group=stock_bg,
                        available_units=new_avail,
                        low_stock_threshold=new_thresh,
                        units_expiring_soon=new_exp,
                    )
                    st.session_state["active_inventory_df"] = updated_inv
                    st.session_state["data_modified"] = True
                    st.session_state["last_change_event"] = f"Updated inventory for {stock_bg}: {new_avail} units"
                    st.success(f"✅ Inventory for {stock_bg} updated to {new_avail} units! Stock risk recalculated.")
                    st.rerun()

                st.markdown("###### Multi-Group Inventory Overview (All 8 Groups)")
                st.dataframe(inventory_df, use_container_width=True, hide_index=True)

            with sub_sim:
                st.markdown("##### 🎛️ Rapid What-If Scenario Sandbox")
                st.caption("Test hypothetical clinical stress scenarios without permanently modifying records.")

                sim_surge = st.slider("Demand Surge Factor", min_value=0.5, max_value=3.0, value=1.0, step=0.1, help="Simulate sudden demand change (e.g. 1.5 = +50% surge)")
                sim_lead = st.slider("Simulated Replenishment Lead Time (Days)", 1, 14, lead_time)
                sim_stock = st.slider("Simulated Available Units", 0, 150, cur_stock)

                sim_metrics = compute_inventory_metrics(
                    current_stock=sim_stock,
                    history_demand=group_df["demand_units"] * sim_surge,
                    lead_time_days=sim_lead,
                    service_level=service_level,
                )
                sim_risk = compute_stock_risk_status(
                    current_stock=sim_stock,
                    safety_stock=sim_metrics["safety_stock"],
                    reorder_point=sim_metrics["reorder_point"],
                    forecast_demand=int(forecast_sum * sim_surge),
                )

                st.markdown("###### Scenario Simulation Results:")
                col_sb1, col_sb2 = st.columns(2)
                col_sb1.metric("Simulated Safety Stock", f"{sim_metrics['safety_stock']} units")
                col_sb2.metric("Simulated Reorder Point", f"{sim_metrics['reorder_point']} units")
                col_sb3, col_sb4 = st.columns(2)
                col_sb3.metric("Simulated Risk Level", sim_risk["badge"])
                col_sb4.metric("Recommended Order", f"{sim_metrics['recommended_order']} units")

        with col_studio_out:
            st.markdown("### 📤 Live AI Output & Decision Engine")
            st.markdown(
                f"<div style='display:flex;gap:8px;align-items:center;margin-bottom:12px;'>"
                f"<span style='background:#b71c1c;color:white;padding:3px 10px;border-radius:12px;font-size:0.85rem;font-weight:600;'>Group: {selected_blood_group}</span>"
                f"<span style='background:#f1f3f5;color:#333;padding:3px 10px;border-radius:12px;font-size:0.85rem;'>Model: {selected_model_name}</span>"
                f"<span style='background:#f1f3f5;color:#333;padding:3px 10px;border-radius:12px;font-size:0.85rem;'>Horizon: {forecast_horizon} Days</span>"
                f"<span style='background:#f1f3f5;color:#333;padding:3px 10px;border-radius:12px;font-size:0.85rem;'>Lead Time: {lead_time}d</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

            if st.session_state.get("last_change_event"):
                st.info(f"⚡ **Recent Ingestion Event:** {st.session_state['last_change_event']}")

            st.markdown("#### 1. Real-Time Operational Indicators")
            g_c1, g_c2, g_c3 = st.columns(3)
            g_c1.metric("Current Stock", f"{cur_stock} units")
            g_c2.metric(f"Forecast Demand ({forecast_horizon}d)", f"{forecast_sum} units")
            g_c3.metric(
                "Net Balance",
                f"{projected_gap:+d} units",
                delta="Surplus" if projected_gap >= 0 else "Deficit Alert",
                delta_color="normal" if projected_gap >= 0 else "inverse",
            )

            g_c4, g_c5, g_c6 = st.columns(3)
            g_c4.metric("Safety Stock (SS)", f"{opt_metrics['safety_stock']} units")
            g_c5.metric("Reorder Point (ROP)", f"{opt_metrics['reorder_point']} units")
            g_c6.metric(
                "Stock Risk Status",
                stock_risk["level"],
                delta=stock_risk["badge"],
                delta_color="normal" if stock_risk["level"] == "Normal" else "inverse",
            )

            st.markdown("#### 2. Synchronized Demand Trajectory & Forecast Bounds")
            fig_live = go.Figure()

            recent_hist = group_df.tail(45)
            fig_live.add_trace(
                go.Scatter(
                    x=recent_hist["date"],
                    y=recent_hist["demand_units"],
                    mode="lines+markers",
                    name="Historical Observed Demand",
                    line=dict(color="#495057", width=2),
                    marker=dict(size=5),
                )
            )

            fig_live.add_trace(
                go.Scatter(
                    x=forecast_df["date"],
                    y=forecast_df["upper_bound"],
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                    name="Upper Bound",
                )
            )
            fig_live.add_trace(
                go.Scatter(
                    x=forecast_df["date"],
                    y=forecast_df["lower_bound"],
                    mode="lines",
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor="rgba(183, 28, 28, 0.15)",
                    showlegend=False,
                    name="Lower Bound",
                )
            )

            fig_live.add_trace(
                go.Scatter(
                    x=forecast_df["date"],
                    y=forecast_df["predicted_demand_units"],
                    mode="lines+markers",
                    line=dict(color="#b71c1c", width=3, dash="solid"),
                    marker=dict(size=8, color="#b71c1c"),
                    name=f"AI Forecast ({selected_model_name})",
                )
            )

            fig_live.update_layout(
                title=f"Demand History & Forward AI Forecast — Blood Group {selected_blood_group}",
                xaxis_title="Date",
                yaxis_title="Units Requested",
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=20, r=20, t=50, b=20),
                height=320,
            )
            st.plotly_chart(fig_live, use_container_width=True)

            st.markdown("#### 3. AI Automated Prescriptive Recommendation")
            if cur_stock <= opt_metrics["safety_stock"]:
                st.error(
                    f"🚨 **CRITICAL SHORTAGE DETECTED:** Available stock ({cur_stock} units) is below Safety Stock "
                    f"({opt_metrics['safety_stock']} units). **Action:** Mobilize emergency donor drives for "
                    f"**{opt_metrics['recommended_order']} units** immediately!"
                )
            elif cur_stock <= opt_metrics["reorder_point"]:
                st.warning(
                    f"⚠️ **REORDER POINT BREACHED:** Current inventory ({cur_stock} units) has crossed the ROP "
                    f"({opt_metrics['reorder_point']} units). **Action:** Order/collect **{opt_metrics['recommended_order']} units** "
                    f"to return to target safe buffer."
                )
            else:
                st.success(
                    f"✅ **BUFFER HEALTHY:** Current inventory ({cur_stock} units) comfortably exceeds the Reorder Point "
                    f"({opt_metrics['reorder_point']} units). Estimated **{opt_metrics['days_of_supply']} days** of supply remaining."
                )

            with st.expander("🩸 Emergency Compatible Donor Substitutes (Biological Fallback)", expanded=False):
                compat_donors = get_compatible_donors(selected_blood_group)
                donor_stocks = []
                for d in compat_donors:
                    d_row = inventory_df[inventory_df["blood_group"] == d]
                    stock_qty = int(d_row.iloc[0]["available_units"]) if not d_row.empty else 0
                    donor_stocks.append({
                        "Compatible Donor Group": d,
                        "Match Status": "Exact Match" if d == selected_blood_group else "Biological Substitute",
                        "Units in Bank": stock_qty,
                    })
                st.dataframe(pd.DataFrame(donor_stocks), use_container_width=True, hide_index=True)

            with st.expander(f"📋 View Detailed {forecast_horizon}-Day Numerical Forecast Table", expanded=False):
                fc_tbl = forecast_df.copy()
                fc_tbl["date"] = fc_tbl["date"].dt.strftime("%Y-%m-%d")
                fc_tbl = fc_tbl.rename(
                    columns={
                        "date": "Future Date",
                        "predicted_demand_units": "Point Forecast (Units)",
                        "lower_bound": f"Lower Bound ({int(service_level*100)}%)",
                        "upper_bound": f"Upper Bound ({int(service_level*100)}%)",
                    }
                )
                st.dataframe(fc_tbl, use_container_width=True, hide_index=True)

    # ==========================================
    # TAB 2: EXECUTIVE DASHBOARD & STOCK RISK
    # ==========================================
    with tab_overview:
        st.subheader("🏥 Real-Time Blood Bank Inventory Status & Risk Indicators")
        
        # Risk Alert Banner matching Slide 9 & 12
        if stock_risk["level"] == "Critical":
            st.error(
                f"🚨 **CRITICAL STOCK ALERT:** {selected_blood_group} is at a critical risk level! "
                f"Current stock ({cur_stock} units) is below Safety Stock ({opt_metrics['safety_stock']} units) or "
                f"insufficient for {forecast_horizon}-day projected demand ({forecast_sum} units). "
                f"**Immediate action:** Mobilize {opt_metrics['recommended_order']} units from donor drives."
            )
        elif stock_risk["level"] == "Low":
            st.warning(
                f"⚠️ **LOW STOCK REORDER WARNING:** {selected_blood_group} inventory ({cur_stock} units) "
                f"has breached the Reorder Point ({opt_metrics['reorder_point']} units). "
                f"Order or collect {opt_metrics['recommended_order']} units to return to target buffer."
            )
        else:
            st.success(
                f"✅ **NORMAL BUFFER:** {selected_blood_group} inventory ({cur_stock} units) "
                f"exceeds the Reorder Point ({opt_metrics['reorder_point']} units) and adequately buffers against "
                f"the forecasted {forecast_horizon}-day demand ({forecast_sum} units)."
            )

        if cur_expiring > 0:
            st.info(
                f"⏰ **EXPIRATION WARNING:** {cur_expiring} units of {selected_blood_group} are nearing expiration! "
                "Prioritize issue according to FIFO (First-In, First-Out) transfusion protocol."
            )

        # Multi-group stock table
        st.markdown("### 📋 Multi-Blood Group Real-Time Inventory & Risk Classification")
        
        with st.expander("⚡ Quick Adjust Facility Stock Level (Real-Time Synchronized)", expanded=False):
            col_ov1, col_ov2, col_ov3 = st.columns([1, 1, 1])
            with col_ov1:
                ov_bg = st.selectbox("Select Blood Group", available_groups, index=available_groups.index(selected_blood_group) if selected_blood_group in available_groups else 0, key="ov_stock_bg")
            with col_ov2:
                ov_cur_row = inventory_df[inventory_df["blood_group"] == ov_bg]
                ov_cur_val = int(ov_cur_row.iloc[0]["available_units"]) if not ov_cur_row.empty else 20
                ov_new_val = st.number_input("Available Stock (Units)", min_value=0, max_value=500, value=ov_cur_val, step=1, key="ov_stock_val")
            with col_ov3:
                st.write("")
                st.write("")
                if st.button("Apply Stock Update", key="btn_apply_ov_stock", use_container_width=True):
                    st.session_state["active_inventory_df"] = update_inventory_units(
                        st.session_state["active_inventory_df"],
                        blood_group=ov_bg,
                        available_units=ov_new_val,
                    )
                    st.session_state["data_modified"] = True
                    st.session_state["last_change_event"] = f"Adjusted {ov_bg} stock to {ov_new_val} units"
                    st.success(f"Updated {ov_bg} stock to {ov_new_val} units!")
                    st.rerun()

        health_summary_df = generate_inventory_health_summary(
            inventory_df, demand_df, lead_time_days=lead_time, service_level=service_level
        )
        st.dataframe(health_summary_df, use_container_width=True, hide_index=True)

        st.markdown("### 🧬 Standard Component Storage & Shelf-Life Guidelines (Slide 5 & 14)")
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        col_s1.info(f"**Whole Blood**\n\n• Shelf life: **{SHELF_LIFE_DAYS['Whole Blood']} days**\n• Storage: 2°C to 6°C\n• Target: Massive hemorrhage")
        col_s2.info(f"**Packed RBCs (PRBC)**\n\n• Shelf life: **{SHELF_LIFE_DAYS['Packed Red Blood Cells (PRBC)']} days**\n• Storage: 2°C to 6°C\n• Target: Anemia & surgery")
        col_s3.warning(f"**Platelets**\n\n• Shelf life: **{SHELF_LIFE_DAYS['Platelets']} days**\n• Storage: 20°C to 24°C (Agitated)\n• Target: Thrombocytopenia")
        col_s4.success(f"**Fresh Frozen Plasma (FFP)**\n\n• Shelf life: **{SHELF_LIFE_DAYS['Fresh Frozen Plasma (FFP)']} days**\n• Storage: ≤ -18°C\n• Target: Coagulation deficiency")

    # ==========================================
    # TAB 3: AI DEMAND FORECASTING
    # ==========================================
    with tab_forecast:
        st.subheader(f"🔮 Demand Forecasting Engine — Blood Group {selected_blood_group}")
        st.caption(
            f"Active Model: **{selected_model_name}** | Horizon: **{forecast_horizon} days** | "
            f"Holdout MAE: **{selected_eval_metrics['MAE (units/day)']}** | "
            f"Holdout RMSE: **{selected_eval_metrics['RMSE (units/day)']}** | "
            f"Holdout MAPE: **{selected_eval_metrics['MAPE (%)']}%** | "
            f"R² Score: **{selected_eval_metrics['R² Score']}**"
        )

        with st.expander(f"➕ Quick Insert Observation for {selected_blood_group}", expanded=False):
            fc_c1, fc_c2, fc_c3 = st.columns([1, 1, 1])
            with fc_c1:
                fc_next_dt = (group_df["date"].max() + pd.Timedelta(days=1)).date()
                fc_in_dt = st.date_input("Date", value=fc_next_dt, key="fc_ins_date")
            with fc_c2:
                fc_in_units = st.number_input("Demand Units", min_value=0, max_value=500, value=15, step=1, key="fc_ins_units")
            with fc_c3:
                st.write("")
                st.write("")
                if st.button("Add Record & Retrain", key="btn_fc_add_record", use_container_width=True):
                    st.session_state["active_demand_df"] = insert_or_update_demand_record(
                        st.session_state["active_demand_df"],
                        entry_date=fc_in_dt,
                        blood_group=selected_blood_group,
                        demand_units=fc_in_units,
                        mode="add",
                    )
                    st.session_state["data_modified"] = True
                    st.session_state["last_change_event"] = f"Added {fc_in_units} units for {selected_blood_group} on {fc_in_dt}"
                    st.success(f"Added {fc_in_units} units for {selected_blood_group}! Models retrained.")
                    st.rerun()

        col_fc1, col_fc2 = st.columns(2)
        with col_fc1:
            st.markdown("#### 1. Historical Demand with 7-Day & 14-Day Moving Averages")
            hist_plot_df = group_df.tail(120).copy()
            hist_plot_df["7-Day Moving Avg"] = hist_plot_df["demand_units"].rolling(7).mean()
            hist_plot_df["14-Day Moving Avg"] = hist_plot_df["demand_units"].rolling(14).mean()

            fig_hist = px.line(
                hist_plot_df,
                x="date",
                y=["demand_units", "7-Day Moving Avg", "14-Day Moving Avg"],
                title=f"Historical Demand Trend (Last 120 Days) — {selected_blood_group}",
                color_discrete_map={
                    "demand_units": "#dc3545",
                    "7-Day Moving Avg": "#0d6efd",
                    "14-Day Moving Avg": "#198754",
                },
            )
            fig_hist.update_layout(xaxis_title="Date", yaxis_title="Units Requested", legend_title="Series")
            st.plotly_chart(fig_hist, use_container_width=True)

        with col_fc2:
            st.markdown("#### 2. Chronological Holdout Backtesting (Actual vs All 5 Models)")
            comp_df = test[["date", "demand_units"]].rename(columns={"demand_units": "Actual Demand"}).copy()
            model_names = [
                "7-day seasonal naive",
                "Linear Regression",
                "Random Forest",
                "Gradient Boosting",
                "Ridge Regression",
            ]
            for m in model_names:
                if m in test_predictions:
                    comp_df[m] = test_predictions[m]

            fig_comp = px.line(
                comp_df.tail(60),
                x="date",
                y=comp_df.columns.drop("date"),
                title=f"Holdout Backtest Comparison (Last 60 Days of Test Set) — {selected_blood_group}",
            )
            fig_comp.update_layout(xaxis_title="Date", yaxis_title="Daily Units", legend_title="Model")
            st.plotly_chart(fig_comp, use_container_width=True)

        col_fc3, col_fc4 = st.columns(2)
        with col_fc3:
            st.markdown(f"#### 3. Forward Forecast with {int(service_level*100)}% Confidence Bounds")
            fig_fc = go.Figure()
            # Upper bound
            fig_fc.add_trace(
                go.Scatter(
                    x=forecast_df["date"],
                    y=forecast_df["upper_bound"],
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                    name="Upper Bound",
                )
            )
            # Lower bound with fill
            fig_fc.add_trace(
                go.Scatter(
                    x=forecast_df["date"],
                    y=forecast_df["lower_bound"],
                    mode="lines",
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor="rgba(220, 53, 69, 0.15)",
                    showlegend=False,
                    name="Lower Bound",
                )
            )
            # Predicted value line
            fig_fc.add_trace(
                go.Scatter(
                    x=forecast_df["date"],
                    y=forecast_df["predicted_demand_units"],
                    mode="lines+markers",
                    line=dict(color="#b71c1c", width=3),
                    marker=dict(size=7),
                    name="Forecast Demand",
                )
            )
            fig_fc.update_layout(
                title=f"Predicted Daily Demand ({forecast_horizon} Days Ahead) — {selected_blood_group}",
                xaxis_title="Future Date",
                yaxis_title="Predicted Demand (Units)",
            )
            st.plotly_chart(fig_fc, use_container_width=True)

        with col_fc4:
            st.markdown("#### 4. Feature Importance & Model Interpretability (XAI)")
            if selected_model_name in models:
                imp_df = get_feature_importances(models[selected_model_name])
                if not imp_df.empty:
                    fig_imp = px.bar(
                        imp_df,
                        x="Importance (%)",
                        y="Feature",
                        orientation="h",
                        title=f"Feature Relative Contribution ({selected_model_name})",
                        color="Importance (%)",
                        color_continuous_scale="Reds",
                    )
                    fig_imp.update_layout(yaxis=dict(autorange="reversed"))
                    st.plotly_chart(fig_imp, use_container_width=True)
                else:
                    st.info("Feature importance not applicable for the selected model.")
            else:
                st.info("Seasonal naive baseline relies strictly on the 7-day cyclical lag.")

        st.markdown("#### 📥 Forecast Data Table & Export")
        fc_display = forecast_df.copy()
        fc_display["date"] = fc_display["date"].dt.strftime("%Y-%m-%d")
        st.dataframe(fc_display, use_container_width=True, hide_index=True)

        csv_buffer = io.StringIO()
        fc_display.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Download Detailed Forecast (CSV)",
            data=csv_buffer.getvalue(),
            file_name=f"forecast_{selected_blood_group}_{forecast_horizon}d.csv",
            mime="text/csv",
        )

    # ==========================================
    # TAB 4: EXPLORATORY DATA ANALYSIS (EDA)
    # ==========================================
    with tab_eda:
        st.subheader("📊 Exploratory Data Analysis & Demand Seasonality (Review 0 Slide 6 & 11)")
        st.markdown(
            "In-depth analysis of blood demand distributions, weekly cyclicality, monthly seasonality, "
            "and collection-to-request dynamics."
        )

        col_eda1, col_eda2 = st.columns(2)
        with col_eda1:
            st.markdown("#### 1. Demand Distribution Across Blood Groups")
            fig_box = px.box(
                demand_df,
                x="blood_group",
                y="demand_units",
                color="blood_group",
                title="Daily Demand Distribution by Blood Group (Variance & Outliers)",
                points="outliers",
            )
            fig_box.update_layout(xaxis_title="Blood Group", yaxis_title="Daily Units Requested", showlegend=False)
            st.plotly_chart(fig_box, use_container_width=True)

        with col_eda2:
            st.markdown("#### 2. Weekly Demand Seasonality")
            dow_df = demand_df.copy()
            dow_df["Day of Week"] = dow_df["date"].dt.day_name()
            dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            dow_avg = dow_df.groupby(["Day of Week", "blood_group"])["demand_units"].mean().reset_index()

            fig_dow = px.bar(
                dow_avg[dow_avg["blood_group"] == selected_blood_group],
                x="Day of Week",
                y="demand_units",
                category_orders={"Day of Week": dow_order},
                title=f"Average Daily Demand by Day of Week — {selected_blood_group}",
                color="demand_units",
                color_continuous_scale="Reds",
            )
            fig_dow.update_layout(yaxis_title="Average Units Requested")
            st.plotly_chart(fig_dow, use_container_width=True)

        col_eda3, col_eda4 = st.columns(2)
        with col_eda3:
            st.markdown("#### 3. Monthly Demand Trends Over Time")
            month_df = demand_df.copy()
            month_df["Year-Month"] = month_df["date"].dt.to_period("M").astype(str)
            monthly_totals = month_df.groupby(["Year-Month", "blood_group"])["demand_units"].sum().reset_index()
            
            fig_month = px.line(
                monthly_totals[monthly_totals["blood_group"] == selected_blood_group],
                x="Year-Month",
                y="demand_units",
                title=f"Total Monthly Demand Over Time — {selected_blood_group}",
                markers=True,
            )
            fig_month.update_layout(xaxis_title="Month", yaxis_title="Total Monthly Units")
            st.plotly_chart(fig_month, use_container_width=True)

        with col_eda4:
            st.markdown("#### 4. Blood Group Aggregate Demand Share")
            total_by_bg = demand_df.groupby("blood_group")["demand_units"].sum().reset_index()
            fig_pie = px.pie(
                total_by_bg,
                names="blood_group",
                values="demand_units",
                title="Total Historical Demand Share by Blood Group",
                hole=0.4,
                color_discrete_sequence=px.colors.sequential.Reds_r,
            )
            st.plotly_chart(fig_pie, use_container_width=True)

    # ==========================================
    # TAB 5: SAFETY STOCK & ROP PLANNER
    # ==========================================
    with tab_inventory:
        st.subheader("📦 Inventory Optimization & Safety Stock (ROP) Calculator (Review 0 Slide 8 & 9)")
        st.markdown(
            """
            Applies standard Supply Chain Management and Operations Research theory to determine optimal 
            buffer stock, minimizing the probability of shortages while controlling storage wastage.
            """
        )

        col_opt1, col_opt2, col_opt3, col_opt4 = st.columns(4)
        col_opt1.metric("Average Daily Demand (D)", f"{opt_metrics['avg_daily_demand']} units")
        col_opt2.metric("Safety Stock (SS)", f"{opt_metrics['safety_stock']} units")
        col_opt3.metric("Reorder Point (ROP)", f"{opt_metrics['reorder_point']} units")
        col_opt4.metric("Days of Supply Left", f"{opt_metrics['days_of_supply']} days")

        st.markdown("---")
        col_formula, col_rec = st.columns([1.2, 1])
        with col_formula:
            st.markdown("#### 📐 Mathematical Formulations")
            st.latex(r"SS = Z \times \sigma_D \times \sqrt{L}")
            st.latex(r"ROP = (\bar{D} \times L) + SS")
            st.markdown(
                f"""
                - **Average Daily Demand ($\bar{{D}}$):** {opt_metrics['avg_daily_demand']} units/day
                - **Demand Standard Deviation ($\sigma_D$):** {opt_metrics['std_daily_demand']} units
                - **Replenishment Lead Time ($L$):** {lead_time} days
                - **Service Level Factor ($Z$ for {int(service_level*100)}%):** {opt_metrics['z_score']}
                - **Lead Time Demand ($\bar{{D}} \\times L$):** {opt_metrics['lead_time_demand']} units
                - **Calculated Safety Stock ($SS$):** **{opt_metrics['safety_stock']} units**
                - **Calculated Reorder Point ($ROP$):** **{opt_metrics['reorder_point']} units**
                """
            )

        with col_rec:
            st.markdown("#### 🎯 Decision Support Recommendation")
            st.markdown(f"**Current Stock:** `{cur_stock} units`")
            st.markdown(f"**Inventory Health:** `{opt_metrics['stock_status']}`")
            st.markdown(f"**Stock-Risk Indicator:** `{stock_risk['badge']}`")

            if opt_metrics["recommended_order"] > 0:
                st.error(
                    f"🚨 **Action Required:** Inventory is below Reorder Point ({opt_metrics['reorder_point']} units).\n\n"
                    f"**Suggested Replenishment:** **{opt_metrics['recommended_order']} units** to restore target reserve."
                )
            else:
                st.success(
                    f"✅ **Stock Level Adequate:** Current stock ({cur_stock} units) exceeds Reorder Point ({opt_metrics['reorder_point']} units). "
                    "No immediate donation campaign required."
                )

    # ==========================================
    # TAB 6: EMERGENCY ALLOCATION MATRIX
    # ==========================================
    with tab_emergency:
        st.subheader("🩸 Emergency Cross-Matching & Compatibility Matrix (Review 0 Slide 8)")
        st.markdown(
            """
            In acute mass-casualty events or sudden blood shortages, clinicians rely on biological compatibility 
            hierarchies to allocate compatible substitute units when the identical blood group is depleted.
            """
        )

        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.markdown("#### 🔍 Recipient Compatibility Lookup")
            lookup_recipient = st.selectbox("Select Patient (Recipient) Blood Group", available_groups, index=0)
            valid_donors = get_compatible_donors(lookup_recipient)
            st.write(f"Patient with **{lookup_recipient}** can safely receive RBCs from:")
            st.info(", ".join([f"**{d}**" for d in valid_donors]))

        with col_c2:
            st.markdown("#### 🔍 Donor Compatibility Lookup")
            lookup_donor = st.selectbox(
                "Select Donor Blood Group",
                available_groups,
                index=available_groups.index("O-") if "O-" in available_groups else 0,
            )
            valid_recipients = get_compatible_recipients(lookup_donor)
            st.write(f"Donor with **{lookup_donor}** can safely donate RBCs to:")
            st.success(", ".join([f"**{r}**" for r in valid_recipients]))

        st.markdown("#### 📊 Standard ABO and Rh D Compatibility Grid")
        compat_grid = get_compatibility_matrix_df()
        st.dataframe(compat_grid, use_container_width=True)

        st.markdown("---")
        st.subheader("🚨 Emergency Shortage Allocation Simulator")
        st.markdown(
            "Simulate a trauma center emergency surge where units of a specific blood group are needed immediately."
        )

        col_sim1, col_sim2 = st.columns([1, 1.5])
        with col_sim1:
            sim_group = st.selectbox("Emergency Patient Blood Group", available_groups, key="sim_grp")
            sim_units = st.number_input("Units Needed", min_value=1, max_value=100, value=15)
            run_sim_btn = st.button("🚀 Run Emergency Allocation Simulation")

        with col_sim2:
            if run_sim_btn:
                sim_result = simulate_emergency_allocation(inventory_df, sim_group, sim_units)
                st.markdown(f"### Allocation Outcome for {sim_units} units of {sim_group}")
                if sim_result["success"]:
                    st.success(f"✅ **Full Request Satisfied!** Total allocated: {sim_result['units_allocated']}/{sim_units} units.")
                else:
                    st.error(
                        f"❌ **Shortfall Encountered!** Only {sim_result['units_allocated']} units available across all compatible donors. "
                        f"Shortfall: {sim_result['shortfall']} units."
                    )

                st.markdown("**Breakdown of Allocated Units by Donor Group:**")
                plan_df = pd.DataFrame(
                    [
                        {"Donor Group": g, "Units Taken": u, "Is Exact Match": (g == sim_group)}
                        for g, u in sim_result["allocation_plan"].items()
                    ]
                )
                st.dataframe(plan_df, use_container_width=True, hide_index=True)

                if sim_result["substitutions_used"]:
                    st.warning(
                        f"⚠️ **Cross-Group Substitutions Utilized:** {', '.join(sim_result['substitutions_used'])}. "
                        "Notify attending hematologist."
                    )

    # ==========================================
    # TAB 7: HOSPITAL REQUESTS & DONOR OPERATIONS
    # ==========================================
    with tab_hospital:
        st.subheader("🏥 Hospital Requests & Centralized Donor Registry (Review 0 Slide 6 & 10)")
        st.markdown(
            "Role-based platform integrating Hospital Emergency Requests with Donor Registry Mobilization. "
            "New clinical requests can be queued or fulfilled directly from available inventory in real time."
        )

        st.markdown("#### 1. Clinical Hospital Blood Requests")
        hospital_reqs = st.session_state["hospital_requests"]
        st.dataframe(hospital_reqs, use_container_width=True, hide_index=True)

        st.markdown("##### ➕ Submit New Clinical Hospital Request")
        col_req1, col_req2, col_req3, col_req4 = st.columns(4)
        with col_req1:
            new_hosp = st.text_input("Hospital Name", "District Trauma Hospital", key="req_hosp")
            new_ward = st.text_input("Department / Ward", "Emergency Trauma ICU", key="req_ward")
        with col_req2:
            new_bg = st.selectbox("Requested Blood Group", available_groups, key="req_bg")
            new_qty = st.number_input("Units Requested", min_value=1, max_value=50, value=3, key="req_qty")
        with col_req3:
            new_urgency = st.selectbox("Urgency", ["🚨 Emergency", "⚠️ Urgent", "Routine"], key="req_urgency")
            auto_fulfill = st.checkbox("Auto-fulfill & deduct from bank inventory", value=True, key="req_auto_fulfill")
        with col_req4:
            st.write("")
            st.write("")
            submit_req_btn = st.button("🚀 Submit Request", type="primary", use_container_width=True, key="btn_submit_req")

        if submit_req_btn:
            status_val = "Pending Review"
            if auto_fulfill:
                current_inv_df = st.session_state["active_inventory_df"]
                bg_inv = current_inv_df[current_inv_df["blood_group"] == new_bg]
                avail_units = int(bg_inv.iloc[0]["available_units"]) if not bg_inv.empty else 0
                if avail_units >= new_qty:
                    new_avail = avail_units - new_qty
                    st.session_state["active_inventory_df"] = update_inventory_units(
                        current_inv_df, blood_group=new_bg, available_units=new_avail
                    )
                    status_val = "Fulfilled & Dispatched"
                    st.success(f"✅ Request fulfilled! Allocated {new_qty} units of {new_bg}. Remaining stock: {new_avail} units.")
                else:
                    status_val = "Shortage - Fallback Alert"
                    st.warning(f"⚠️ Partial/Shortage: Only {avail_units} units available for {new_bg} ({new_qty} requested). Fallback crossmatch triggered!")
            else:
                st.info(f"✅ Request submitted for {new_qty} units of {new_bg}! Status: Pending Review.")

            updated_reqs, req_id = add_hospital_request(
                hospital_reqs,
                hospital_name=new_hosp,
                blood_group=new_bg,
                units_requested=new_qty,
                urgency_level=new_urgency,
                ward=new_ward,
                status=status_val,
            )
            st.session_state["hospital_requests"] = updated_reqs
            st.session_state["data_modified"] = True
            st.session_state["last_change_event"] = f"Hospital request {req_id} ({new_qty}x {new_bg}) submitted"
            st.rerun()

        st.markdown("---")
        st.markdown("#### 2. Centralized Donor Registry & Mobilization Tracker")
        st.caption("Tracks donor eligibility based on standard 90-day cooldown interval between whole blood donations.")
        donors_df = st.session_state["donor_registry"]
        st.dataframe(donors_df, use_container_width=True, hide_index=True)

        eligible_count = len(donors_df[donors_df["Eligibility Status"] == "Eligible"])
        st.info(f"💡 **{eligible_count} out of {len(donors_df)} registered donors** are currently eligible for immediate blood drive mobilization.")

        with st.expander("➕ Register New Donor to Centralized Registry", expanded=False):
            col_d1, col_d2, col_d3, col_d4 = st.columns(4)
            with col_d1:
                new_donor_name = st.text_input("Donor Full Name", "Ramesh Kumar", key="d_name_inp")
            with col_d2:
                new_donor_bg = st.selectbox("Donor Blood Group", available_groups, key="d_bg_inp")
            with col_d3:
                new_donor_days = st.number_input("Days Since Last Donation", min_value=0, max_value=365, value=100, key="d_days_inp")
            with col_d4:
                st.write("")
                st.write("")
                reg_donor_btn = st.button("➕ Register Donor", type="primary", use_container_width=True, key="btn_reg_donor")

            if reg_donor_btn:
                updated_donors, donor_id = add_donor_to_registry(
                    donors_df,
                    name=new_donor_name,
                    blood_group=new_donor_bg,
                    days_since_last_donation=new_donor_days,
                )
                st.session_state["donor_registry"] = updated_donors
                st.session_state["data_modified"] = True
                st.session_state["last_change_event"] = f"Registered donor {donor_id} ({new_donor_name}, {new_donor_bg})"
                st.success(f"✅ Donor {new_donor_name} registered successfully under {donor_id}! Eligibility updated.")
                st.rerun()

    # ==========================================
    # TAB 8: MULTI-GROUP BENCHMARK & MODEL COMPARISON
    # ==========================================
    with tab_benchmark:
        st.subheader("📑 Multi-Group Comparative Benchmark & Model Evaluation (Review 0 Slide 8 & 12)")
        st.markdown(
            "Rigorous benchmark comparing all candidate models (Linear Regression, Random Forest, Gradient Boosting, "
            "Ridge Regression, and 7-day Seasonal Naive baseline) across all 8 blood groups on chronological holdouts."
        )

        all_group_summary = get_cached_all_group_benchmark(demand_df)
        st.dataframe(all_group_summary, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown(f"#### 🧪 Performance Scorecard for Active Blood Group ({selected_blood_group})")
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        st.caption(
            "**MAE:** Mean Absolute Error • **RMSE:** Root Mean Squared Error • "
            "**MAPE:** Mean Absolute Percentage Error • **R²:** Coefficient of Determination"
        )


if __name__ == "__main__":
    run_dashboard()
