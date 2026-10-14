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
    generate_sample_donor_registry,
    generate_sample_hospital_requests,
    get_dataset_summary,
    load_demand_data,
    load_huggingface_facility_data,
    load_inventory_data,
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

        demand_df = None
        inventory_df = None

        if data_source_mode == "🌐 Hugging Face Healthcare Dataset (Real)":
            facility_choice = st.selectbox(
                "Select Healthcare Facility Tier",
                ["District Hospital", "National Blood Centre", "Rural Hospital"],
                index=0,
                help="Dataset from Hugging Face: electricsheepafrica/blood-bank-supply-management (~30,000 facility observations).",
            )
            demand_df, inventory_df = get_cached_hf_data(facility_choice)
            st.success(f"Loaded Hugging Face: **{facility_choice}** ({len(demand_df):,} daily records).")

        elif data_source_mode == "📁 Upload Custom Demand CSV":
            uploaded_file = st.file_uploader(
                "Upload Demand CSV (columns: date, blood_group, demand_units)",
                type=["csv"],
            )
            if uploaded_file is not None:
                try:
                    demand_df = load_demand_data(uploaded_file.getvalue())
                    inventory_df = load_default_inventory()
                    st.success("Uploaded CSV successfully loaded and validated!")
                except Exception as e:
                    st.error(f"Error loading uploaded CSV: {e}")
                    st.stop()
            else:
                st.info("Using default benchmark data until file is uploaded.")
                demand_df = load_default_demand()
                inventory_df = load_default_inventory()
        else:
            demand_df = load_default_demand()
            inventory_df = load_default_inventory()

        available_groups = sorted(demand_df["blood_group"].unique())

        st.markdown("---")
        st.subheader("🎯 Forecasting Parameters")
        selected_blood_group = st.selectbox(
            "Select Blood Group",
            available_groups,
            index=0 if "A+" not in available_groups else available_groups.index("A+"),
        )

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
    if len(group_df) < 55:
        st.error(f"Selected blood group {selected_blood_group} has insufficient historical records (needs >= 55).")
        st.stop()

    # Train and evaluate models
    models, features, test, test_predictions, metrics_df = train_and_evaluate(group_df)

    # Generate forward forecast
    selected_eval_metrics = metrics_df.set_index("model").loc[selected_model_name]
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
    c5.metric(
        f"{selected_model_name} MAE",
        f"{selected_eval_metrics['MAE (units/day)']:.2f} units",
        f"R²: {selected_eval_metrics['R² Score']:.2f}",
    )

    # ==========================================
    # 7 MAIN PRESENTATION NAVIGATION TABS
    # ==========================================
    tab_overview, tab_forecast, tab_eda, tab_inventory, tab_emergency, tab_hospital, tab_benchmark = st.tabs(
        [
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
    # TAB 1: EXECUTIVE DASHBOARD & STOCK RISK
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
    # TAB 2: AI DEMAND FORECASTING
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
    # TAB 3: EXPLORATORY DATA ANALYSIS (EDA)
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
    # TAB 4: SAFETY STOCK & ROP PLANNER
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
    # TAB 5: EMERGENCY ALLOCATION MATRIX
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
    # TAB 6: HOSPITAL REQUESTS & DONOR OPERATIONS
    # ==========================================
    with tab_hospital:
        st.subheader("🏥 Hospital Requests & Centralized Donor Registry (Review 0 Slide 6 & 10)")
        st.markdown(
            "Role-based platform integrating Hospital Emergency Requests with Donor Registry Mobilization."
        )

        st.markdown("#### 1. Clinical Hospital Blood Requests")
        hospital_reqs = generate_sample_hospital_requests()
        st.dataframe(hospital_reqs, use_container_width=True, hide_index=True)

        st.markdown("##### ➕ Submit New Clinical Hospital Request")
        col_req1, col_req2, col_req3, col_req4 = st.columns(4)
        with col_req1:
            new_hosp = st.text_input("Hospital Name", "District Trauma Hospital")
        with col_req2:
            new_bg = st.selectbox("Requested Blood Group", available_groups, key="new_bg")
        with col_req3:
            new_qty = st.number_input("Units", min_value=1, max_value=20, value=3)
        with col_req4:
            new_urgency = st.selectbox("Urgency", ["🚨 Emergency", "⚠️ Urgent", "Routine"])

        if st.button("Submit Request"):
            st.success(f"✅ Request submitted for **{new_qty} units of {new_bg}** for **{new_hosp}**! Status: *Queued for Crossmatch*.")

        st.markdown("---")
        st.markdown("#### 2. Centralized Donor Registry & Mobilization Tracker")
        st.caption("Tracks donor eligibility based on standard 90-day cooldown interval between whole blood donations.")
        donors_df = generate_sample_donor_registry()
        st.dataframe(donors_df, use_container_width=True, hide_index=True)

        eligible_count = len(donors_df[donors_df["Eligibility Status"] == "Eligible"])
        st.info(f"💡 **{eligible_count} out of {len(donors_df)} registered donors** are currently eligible for immediate blood drive mobilization.")

    # ==========================================
    # TAB 7: MULTI-GROUP BENCHMARK & MODEL COMPARISON
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
