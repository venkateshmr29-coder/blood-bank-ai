# 🩸 AI Blood Bank Demand Forecasting & Intelligent Management System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://venkateshmr29-coder-blood-bank-ai-app-98ipak.streamlit.app)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests: 42 Passed](https://img.shields.io/badge/tests-42%20passed-brightgreen.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end Machine Learning, Operations Research, and Clinical Decision Support System for Blood Banks developed for **Review 0**, Department of Computer Science & Engineering (Data Science), **Vemana Institute of Technology** (Academic Year 2026–27).

**Project Team:**
- **Venkatesh M.R** (`1VI24CD045`)
- **Tejas S** (`1VI24CD043`)
- **Karthik S** (`1VI24CD018`)

---

## 🌐 Live Web Application

The system is deployed and publicly accessible 24/7 on Streamlit Community Cloud:
👉 **[https://venkateshmr29-coder-blood-bank-ai-app-98ipak.streamlit.app](https://venkateshmr29-coder-blood-bank-ai-app-98ipak.streamlit.app)**

---

## 📌 Project Overview

Blood banks continuously face the operational challenge of maintaining sufficient reserves across eight distinct blood groups (`A+`, `A-`, `B+`, `B-`, `AB+`, `AB-`, `O+`, `O-`) while minimizing biological outdating and discard wastage. 

This platform transitions blood bank management from reactive tracking to **proactive, AI-driven healthcare operations** by integrating:

1. **Multi-Model Machine Learning Time-Series Regressors:** Evaluates and benchmarks **Linear Regression**, **Random Forest**, **Gradient Boosting**, **Ridge Regression**, and a **Weighted Stacking Ensemble (AI Stacking)** against a **7-Day Seasonal Naive Baseline** on chronological holdouts.
2. **Real-World Healthcare Datasets:** Ingests and trains on multi-facility hospital datasets from **Hugging Face** (`electricsheepafrica/blood-bank-supply-management`) across District Hospitals, National Blood Centres, and Rural Hospitals (~30,000 facility observations).
3. **AI Clinical Decision Support Copilot:** Offline-capable clinical decision support engine synthesizing real-time inventory levels, 7-day demand projections, biological compatibility, and donor readiness with automated SMS/alert templates.
4. **Regional Inter-Hospital Transfer Optimizer:** Multi-facility logistics network coordinating between National, District, and Rural hospital tiers to eliminate localized stockouts via cold-chain transfers.
5. **Disaster Surge & Mass-Casualty Incident (MCI) Triage Simulator:** Simulates acute trauma surges (highway pileups, blast incidents, dengue inpatient outbreaks) with universal donor exhaustion tracking and emergency cross-matching fallbacks.
6. **Component Shelf-Life & FEFO (First-Expired, First-Out) Optimization:** Granular batch age cohort tracking (Platelets 5d, PRBC 42d, Whole Blood 35d, FFP 365d) with automated FEFO dispatch queues to minimize biological discard waste.
7. **Demand Anomaly & Surge Detector:** Statistical 14-day rolling Z-score anomaly detector identifying unexpected clinical consumption spikes and intake deficits.
8. **Automated Clinical Audit Report Generator:** 1-click comprehensive executive audit reports with facility preparedness scoring (0-100) downloadable in Markdown.
9. **Probabilistic Inventory Optimization:** Supply Chain Safety Stock ($SS$), Reorder Point ($ROP$), Days of Supply, and 3-tier Stock-Risk classification (**Normal**, **Low**, **Critical**).
10. **Clinical Immunohematology & Surge Allocation:** Standard $8 \times 8$ ABO and Rh D compatibility matrices with dynamic emergency shortage allocation simulators.

> **Academic & Simulation Notice:** This application is an academic mini-project prototype developed for coursework evaluation and research demonstration. It is not an approved medical device or clinical system.

---

## 🏗️ System Architecture

```text
blood_bank_ai_mini_project/
├── app.py                             # Main Streamlit application entry point
├── blood_bank_ai/                     # Core application package
│   ├── __init__.py
│   ├── dashboard.py                   # 12-Tab interactive Streamlit UI
│   ├── copilot.py                     # AI Clinical Decision Support Copilot
│   ├── network.py                     # Inter-hospital supply chain & transfer optimizer
│   ├── disaster.py                    # Disaster Surge & MCI triage simulator
│   ├── spoilage.py                    # Component shelf-life & FEFO waste mitigation
│   ├── report.py                      # Automated clinical executive audit report generator
│   ├── data.py                        # Hugging Face ingestion, schemas, and profiling
│   ├── forecasting.py                 # Time-series features, ML regressors, and bounds
│   ├── inventory.py                   # Safety Stock (SS) & Reorder Point (ROP) math
│   └── compatibility.py               # ABO/Rh D compatibility & emergency substitution
├── data/                              # Datasets (HF real-world + synthetic benchmark)
│   ├── hf_district_daily_demand.csv
│   ├── hf_district_inventory.csv
│   ├── hf_national_daily_demand.csv
│   ├── hf_national_inventory.csv
│   ├── hf_rural_daily_demand.csv
│   ├── hf_rural_inventory.csv
│   ├── sample_demand_synthetic.csv
│   └── sample_inventory_synthetic.csv
├── tests/                             # Comprehensive test suite (42 unit tests)
│   ├── test_copilot.py
│   ├── test_network.py
│   ├── test_disaster.py
│   ├── test_spoilage.py
│   ├── test_report.py
│   ├── test_compatibility.py
│   ├── test_data.py
│   ├── test_forecasting.py
│   └── test_inventory.py
├── .streamlit/
│   └── config.toml                    # Server and browser configuration
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🚀 Quickstart & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/venkateshmr29-coder/blood-bank-ai.git
cd blood-bank-ai
```

### 2. Set Up Virtual Environment
On macOS / Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

On Windows (PowerShell):
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Run the Comprehensive Unit Test Suite
```bash
python -m unittest discover -s tests -v
```
*(All 42 unit tests pass cleanly).*

### 4. Launch the Streamlit Dashboard Locally
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📊 Dashboard Modules & Advanced Features

The platform is structured into 12 dedicated interactive presentation tabs:

1. **⚡ Live Data Studio & Instant AI Output (Real-Time Ingestion):** Side-by-side interactive studio where users can insert new daily records, edit data via an interactive spreadsheet table (`st.data_editor`), adjust facility stock, or test stress scenarios. **All AI forecasts, safety stock thresholds, ROP, and risk alerts update simultaneously in real time on the same screen.**
2. **🩺 Executive Dashboard & Clinical Audit Report:** Top Expected Results banner (A+, B+, O-, AB+), real-time stock levels with `Normal`, `Low`, and `Critical` indicators, inline stock adjuster, component shelf-life guidelines, and **1-click Executive Clinical Operational Audit Report Generator** with facility readiness scoring (0-100).
3. **🤖 AI Clinical Decision Support Copilot:** Conversational clinical operational intelligence answering critical stockout questions, drafting emergency donor mobilization SMS broadcasts, and generating morning executive rounds briefings.
4. **🔮 AI Demand Forecasting & Anomaly Detection:** Multi-step recursive forecasts with analytical confidence intervals, **AI Top Recommended Model** indicator, **Weighted Stacking Ensemble Regressor**, Explainable AI (XAI) feature importances, and statistical rolling Z-score demand anomaly detection.
5. **🌐 Regional Inter-Hospital Network Rebalancing:** Multi-facility logistics optimizer matching surplus at National Blood Centre with deficits at District and Rural hospitals. Generates transit routing, cold-chain specifications (2°C to 6°C), and 1-click transfer dispatch.
6. **🚨 Disaster Surge & Mass-Casualty Incident (MCI) Simulator:** Stress-tests blood bank resilience against mass-casualty scenarios (Highway Multi-Vehicle Crash, Dengue Epidemic, Code Black Blast Incident). Calculates universal O- exhaustion, operational runway in hours, and emergency action directives.
7. **⏳ FEFO & Component Spoilage Planner:** Component perishability modeling (Platelets 5d, PRBC 42d, FFP 365d), batch age distribution cohorts (Critical, Aging, Prime, Fresh), 3-day outdating risk matrix, and interactive FEFO order dispatch simulator.
8. **📊 Exploratory Data Analysis (EDA):** Blood-group demand distribution boxplots, day-of-week seasonality, and longitudinal monthly demand patterns.
9. **📦 Safety Stock & ROP Planner:** Mathematical supply chain formulas:
   $$SS = Z \times \sigma_D \times \sqrt{L}$$
   $$ROP = (\bar{D} \times L) + SS$$
10. **🩸 Emergency Cross-Matching & Allocation Matrix:** $8 \times 8$ Red Blood Cell and Plasma compatibility matrices with emergency trauma surge allocation simulator.
11. **🏥 Hospital Requests & Donor Operations:** Interactive clinical hospital request dispatcher with real-time inventory deduction/allocation and centralized donor registry tracking 90-day donation cooldown intervals.
12. **📑 Multi-Group Benchmark:** Unified comparative performance scorecard across all 8 blood groups and 6 candidate models.

---

## 📈 Model Performance on Hugging Face Data

Evaluated on chronological 80/20 holdout splits:

| Model | Architecture | MAE (units/day) | RMSE (units/day) | $R^2$ Score |
| :--- | :--- | :---: | :---: | :---: |
| **7-Day Seasonal Naive** | Baseline Heuristic | 11.93 | 15.28 | 0.14 |
| **Linear Regression** | Ordinary Least Squares | 9.32 | 11.81 | 0.49 |
| **Ridge Regression** | L2 Regularized Linear | 9.32 | 11.81 | 0.49 |
| **Random Forest** | 100 Trees Ensemble | 9.72 | 12.48 | 0.43 |
| **Gradient Boosting** | Sequential Shallow Trees | 8.94 | 11.58 | 0.51 |
| **Weighted Ensemble (AI Stacking)** | Inverse-RMSE Stacked Trees + Ridge | **8.81** | **11.42** | **0.53** |

---

## 🌐 Live Cloud Deployment

The live web application is hosted at:  
👉 **`https://venkateshmr29-coder-blood-bank-ai-app-98ipak.streamlit.app`**

Automatic continuous deployment is enabled via GitHub integration on `main` branch.

---

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).
