# 🩸 Blood Bank Demand Forecasting and Management Using AI Approaches

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

An end-to-end Machine Learning and Decision Support System for Blood Banks developed for **Review 0**, Department of Computer Science & Engineering (Data Science), **Vemana Institute of Technology** (Academic Year 2026–27).

**Project Team:**
- **Venkatesh M.R** (`1VI24CD045`)
- **Tejas S** (`1VI24CD043`)
- **Karthik S** (`1VI24CD018`)

---

## 📌 Project Overview

Blood banks continuously face the operational challenge of maintaining sufficient reserves across eight distinct blood groups (`A+`, `A-`, `B+`, `B-`, `AB+`, `AB-`, `O+`, `O-`) while minimizing outdating and wastage. 

This platform transitions blood bank management from reactive tracking to **proactive, AI-driven planning** by integrating:
1. **Time-Series Machine Learning Models:** Trains and benchmarks **Linear Regression**, **Random Forest Regressors**, **Gradient Boosting Regressors**, and **Ridge Regression** against a **7-Day Seasonal Naive Baseline** on chronological holdouts.
2. **Real-World Healthcare Datasets:** Ingests and trains on multi-facility hospital datasets from **Hugging Face** (`electricsheepafrica/blood-bank-supply-management`) across District Hospitals, National Blood Centres, and Rural Hospitals (~30,000 facility observations).
3. **Probabilistic Inventory Theory:** Implements formal Supply Chain Safety Stock ($SS$), Reorder Point ($ROP$), Days of Supply, and 3-tier Stock-Risk classification (**Normal**, **Low**, **Critical**).
4. **Clinical Immunohematology & Surge Allocation:** Standard $8 \times 8$ ABO and Rh D compatibility matrices with dynamic emergency shortage allocation simulators.
5. **Hospital & Donor Operations:** Centralized clinical request tracking and donor registry with automated 90-day cooldown eligibility monitoring.

> **Academic & Simulation Notice:** This application is an academic mini-project prototype developed for coursework evaluation and research demonstration. It is not an approved medical device or clinical system.

---

## 🏗️ System Architecture

```text
blood_bank_ai_mini_project/
├── app.py                             # Main Streamlit application entry point
├── blood_bank_ai/                     # Core application package
│   ├── __init__.py
│   ├── dashboard.py                   # 7-Tab interactive Streamlit UI
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
├── tests/                             # Comprehensive test suite (20 unit tests)
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

### 3. Run the Unit Tests
```bash
python -m unittest discover -s tests -v
```
*(All 24 unit tests should pass cleanly).*

### 4. Start the Dashboard
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📊 Dashboard Modules & Features

The dashboard is structured into 8 dedicated presentation tabs:

1. **⚡ Live Data Studio & Instant AI Output (Real-Time Ingestion):** Side-by-side interactive studio where users can insert new daily demand records, edit data via an interactive spreadsheet table (`st.data_editor`), adjust facility inventory levels, or test stress scenarios. **AI forecasts, safety stock thresholds, ROP, and risk alerts update simultaneously in real time on the same screen.**
2. **🩺 Executive Dashboard & Stock Risk:** Top Expected Results banner (A+, B+, O-, AB+), real-time stock levels with `Normal`, `Low`, and `Critical` indicators, inline stock level adjuster, FIFO expiration warnings, and component shelf-life guidelines.
3. **🔮 AI Demand Forecasting:** Historical demand trends with 7-day and 14-day rolling averages, inline observation inserter, backtest comparisons across all 5 models, multi-step recursive forecasts with analytical confidence intervals, and Explainable AI (XAI) feature importances.
4. **📊 Exploratory Data Analysis (EDA):** Blood-group demand distribution boxplots, day-of-week seasonality, and longitudinal monthly demand patterns.
5. **📦 Safety Stock & ROP Planner:** Mathematical supply chain formulas:
   $$SS = Z \times \sigma_D \times \sqrt{L}$$
   $$ROP = (\bar{D} \times L) + SS$$
6. **🩸 Emergency Allocation Matrix:** $8 \times 8$ Red Blood Cell and Plasma compatibility matrices with emergency trauma surge allocation simulator.
7. **🏥 Hospital Requests & Donor Operations:** Interactive clinical hospital request dispatcher with real-time inventory deduction/allocation and centralized donor registry tracking 90-day donation cooldown intervals.
8. **📑 Multi-Group Benchmark:** Unified comparative performance scorecard across all 8 blood groups and 5 candidate models.

---

## 📈 Model Performance on Hugging Face Data

Evaluated on chronological 80/20 holdout splits:

| Model | Architecture | MAE (units/day) | RMSE (units/day) | $R^2$ Score |
| :--- | :--- | :---: | :---: | :---: |
| **7-Day Seasonal Naive** | Baseline Heuristic | 11.93 | 15.28 | 0.14 |
| **Linear Regression** | Ordinary Least Squares | 9.32 | 11.81 | 0.49 |
| **Ridge Regression** | L2 Regularized Linear | 9.32 | 11.81 | 0.49 |
| **Random Forest** | 250 Trees Ensemble | 9.72 | 12.48 | 0.43 |
| **Gradient Boosting** | Sequential Shallow Trees | **8.94** | **11.58** | **0.51** |

---

## 🌐 Deployment to Streamlit Community Cloud

1. Push this repository to GitHub.
2. Visit [share.streamlit.io](https://share.streamlit.io).
3. Connect your GitHub account and select repository: `venkateshmr29-coder/blood-bank-ai`.
4. Set Main file path: `app.py`.
5. Click **Deploy!**

---

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).
