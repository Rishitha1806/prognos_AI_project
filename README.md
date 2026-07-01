<div align="center">
  <h1>⚙️ PrognosAI</h1>
  <p><b>AI-Driven Predictive Maintenance for Turbofan Engines</b></p>
  
  ![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
  ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
  ![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-yellow.svg)
  ![Status](https://img.shields.io/badge/Status-Complete-success.svg)
</div>

<br/>

## 📖 Overview

**PrognosAI** is a robust, end-to-end Machine Learning pipeline designed to predict the **Remaining Useful Life (RUL)** of industrial machinery using multivariate time-series sensor data. 

Built around the benchmark **NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation)** dataset, this system trains a deep **Stacked Long Short-Term Memory (LSTM)** neural network to forecast exactly how many operational cycles an engine has left before critical failure. These predictions are then parsed through a business-logic layer to generate automated maintenance alerts and executive health dashboards.

---

## ✨ Key Technical Features

- **Advanced Sequence Engineering:** Transforms raw telemetry logs into 30-cycle sliding windows to capture temporal degradation patterns.
- **Piecewise Linear RUL Capping:** Implements a domain-specific `clip(upper=125)` target constraint to prevent gradient collapse during early, healthy engine life.
- **Deep Sequence Modeling:** Utilizes a stacked dual-layer LSTM architecture equipped with Dropout layers and **Gradient Clipping (`clipnorm=1.0`)** for smooth, stable convergence.
- **Dynamic Optimization:** Leverages `EarlyStopping` (with best-weight restoration) and `ReduceLROnPlateau` over a 100-epoch runway.
- **Business Logic Integration:** Maps numerical AI predictions to actionable fleet categories (`HEALTHY`, `WARNING`, `CRITICAL`, `FAILURE`).
- **Data Visualization:** Generates programmatic matplotlib dashboards detailing fleet-wide health distribution and individual engine degradation curves.

---

## 🏗️ Architecture & Pipeline

The pipeline is designed in a highly modular, object-oriented structure across 5 distinct milestones:

| Component | File | Description |
| :--- | :--- | :--- |
| **Data Prep** | `data_preparation.py` | Ingests data, calculates clipped RUL targets, applies MinMax scaling, and yields rolling sequences. |
| **Model Builder** | `model_development.py` | Defines, compiles, and trains the Deep LSTM neural network. |
| **Evaluator** | `model_evaluation.py` | Extracts RMSE, MAE, and R² metrics alongside deviation sanity checks. |
| **Alert System** | `alert_system.py` | Translates predictions into tailored, threshold-based maintenance alerts. |
| **Dashboards** | `dashboard.py` | Renders visual health gauges, pie charts, and trajectory plots for the fleet. |
| **Entry Point** | `main.py` | The unified execution script that orchestrates the entire pipeline. |

---

## 🚦 Alert System Thresholds

Predictions are categorized based on their severity (scaled to accommodate the 125-cycle maximum constraint):

* 🟢 **HEALTHY:** RUL > 100 cycles
* 🟡 **WARNING:** 50 < RUL ≤ 100 cycles *(Schedule Inspection)*
* 🟠 **CRITICAL:** 20 < RUL ≤ 50 cycles *(Maintenance Required)*
* 🔴 **FAILURE:** RUL ≤ 20 cycles *(Immediate Grounding)*

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python 3.8+ installed. Clone this repository and install the required dependencies:
```bash
git clone https://github.com/Rishitha1806/prognos_AI_project.git
cd prognos_AI_project
pip install -r requirements.txt
```

### 2. Add the Dataset
Due to file size constraints, the NASA CMAPSS `.txt` files are ignored by git. You must download the **FD001** dataset and place the following files in the root directory:
* `train_FD001.txt`
* `test_FD001.txt`
* `RUL_FD001.txt`

### 3. Run the Pipeline
Execute the main script to trigger the full pipeline from data preprocessing to dashboard rendering:
```bash
python main.py
```
*(Note: If a trained `rul_model.h5` is detected in the directory, the pipeline will skip training and proceed directly to evaluation and dashboard generation).*

---

## 🔮 Future Enhancements
* **Multi-Condition Scaling:** Implement K-Means clustering to normalize sensor readings across the 6 operating conditions present in the advanced `FD002` and `FD004` datasets.
* **Fault-Mode Classification:** Expand the network head to predict not only *when* the engine will fail, but *which* fault mode (e.g., HPC vs. Fan degradation) is the cause.
