# PrognosAI: AI-Driven Predictive Maintenance System

PrognosAI is a predictive maintenance system that estimates the **Remaining Useful Life (RUL)** of industrial machinery using multivariate time-series sensor data. 

This project uses the **NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation)** dataset to train a Stacked Long Short-Term Memory (LSTM) neural network. It accurately predicts how many cycles an engine has left before failure, converting those predictions into actionable maintenance alerts.

## 🚀 Project Architecture (Milestone by Milestone)

This project is built and structured across 5 distinct milestones:

### 1️⃣ Milestone 1: Data Preparation (`data_preparation.py`)
- Loads the raw NASA CMAPSS sensor data (`train_FD001.txt`, `test_FD001.txt`).
- Computes the ground-truth Remaining Useful Life (RUL) for training.
- Applies **Piecewise Linear Target Capping (max 125 cycles)** to prevent gradient collapse during early engine life.
- Normalizes all 21 sensor readings using `MinMaxScaler`.
- Creates 30-cycle "rolling window" sequences to feed into the LSTM.

### 2️⃣ Milestone 2: Model Development (`model_development.py`)
- Implements a Stacked LSTM deep learning model using TensorFlow/Keras.
- Configured with Dropout layers and **Gradient Clipping (`clipnorm=1.0`)** to prevent overfitting and unstable updates.
- Includes dynamic learning rate reduction and `EarlyStopping` (over 100 epochs) to capture the absolute best model weights.

### 3️⃣ Milestone 3: Model Evaluation (`model_evaluation.py`)
- Evaluates the trained model against the unseen test dataset.
- Calculates key regression metrics: **RMSE** (Root Mean Squared Error), **MAE** (Mean Absolute Error), and **R² Score**.
- Generates error distribution and residual plots.

### 4️⃣ Milestone 4: Alert System (`alert_system.py`)
- Translates raw RUL predictions into business-actionable maintenance alerts.
- **Thresholds** (Scaled for 125-cycle max RUL): 
  - 🟢 **HEALTHY**: RUL > 100 cycles
  - 🟡 **WARNING**: 50 < RUL ≤ 100 cycles
  - 🟠 **CRITICAL**: 20 < RUL ≤ 50 cycles
  - 🔴 **FAILURE**: RUL ≤ 20 cycles

### 5️⃣ Milestone 5: Dashboard (`dashboard.py`)
- Uses `matplotlib` to generate executive fleet health dashboards.
- Features health gauges, fleet status distributions, RUL histograms, and granular unit-by-unit degradation curves.

---

## 🛠️ How to Run Locally

### 1. Install Dependencies
Make sure you have Python installed, then run:
```bash
pip install -r requirements.txt
```

### 2. Download the Dataset
The project expects the following NASA CMAPSS dataset files in the root directory:
- `train_FD001.txt`
- `test_FD001.txt`
- `RUL_FD001.txt`

### 3. Execute the Pipeline
Run the unified `main.py` script to execute the entire pipeline from data prep to dashboard generation:
```bash
python main.py
```
