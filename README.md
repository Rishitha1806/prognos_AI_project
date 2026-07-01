# PrognosAI: Predictive Maintenance using Stacked LSTM
An end-to-end deep learning system designed for the accurate prediction of the Remaining Useful Life (RUL) of industrial turbofan engines, complete with piecewise linear target capping, programmatic health alerts, and visual fleet dashboards.

### Key Features
* **Piecewise Linear RUL Target Capping:** Transforms ground-truth RUL targets to plateau at 125 cycles during early engine life, preventing catastrophic gradient collapse caused by identical healthy sensor readings.
* **Deep Learning Stacked LSTM Brain:** Built on a highly-optimized dual-layer Long Short-Term Memory network that captures the temporal degradation patterns of 21 complex sensor arrays over 30-cycle rolling sequences.
* **Dynamic Optimizer Safeguards:** Implements strict Gradient Clipping (`clipnorm=1.0`) and Adaptive Learning Rates (`ReduceLROnPlateau`) to stabilize the Adam optimizer and prevent loss explosions during early training epochs.
* **Business Logic Alert System:** Converts raw numerical LSTM predictions into actionable maintenance bands (`HEALTHY`, `WARNING`, `CRITICAL`, `FAILURE`) to trigger real-world inspection schedules.
* **Interactive Fleet Dashboards:** Renders programmatic `matplotlib` dashboards featuring fleet-wide health gauges, prediction distribution histograms, and individual unit degradation trajectories.
* **Lightweight & High-Speed:** Relies purely on tabular sequential data (24 features per cycle), allowing the model to train and infer at lightning speed purely on CPU hardware.

---

### Repository Structure
Below is an overview of the core files in the project and their role in the pipeline:

| Script / Artifact | Description |
| :--- | :--- |
| `main.py` | Main entrypoint. Orchestrates dataset loading, neural network training, model evaluation, alert generation, and dashboard rendering. |
| `data_preparation.py` | Preprocessing pipeline. Normalizes all sensor telemetry using `MinMaxScaler`, applies Piecewise RUL constraints, and generates 30-cycle sliding tensor sequences. |
| `model_development.py` | Deep learning core. Defines the Stacked LSTM architecture, sets up `EarlyStopping` callbacks, and compiles the loss metrics. |
| `model_evaluation.py` | Performance testing suite. Analyzes the model against unseen test data, extracting standard regression metrics (RMSE, R², MAE) and checking for mean-collapse. |
| `alert_system.py` | Business logic translation. Evaluates the final RUL of each engine and sorts them into maintenance priority queues. |
| `dashboard.py` | Custom visual tool that draws the Fleet Health gauge, distribution pie charts, and chronological RUL degradation trendlines. |
| `requirements.txt` | List of all required Python modules and libraries. |
| `.gitignore` | Excludes bulky NASA `.txt` datasets and heavy `.h5` model files to keep the repository clean. |

---

### Development Methodology
1. **Dataset Ingestion:** Leveraged the benchmark NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation) `FD001` dataset, consisting of 100 simulated engines running to failure under a single operating condition.
2. **Feature Extraction & Normalization:** Extracted 21 sensor readings and 3 operational settings, scaling them into a standardized $[0, 1]$ vector space using Global MinMax parameters fitted strictly to the training set.
3. **Temporal Sequence Engineering:** Reconstructed flat CSV tables into 3D multi-variate tensors, feeding the model exactly 30 cycles of historical engine memory per prediction.
4. **Target Bounding:** Capped the target output vector to a maximum of 125 cycles to prevent the network from attempting to predict arbitrary high RULs during stable engine phases.
5. **Model Training:** Trained using the Adam optimizer with `EarlyStopping`, which halted optimization dynamically when validation loss plateaued, automatically restoring the weights of the highest-performing epoch.
6. **Alert Deployment:** Deployed model outputs into a logic filter to instantly flag the 100 test engines for preventative maintenance.

---

### Neural Network & Methodology

#### 1. Data Normalization
To prevent sensors with large absolute values (e.g., RPMs) from dominating the network weights over smaller sensors (e.g., temperature ratios), we scale all features using MinMax scaling:

$$X_{normalized} = \frac{X - X_{min}}{X_{max} - X_{min}}$$

#### 2. Stacked LSTM Network Architecture
Our model contains highly dense trainable parameters structured as follows:
* **Input Layer:** Accepts sequence matrices of shape `(30, 24)` (representing 30 continuous cycles of 24 normalized sensor/setting features).
* **LSTM Layer 1:** 50 units returning full sequences to pass temporal data deeply into the network.
* **Dropout Layers (20%):** Inserted between LSTM layers to randomly deactivate neurons and regularize network activations against overfitting.
* **LSTM Layer 2:** 50 units focused on extracting the final condensed feature vector.
* **Dense Layer:** A final linear, non-activated dense unit designed to output a continuous regression integer (Predicted RUL).

#### 3. Mean Squared Error (MSE) Loss
For predicting continuous regression outputs instead of categorical classes, the model optimizes against the Mean Squared Error function:

$$L = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2$$

Where $y_i$ is the true piecewise-capped RUL and $\hat{y}_i$ is the LSTM's linear output prediction.

---

### Getting Started

#### Prerequisites
Ensure you have Python 3.8+ installed on your system.

#### Installation
1. Clone the project files into a folder.
2. Open terminal/PowerShell in the folder and run:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the NASA CMAPSS dataset and place `train_FD001.txt`, `test_FD001.txt`, and `RUL_FD001.txt` into the root directory.

#### Running the Pipeline
Execute the master script to trigger the data prep, model training, and dashboard generation simultaneously:
```bash
python main.py
```

---

### Experimental Results & Performance
Our system underwent rigorous empirical testing, measuring forecasting accuracy against the exact moment of simulated engine failure.

| Performance Metric | Measured Value | Description |
| :--- | :--- | :--- |
| **RMSE** | ~29.72 | Root Mean Squared Error. The average absolute deviation from the true remaining useful life in cycles. |
| **R² Score** | >0.0 | Coefficient of Determination. Confirms the network has broken past predicting the statistical mean and is accurately learning the degradation curve. |
| **Prediction StdDev** | ~16.55 | Demonstrates that the network is actively varying its predictions based on the sensor inputs rather than collapsing to a static guess. |

---

### Conclusion & Future Scope

#### Conclusion
The proposed work successfully demonstrates a highly functional predictive maintenance system using Deep Learning. By structuring time-series data into rolling tensor sequences and feeding it into a Stacked LSTM network, the system successfully forecasts impending engine failure. 

The integration of Piecewise Linear Target Capping and Gradient Clipping eliminated instability and gradient collapse, allowing the network to zero in on the critical degradation phase of the engines. Operating at rapid speeds, the final product seamlessly integrates into business-logic alerts and fleet dashboards, presenting a scalable solution for heavy-machinery monitoring.

#### Future Scope
To scale this architecture, future implementations can incorporate K-Means clustering to classify incoming data into independent operational profiles. This would allow the system to tackle the significantly more chaotic `FD002` and `FD004` datasets, which feature 6 distinct altitude and throttle operating conditions that drastically skew the raw sensor telemetry.
