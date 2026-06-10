# FermentAI: AI-Based Fermentation Yield Prediction & Media Optimization

FermentAI is an advanced machine learning and mathematical optimization platform built for industrial biotechnology. The system predicts fermentation yield (e.g., ethanol, citric acid, lactic acid, biomass) based on process parameters and uses optimization techniques (Differential Evolution and Randomized Search) to recommend optimal operating conditions for maximizing productivity.

---

## 🧬 Project Overview
Traditional fermentation optimization requires numerous time-consuming and expensive laboratory trials. **FermentAI** accelerates this process by leveraging data-driven predictions and global optimization algorithms.

### Key Features
* **Bioluminescent Lab Web Dashboard**: A Streamlit interface designed with a dark, glow-accented sci-fi aesthetic.
* **Multi-Model ML Pipeline**: Automated training and evaluation of Linear Regression, Random Forest, and XGBoost regressors, selecting the best model automatically.
* **Yield Predictor**: Interactive parameters with real-time predictions and uncertainty/confidence estimates.
* **Dual-Algorithm Optimization Engine**: Combines SciPy's global **Differential Evolution** algorithm and **Randomized Search** to identify peak parameters.
* **Visualizations & Metrics**: R², MAE, RMSE performance tables, feature importance metrics, and interactive correlation charts.

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have **Python 3.8+** installed.

### 2. Installation
Clone or navigate to the project directory and install the required dependencies:
```bash
pip install -r requirements.txt
```

### 3. Usage

#### A. Train and Export the ML Model
Run the model development script to process dataset files, build models, compare metrics, and export the best-performing pipeline:
```bash
python train_models.py
```
* **Inputs**: Reads `cleaned_fermentation_dataset.csv`
* **Outputs**: Displays model scores, saves `feature_importance.png`, and serializes the complete preprocessing/prediction pipeline to `fermentation_pipeline.pkl`.

#### B. Command-Line Optimization
To run the optimization engine in the command line for a specific strain and substrate configuration:
```bash
python optimizer.py --strain "Clostridium strain AK1" --substrate "D-Glucose" --country "India"
```
* **CLI Arguments**:
  * `--strain` (default: `"Clostridium strain AK1"`): Microbial strain to optimize.
  * `--substrate` (default: `"D-Glucose"`, choices: `Control`, `D-Glucose`, `L-Fucose`, `L-Rhamnose`): Carbon substrate.
  * `--country` (default: `"India"`, choices: `Brazil`, `China`, `Germany`, `India`, `USA`): Macroeconomic context.
  * `--popsize` (default: `15`): Population multiplier for Differential Evolution.
  * `--maxiter` (default: `200`): Max solver iterations.
* **Outputs**: Prints comparison tables, writes `optimisation_results.csv`, and outputs `optimisation_history.png` depicting convergence.

#### C. Launch the Streamlit Web Application
Start the interactive dashboard interface:
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

---

## 📂 Project Structure
```
├── app.py                      # Main Streamlit dashboard script (Bioluminescent Lab interface)
├── train_models.py             # Script to train, compare, and serialize the best ML model
├── optimizer.py                # Core Optimization Engine (CLI & library functions)
├── requirements.txt            # List of Python library dependencies
├── project.md                  # Project specification, scope, and parameters
├── fix_deprecated.py           # Minor utility script for app package adjustments
│
├── fermentation_pipeline.pkl   # Serialized model, scalers, mappings, and metadata (generated)
│
├── cleaned_fermentation_dataset.csv         # Main cleaned fermentation dataset
├── cleaned_fermentation_dataset_scaled.csv  # Pre-scaled training dataset
├── bioethanol_growth_prediction_dataset (2).csv # Raw dataset sources
├── cleaned_bioethanol_growth.csv            # Cleaned growth prediction dataset
├── cleaned_clostridium_ak1.csv              # Lab Clostridium dataset
│
├── correlation_heatmap.png      # Matrix plot of feature correlations
├── feature_importance.png       # Importance bars from the best ML model (generated)
├── optimisation_history.png     # Convergence path plot (generated)
├── optimisation_results.csv     # Baseline vs. Optimized parameter comparison (generated)
│
├── FermentAI_IEEE_Report.docx   # Detailed project final IEEE report
└── FermentAI_IEEE_Report (1).docx
```

---

## 📊 System Parameters & Features

### Process (Optimizable) Inputs
* **Temperature** (°C) - Range: `20.0 - 45.0`
* **pH** - Range: `4.0 - 8.0`
* **Fermentation Time** (hours) - Range: `12.0 - 168.0`
* **Sugar Concentration** (g/L) - Range: `10.0 - 300.0`
* **Agitation Speed** (RPM) - Range: `50.0 - 400.0`
* **Aeration Rate** (VVM) - Range: `0.0 - 2.0`
* **Biomass Concentration** (g/L) - Range: `0.1 - 20.0`
* **Glucose Concentration** (g/L) - Range: `5.0 - 200.0`
* **Packed Cell Volume (PCV)** (%)
* **OD600** (optical density)

### Target Output
* **Yield**: Product concentration (g/L or %).
