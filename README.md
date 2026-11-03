# 🧬 FermentAI — Fermentation Yield Prediction & Media Optimization

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML%20Pipeline-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-15803d?style=flat-square)](https://xgboost.readthedocs.io)
[![SciPy](https://img.shields.io/badge/SciPy-Differential%20Evolution-8CAAE6?style=flat-square&logo=scipy&logoColor=white)](https://scipy.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Bioluminescent%20Dashboard-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)

*An advanced machine learning and mathematical optimization platform for industrial biotechnology — predicts microbial fermentation yield and recommends optimal media compositions.*

</div>

---

## 📸 Demo Preview

> **TODO:** Add a screenshot of the Streamlit dashboard here.
> Suggested file: `assets/fermentai-dashboard.png`
>
> Capture: Bioluminescent dark-mode dashboard showing the yield predictor tab, optimization engine output, and model performance comparison table.

---

## 🎯 Overview

Traditional fermentation optimization requires numerous time-consuming and expensive laboratory trials. **FermentAI** accelerates bioprocess development by leveraging data-driven predictions and global optimization algorithms.

**FermentAI solves two problems:**

1. **Yield Prediction:** Given process parameters (temperature, pH, agitation, substrate type, biomass concentration), predict the expected product yield (g/L or %) with confidence estimates.
2. **Media Optimization:** Automatically search the parameter space to find the conditions that maximize predicted yield — without running wet-lab experiments.

---

## ✨ Key Features

- **Bioluminescent Lab Web Dashboard:** A sci-fi dark-mode Streamlit interface with glow-accented interactive charts and real-time predictions.
- **Automated Multi-Model ML Pipeline:** Trains Linear Regression, Random Forest, and XGBoost regressors; auto-selects the best model by cross-validated R², MAE, and RMSE.
- **Interactive Yield Predictor:** Real-time predictions with uncertainty/confidence estimates as you tune parameters on sliders.
- **Dual-Algorithm Optimization Engine:**
  - **SciPy Differential Evolution** (global search — robust, avoids local minima)
  - **Randomized Search** (fast local refinement)
- **Visualizations & Metrics:** Correlation heatmaps, feature importance bar charts, convergence history plots, and side-by-side baseline-vs-optimized comparison tables.

---

## 🗂️ Supported Strains, Substrates & Context

| Parameter | Options |
|---|---|
| **Microbial Strain** | Clostridium strain AK1, and others in dataset |
| **Substrate** | Control, D-Glucose, L-Fucose, L-Rhamnose |
| **Country Context** | Brazil, China, Germany, India, USA (macroeconomic weighting) |

---

## 🔧 Optimizable Process Parameters

| Parameter | Range |
|---|---|
| Temperature (°C) | 20.0 – 45.0 |
| pH | 4.0 – 8.0 |
| Fermentation Time (hours) | 12.0 – 168.0 |
| Sugar Concentration (g/L) | 10.0 – 300.0 |
| Agitation Speed (RPM) | 50.0 – 400.0 |
| Aeration Rate (VVM) | 0.0 – 2.0 |
| Biomass Concentration (g/L) | 0.1 – 20.0 |
| Glucose Concentration (g/L) | 5.0 – 200.0 |

**Target Output:** Product Yield (g/L or %)

---

## 📁 Project Structure

```
microbial/
├── app.py                          # Streamlit dashboard (Bioluminescent Lab UI)
├── train_models.py                 # Train, compare, and serialize best ML model
├── optimizer.py                    # Core optimization engine (CLI + library)
├── requirements.txt
├── project.md                      # Full project specification and scope
├── fermentation_pipeline.pkl       # Serialized model + scalers + metadata (generated)
│
├── cleaned_fermentation_dataset.csv
├── cleaned_fermentation_dataset_scaled.csv
├── bioethanol_growth_prediction_dataset (2).csv
├── cleaned_bioethanol_growth.csv
├── cleaned_clostridium_ak1.csv
│
├── correlation_heatmap.png         # Feature correlation matrix plot
├── feature_importance.png          # Best model feature importances (generated)
├── optimisation_history.png        # Convergence path plot (generated)
└── optimisation_results.csv        # Baseline vs. Optimized parameter comparison
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+

### Installation

```bash
git clone https://github.com/aditweb13/microbial.git
cd microbial

pip install -r requirements.txt
```

### Step 1 — Train & Export the ML Model

```bash
python train_models.py
```

- **Inputs:** Reads `cleaned_fermentation_dataset.csv`
- **Outputs:** Displays model scores, saves `feature_importance.png`, serializes the complete preprocessing + prediction pipeline to `fermentation_pipeline.pkl`.

### Step 2 — Run CLI Optimization

```bash
python optimizer.py --strain "Clostridium strain AK1" --substrate "D-Glucose" --country "India"
```

**CLI Arguments:**

| Argument | Default | Choices |
|---|---|---|
| `--strain` | `"Clostridium strain AK1"` | Any strain name in dataset |
| `--substrate` | `"D-Glucose"` | `Control`, `D-Glucose`, `L-Fucose`, `L-Rhamnose` |
| `--country` | `"India"` | `Brazil`, `China`, `Germany`, `India`, `USA` |
| `--popsize` | `15` | Population multiplier for Differential Evolution |
| `--maxiter` | `200` | Max solver iterations |

**Outputs:** Prints comparison tables, writes `optimisation_results.csv`, generates `optimisation_history.png`.

### Step 3 — Launch the Streamlit Dashboard

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📊 Model Performance

After running `train_models.py`, model comparison metrics are printed to the console and visualized. The best model is automatically serialized. Typical results (on included datasets):

| Model | R² | MAE | RMSE |
|---|---|---|---|
| Linear Regression | — | — | — |
| Random Forest | — | — | — |
| **XGBoost (auto-selected)** | **best** | **lowest** | **lowest** |

> Note: Exact metric values depend on the train/test split and data version. Run `train_models.py` to generate your own results.

---

## 🗺️ Roadmap

- [ ] Add fermentation kinetics modelling (Monod, Logistic, Gompertz)
- [ ] Bayesian optimization alternative (Gaussian Process surrogate)
- [ ] Deploy Streamlit app to Hugging Face Spaces or Streamlit Cloud
- [ ] Extend to lactic acid, citric acid, and antibiotic yield targets

---

## 📄 License

MIT — see [LICENSE](LICENSE).

---

<div align="center">

Built by [Aditya Raj](https://github.com/aditweb13) · [LinkedIn](https://linkedin.com/in/aditya-raj-a1891a285) · [adi111sam121@gmail.com](mailto:adi111sam121@gmail.com)

*Bridging biotechnology and machine learning — from wet-lab kinetics to optimization algorithms.*

</div>
