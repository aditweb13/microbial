"""
optimizer.py
============
Phase 3 – Fermentation Optimization Engine

Uses scipy's Differential Evolution and a Randomized Search fallback to find
the input conditions that MAXIMIZE predicted fermentation Yield, as determined
by the trained Random Forest pipeline stored in fermentation_pipeline.pkl.

Usage (CLI):
    python optimizer.py
    python optimizer.py --target ethanol --strain "Clostridium strain AK1"

Outputs:
    - Prints baseline vs. optimised parameter table to stdout
    - Saves optimisation_results.csv with the full results
    - Saves optimisation_history.png with a convergence plot
"""

import argparse
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe for servers)
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution

warnings.filterwarnings("ignore")

PIPELINE_PATH = Path(__file__).parent / "fermentation_pipeline.pkl"

# ─────────────────────────────────────────────────────────────────────────────
# Feature Search Space
# Each numerical feature gets (lower_bound, upper_bound) derived from
# realistic fermentation literature and the dataset's own distribution.
# ─────────────────────────────────────────────────────────────────────────────
NUMERIC_BOUNDS = {
    "Temperature":           (20.0,  45.0),   # °C
    "pH":                    (4.0,   8.0),    # unitless
    "Fermentation_Time":     (12.0,  168.0),  # hours
    "Sugar_Concentration":   (10.0,  300.0),  # g/L
    "Agitation_Speed":       (50.0,  400.0),  # rpm
    "Aeration":              (0.0,   2.0),    # vvm
    "Biomass":               (0.1,   20.0),   # g/L
    "PCV":                   (1.0,   50.0),   # %
    "Phosphate":             (0.0,   5.0),    # g/L
    "L_G_Ratio":             (0.1,   5.0),    # unitless
    "Glucose_Concentration": (5.0,   200.0),  # g/L
    "OD600":                 (0.05,  5.0),    # unitless
    # Macro-economic / country-level features — kept near median during optimisation
    "GDP_per_capita":        (500.0, 80000.0),
    "Sugarcane_Production":  (0.0,   900000000.0),
    "Corn_Production":       (0.0,   400000000.0),
    "Cassava_Production":    (0.0,   50000000.0),
    "Total_Biofuel_Feedstock_Production": (0.0, 1200000000.0),
    "Year":                  (2000.0, 2030.0),
    "Policy_Support_Index":  (0.0,   10.0),
    "CO2_Emissions":         (0.0,   1e10),
    "Trade_Balance":         (-1e9,  1e9),
    "Production_Cost_Index": (0.0,   200.0),
    "Market_Demand_Index":   (0.0,   10.0),
    "Ethanol_Yield":         (0.0,   1000.0),
    "Investment_in_Bioenergy":(0.0,  1e12),
    "Renewable_Energy_Share":(0.0,   100.0),
    "Subsidy_Amount":        (0.0,   1e9),
}


def load_pipeline(path: Path) -> dict:
    """Load the serialised sklearn pipeline created in train_models.py."""
    if not path.exists():
        raise FileNotFoundError(
            f"Pipeline file not found: {path}\n"
            "Please run train_models.py first."
        )
    with open(path, "rb") as fh:
        return pickle.load(fh)


def encode_categoricals(row: dict, mappings: dict, cat_cols: list) -> dict:
    """Apply label-encoding mappings to categorical columns."""
    encoded = row.copy()
    for col in cat_cols:
        if col in encoded and col in mappings:
            val = encoded[col]
            if isinstance(val, str):
                encoded[col] = mappings[col].get(val, 0)
    return encoded


def build_feature_vector(
    num_values: np.ndarray,
    num_cols_order: list,
    fixed_cats: dict,
    feature_columns: list,
) -> np.ndarray:
    """
    Construct a 1-D feature vector in the exact column order used during training.

    Parameters
    ----------
    num_values      : array of optimisable numerical values
    num_cols_order  : list of numerical column names matching num_values order
    fixed_cats      : dict of already-encoded categorical values
    feature_columns : full column order from the pipeline
    """
    row = dict(zip(num_cols_order, num_values))
    row.update(fixed_cats)
    # Fill any missing columns with 0
    vec = np.array([row.get(c, 0.0) for c in feature_columns], dtype=float)
    return vec


def make_objective(pipeline: dict, fixed_cats: dict, opt_num_cols: list, history: list):
    """
    Return a scalar objective function for scipy's differential_evolution.
    Minimises NEGATIVE yield (i.e., maximises yield).
    """
    model   = pipeline["model"]
    scaler  = pipeline["scaler"]
    feat_cols = pipeline["feature_columns"]

    def objective(x: np.ndarray) -> float:
        vec = build_feature_vector(x, opt_num_cols, fixed_cats, feat_cols)
        scaled = scaler.transform(vec.reshape(1, -1))
        pred   = model.predict(scaled)[0]
        history.append(-pred)          # track minimised value
        return -pred                   # negate for maximisation

    return objective


def randomised_search(
    objective_fn,
    bounds: list,
    n_iter: int = 5000,
    seed: int = 42,
) -> tuple:
    """
    Fallback / supplemental random search over the parameter space.
    Returns (best_x, best_neg_yield).
    """
    rng = np.random.default_rng(seed)
    best_x   = None
    best_val = np.inf

    low  = np.array([b[0] for b in bounds])
    high = np.array([b[1] for b in bounds])

    for _ in range(n_iter):
        x   = rng.uniform(low, high)
        val = objective_fn(x)
        if val < best_val:
            best_val = val
            best_x   = x.copy()

    return best_x, best_val


def print_comparison_table(
    baseline: dict,
    optimised: dict,
    opt_cols: list,
    units: dict,
):
    """Print a neat before/after comparison table."""
    header = f"\n{'Parameter':<35} {'Baseline':>15} {'Optimised':>15} {'Unit':<15}"
    sep    = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for col in opt_cols:
        bv  = baseline.get(col, "N/A")
        ov  = optimised.get(col, "N/A")
        unit = units.get(col, "")
        bv_s = f"{bv:.4f}" if isinstance(bv, float) else str(bv)
        ov_s = f"{ov:.4f}" if isinstance(ov, float) else str(ov)
        print(f"{col:<35} {bv_s:>15} {ov_s:>15} {unit:<15}")
    print(sep)


UNITS = {
    "Temperature":           "°C",
    "pH":                    "",
    "Fermentation_Time":     "hours",
    "Sugar_Concentration":   "g/L",
    "Agitation_Speed":       "rpm",
    "Aeration":              "vvm",
    "Biomass":               "g/L",
    "PCV":                   "%",
    "Phosphate":             "g/L",
    "L_G_Ratio":             "",
    "Glucose_Concentration": "g/L",
    "OD600":                 "",
}


def run_optimisation(
    pipeline: dict,
    strain: str = "Clostridium strain AK1",
    substrate: str = "D-Glucose",
    source: str = "Clostridium_AK1_Lab_Data",
    country: str = "India",
    popsize: int = 15,
    maxiter: int = 200,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """
    Core optimisation routine.

    Returns a dict with:
        baseline_yield, optimised_yield, improvement_pct,
        baseline_params, optimised_params, history
    """
    mappings      = pipeline["mappings"]
    cat_cols      = pipeline["cat_cols"]
    feature_cols  = pipeline["feature_columns"]
    medians       = pipeline["medians"]
    model         = pipeline["model"]
    scaler        = pipeline["scaler"]

    # ── Fixed categorical values ────────────────────────────────────────────
    fixed_cats = encode_categoricals(
        {
            "Substrate":       substrate,
            "Microbial_Strain": strain,
            "Dataset_Source":  source,
            "Country":         country,
        },
        mappings,
        cat_cols,
    )

    # ── Numerical columns we will optimise ──────────────────────────────────
    opt_num_cols = [
        c for c in feature_cols
        if c not in cat_cols and c in NUMERIC_BOUNDS
    ]

    bounds = [NUMERIC_BOUNDS[c] for c in opt_num_cols]

    # ── Baseline: use dataset medians as the "current" conditions ───────────
    baseline_vec_num = np.array(
        [medians.get(c, (NUMERIC_BOUNDS[c][0] + NUMERIC_BOUNDS[c][1]) / 2)
         for c in opt_num_cols]
    )
    baseline_full = build_feature_vector(
        baseline_vec_num, opt_num_cols, fixed_cats, feature_cols
    )
    baseline_scaled  = scaler.transform(baseline_full.reshape(1, -1))
    baseline_yield   = model.predict(baseline_scaled)[0]

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Fermentation Optimisation Engine")
        print(f"  Strain   : {strain}")
        print(f"  Substrate: {substrate}")
        print(f"  Country  : {country}")
        print(f"{'='*60}")
        print(f"\nBaseline Yield (median conditions): {baseline_yield:.4f}")
        print(f"Running Differential Evolution ({maxiter} iterations, popsize={popsize})...")

    history = []
    objective = make_objective(pipeline, fixed_cats, opt_num_cols, history)

    # ── Differential Evolution ───────────────────────────────────────────────
    result = differential_evolution(
        objective,
        bounds=bounds,
        seed=seed,
        popsize=popsize,
        maxiter=maxiter,
        tol=1e-6,
        mutation=(0.5, 1.5),
        recombination=0.9,
        strategy="best1bin",
        updating="deferred",
        workers=1,
        polish=True,
    )

    best_x_de    = result.x
    best_yield_de = -result.fun

    # ── Randomised Search (cross-check) ─────────────────────────────────────
    if verbose:
        print("Running Randomised Search cross-check (5 000 samples)...")
    best_x_rs, best_neg_rs = randomised_search(objective, bounds, n_iter=5000, seed=seed + 1)
    best_yield_rs = -best_neg_rs

    # Pick whichever found the higher yield
    if best_yield_rs > best_yield_de:
        best_x     = best_x_rs
        best_yield = best_yield_rs
        method_used = "Randomised Search"
    else:
        best_x     = best_x_de
        best_yield = best_yield_de
        method_used = "Differential Evolution"

    if verbose:
        print(f"\nBest method : {method_used}")
        print(f"Optimised Yield: {best_yield:.4f}  (baseline: {baseline_yield:.4f})")

    # ── Build result dicts ───────────────────────────────────────────────────
    baseline_params   = dict(zip(opt_num_cols, baseline_vec_num.tolist()))
    optimised_params  = dict(zip(opt_num_cols, best_x.tolist()))
    improvement_pct   = ((best_yield - baseline_yield) / (abs(baseline_yield) + 1e-9)) * 100

    if verbose:
        print(f"Improvement : {improvement_pct:+.2f}%")
        # Only print the core fermentation params (not macro-economic)
        core_cols = [c for c in opt_num_cols if c in UNITS]
        print_comparison_table(baseline_params, optimised_params, core_cols, UNITS)

    return {
        "baseline_yield":    baseline_yield,
        "optimised_yield":   best_yield,
        "improvement_pct":   improvement_pct,
        "method_used":       method_used,
        "baseline_params":   baseline_params,
        "optimised_params":  optimised_params,
        "history":           history,
        "opt_num_cols":      opt_num_cols,
    }


def save_results(results: dict, out_dir: Path = Path(".")):
    """Save CSV results and convergence plot."""

    # ── CSV ──────────────────────────────────────────────────────────────────
    rows = []
    for col in results["opt_num_cols"]:
        rows.append({
            "Parameter":       col,
            "Baseline":        results["baseline_params"].get(col, ""),
            "Optimised":       results["optimised_params"].get(col, ""),
            "Unit":            UNITS.get(col, ""),
        })
    df_out = pd.DataFrame(rows)
    df_out.loc[len(df_out)] = {
        "Parameter": "Predicted_Yield",
        "Baseline":  round(results["baseline_yield"], 6),
        "Optimised": round(results["optimised_yield"], 6),
        "Unit":      "g/L or %",
    }
    csv_path = out_dir / "optimisation_results.csv"
    df_out.to_csv(csv_path, index=False)
    print(f"\nSaved results to '{csv_path}'")

    # ── Convergence plot ────────────────────────────────────────────────────
    history = results["history"]
    if len(history) > 0:
        # Cumulative best (minimised negative yield → maximised yield)
        cumulative_best = np.minimum.accumulate(history)
        cumulative_yield = -cumulative_best

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Fermentation Optimisation – Convergence", fontsize=14, fontweight="bold")

        # Left: raw history
        axes[0].plot([-v for v in history], alpha=0.4, color="steelblue", linewidth=0.8)
        axes[0].set_title("All Evaluations")
        axes[0].set_xlabel("Evaluation #")
        axes[0].set_ylabel("Predicted Yield")

        # Right: cumulative best
        axes[1].plot(cumulative_yield, color="darkorange", linewidth=2)
        axes[1].axhline(
            results["baseline_yield"], linestyle="--",
            color="grey", label=f"Baseline ({results['baseline_yield']:.4f})"
        )
        axes[1].axhline(
            results["optimised_yield"], linestyle="-.",
            color="green", label=f"Optimised ({results['optimised_yield']:.4f})"
        )
        axes[1].set_title("Cumulative Best Yield")
        axes[1].set_xlabel("Evaluation #")
        axes[1].set_ylabel("Predicted Yield")
        axes[1].legend()

        plt.tight_layout()
        plot_path = out_dir / "optimisation_history.png"
        plt.savefig(plot_path, dpi=200)
        plt.close()
        print(f"Saved convergence plot to '{plot_path}'")


def main():
    parser = argparse.ArgumentParser(
        description="Fermentation Yield Optimisation Engine"
    )
    parser.add_argument(
        "--strain", default="Clostridium strain AK1",
        help="Microbial strain to optimise for"
    )
    parser.add_argument(
        "--substrate", default="D-Glucose",
        choices=["Control", "D-Glucose", "L-Fucose", "L-Rhamnose"],
        help="Fermentation substrate"
    )
    parser.add_argument(
        "--country", default="India",
        choices=["Brazil", "China", "Germany", "India", "USA"],
        help="Country context for macro-economic features"
    )
    parser.add_argument(
        "--popsize", type=int, default=15,
        help="Differential Evolution population size multiplier"
    )
    parser.add_argument(
        "--maxiter", type=int, default=200,
        help="Max iterations for Differential Evolution"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    pipeline = load_pipeline(PIPELINE_PATH)

    results = run_optimisation(
        pipeline=pipeline,
        strain=args.strain,
        substrate=args.substrate,
        country=args.country,
        popsize=args.popsize,
        maxiter=args.maxiter,
        seed=args.seed,
        verbose=True,
    )

    out_dir = Path(__file__).parent
    save_results(results, out_dir=out_dir)

    print("\n[DONE] Optimisation complete.")
    print(f"  Baseline Yield   : {results['baseline_yield']:.4f}")
    print(f"  Optimised Yield  : {results['optimised_yield']:.4f}")
    print(f"  Improvement      : {results['improvement_pct']:+.2f}%")
    print(f"  Method           : {results['method_used']}")


if __name__ == "__main__":
    main()
