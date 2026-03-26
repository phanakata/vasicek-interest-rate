"""
Vasicek Interest Rate Model — Main Script

Run this to:
    1. Pull Federal Funds Rate data from FRED
    2. Fit the Vasicek model (OLS then MLE)
    3. Simulate future interest rate paths
    4. Backtest: train on 80% of history, forecast 20%, compare to actuals

Usage:
    pip install -r requirements.txt
    python main.py
"""

import sys
import os
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — saves plots to files instead of opening windows
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.fetch import fetch_fed_funds_rate
from model.vasicek import VasicekModel
from backtest.backtest import run_backtest, plot_backtest, plot_simulation
from sanity_check import run_sanity_check

# ── Config ────────────────────────────────────────────────────────────────────
DATA_START   = "1994-01-01"   # start of data pull (earlier = more history)
TRAIN_FRAC   = 0.80           # 80% train / 20% test split
N_PATHS      = 5000           # Monte Carlo paths for backtest
FIT_METHOD   = "mle"          # 'mle' (recommended) or 'ols'
HORIZON_YRS  = 5.0            # years to simulate forward from current rate
RANDOM_SEED  = 42

np.random.seed(RANDOM_SEED)

# ── Step 0: Sanity check on synthetic data ────────────────────────────────────
print("=" * 60)
print("STEP 0 — Sanity Check (Synthetic Data)")
print("=" * 60)
run_sanity_check()

# ── Step 1: Pull data ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 1 — Fetch Data")
print("=" * 60)

df = fetch_fed_funds_rate(start=DATA_START)
print(df.tail())

# ── Step 2: Fit the Vasicek model ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2 — Fit Vasicek Model (on full dataset)")
print("=" * 60)

rates = df["rate"].values
model = VasicekModel()

print("\n--- OLS fit ---")
model.fit_ols(rates)
model.summary()

print("\n--- MLE fit (more accurate) ---")
model.fit_mle(rates)
model.summary()

# ── Step 3: Simulate future paths ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3 — Simulate Future Interest Rate Paths")
print("=" * 60)

r_current = rates[-1]
print(f"Current rate: {r_current * 100:.2f}%")
print(f"Simulating {HORIZON_YRS:.0f} years forward with {N_PATHS} paths ...\n")

plot_simulation(
    model,
    r0=r_current,
    horizon_years=HORIZON_YRS,
    n_paths=200,                     # fewer paths for visual clarity
    save_path="simulation.png",
)

# ── Step 4: Backtest ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4 — Backtest")
print("=" * 60)

backtest_model = VasicekModel()      # fresh unfitted model
results = run_backtest(
    model=backtest_model,
    df=df,
    train_frac=TRAIN_FRAC,
    n_paths=N_PATHS,
    fit_method=FIT_METHOD,
)

plot_backtest(results, save_path="backtest.png")

print("\nDone. Plots saved to simulation.png and backtest.png")
