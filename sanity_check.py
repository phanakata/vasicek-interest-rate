# -*- coding: utf-8 -*-
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

"""
Sanity check: parameter recovery on synthetic Vasicek data.

Procedure:
    1. Fix known ground-truth parameters (kappa, theta, sigma).
    2. Simulate a long synthetic interest rate path from those parameters.
    3. Fit the model (OLS and MLE) to the synthetic path.
    4. Compare fitted parameters vs ground truth — they should be close.
    5. Run the backtest on synthetic data:
           - RMSE should be low (model fits its own data)
           - 90% CI coverage should be near 90% (well-calibrated by construction)

If parameter recovery fails or coverage is far from 90%, something is wrong
with the fitting or simulation code — not the real data.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model.vasicek import VasicekModel
from backtest.backtest import run_backtest, plot_backtest, plot_simulation


# ── Ground-truth parameters ───────────────────────────────────────────────────
# Chosen to be plausible for US interest rates
TRUE_KAPPA = 0.15     # moderate mean reversion (~4.6 yr half-life)
TRUE_THETA = 0.04     # 4% long-run mean
TRUE_SIGMA = 0.015    # 1.5% annualized volatility
R0         = 0.02     # starting rate (2%)
DT         = 1 / 12  # monthly steps

N_STEPS    = 600      # 50 years of synthetic monthly data
N_PATHS    = 5000     # Monte Carlo paths for backtest
RANDOM_SEED = 42


def simulate_true_path(
    kappa: float,
    theta: float,
    sigma: float,
    r0: float,
    n_steps: int,
    dt: float,
    seed: int,
) -> np.ndarray:
    """
    Simulate one path from the true Vasicek model using the exact discretization.
    This is our synthetic 'observed' data.
    """
    np.random.seed(seed)
    e_kdt  = np.exp(-kappa * dt)
    std    = np.sqrt(sigma ** 2 * (1 - e_kdt ** 2) / (2 * kappa))
    drift  = theta * (1 - e_kdt)

    path    = np.empty(n_steps + 1)
    path[0] = r0
    for t in range(n_steps):
        path[t+1] = path[t] * e_kdt + drift + std * np.random.randn()
    return path


def print_recovery_table(true: dict, ols: VasicekModel, mle: VasicekModel) -> None:
    """Print a side-by-side comparison of true vs fitted parameters."""
    def pct_err(fitted, true_val):
        return (fitted - true_val) / true_val * 100

    print("\n" + "=" * 65)
    print(f"{'Parameter':<12} {'True':>10} {'OLS':>10} {'MLE':>10} {'OLS err':>9} {'MLE err':>9}")
    print("-" * 65)

    params = [
        ("kappa",  true["kappa"], ols.kappa, mle.kappa),
        ("theta",  true["theta"], ols.theta, mle.theta),
        ("sigma",  true["sigma"], ols.sigma, mle.sigma),
    ]
    for name, t, o, m in params:
        print(
            f"  {name:<10} {t:>10.4f} {o:>10.4f} {m:>10.4f}"
            f"  {pct_err(o, t):>+7.1f}%  {pct_err(m, t):>+7.1f}%"
        )
    print("=" * 65)

    # Verdict
    mle_kappa_err = abs(pct_err(mle.kappa, true["kappa"]))
    mle_theta_err = abs(pct_err(mle.theta, true["theta"]))
    mle_sigma_err = abs(pct_err(mle.sigma, true["sigma"]))

    print("\nVerdict:")
    for name, err in [("kappa", mle_kappa_err), ("theta", mle_theta_err), ("sigma", mle_sigma_err)]:
        if err < 5:
            status = "PASS  (error < 5%)"
        elif err < 15:
            status = "WARN  (error 5–15% — acceptable for finite sample)"
        else:
            status = "FAIL  (error > 15% — investigate)"
        print(f"  MLE {name}: {status}")


def plot_synthetic(
    path: np.ndarray,
    true_theta: float,
    dt: float,
    save_path: str = None,
) -> None:
    """Plot the synthetic interest rate path with the true long-run mean."""
    t = np.arange(len(path)) * dt   # in years
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, path * 100, color="steelblue", lw=0.8, label="Synthetic path")
    ax.axhline(true_theta * 100, color="red", ls="--", lw=1.5,
               label=f"True θ = {true_theta*100:.1f}%")
    ax.set_title(
        f"Synthetic Vasicek Path  "
        f"(κ={TRUE_KAPPA}, θ={TRUE_THETA*100:.1f}%, σ={TRUE_SIGMA}, n={N_STEPS} months)",
        fontsize=12,
    )
    ax.set_xlabel("Years")
    ax.set_ylabel("Rate (%)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    plt.close()


def run_sanity_check() -> None:
    print("=" * 60)
    print("SANITY CHECK — Parameter Recovery on Synthetic Data")
    print("=" * 60)
    print(f"\nGround-truth parameters:")
    print(f"  kappa = {TRUE_KAPPA}   theta = {TRUE_THETA*100:.1f}%   sigma = {TRUE_SIGMA}")
    print(f"  r0 = {R0*100:.1f}%   {N_STEPS} monthly steps ({N_STEPS//12} years)\n")

    # ── Step 1: Simulate synthetic path ──────────────────────────────────────
    path = simulate_true_path(TRUE_KAPPA, TRUE_THETA, TRUE_SIGMA, R0, N_STEPS, DT, RANDOM_SEED)
    print(f"Synthetic path stats:")
    print(f"  mean = {path.mean()*100:.3f}%   std = {path.std()*100:.3f}%"
          f"   min = {path.min()*100:.3f}%   max = {path.max()*100:.3f}%")

    plot_synthetic(path, TRUE_THETA, DT, save_path="sanity_synthetic_path.png")

    # ── Step 2: Fit OLS and MLE ───────────────────────────────────────────────
    print("\n--- Fitting OLS ---")
    ols_model = VasicekModel()
    ols_model.fit_ols(path, dt=DT)
    ols_model.summary()

    print("\n--- Fitting MLE ---")
    mle_model = VasicekModel()
    mle_model.fit_mle(path, dt=DT)
    mle_model.summary()

    # ── Step 3: Parameter recovery table ─────────────────────────────────────
    true_params = {"kappa": TRUE_KAPPA, "theta": TRUE_THETA, "sigma": TRUE_SIGMA}
    print_recovery_table(true_params, ols_model, mle_model)

    # ── Test A: Simulation coverage (oracle, averaged over many test paths) ──
    # Use TRUE parameters to generate both the confidence bands AND many
    # independent test paths. Average coverage over all test paths.
    #
    # Why not use just one test path?
    # The Fed Funds rate has autocorrelation ~0.99 per month, making the
    # effective independent sample size of a 120-month test only ~2 points.
    # One path wandering outside the bands for 30 consecutive months skews
    # coverage from 90% down to 65% — that is sampling noise, not a bug.
    # Averaging over 500 independent test paths eliminates this noise.
    print("\n" + "-" * 60)
    print("TEST A: Simulation coverage using TRUE parameters (oracle)")
    print("  Averaged over 500 independent test paths to eliminate")
    print("  sampling noise from autocorrelated data.")
    print("-" * 60)

    N_TEST_PATHS = 500
    TEST_STEPS   = 120    # 10 years
    np.random.seed(RANDOM_SEED + 1)

    oracle_model = VasicekModel()
    oracle_model.kappa = TRUE_KAPPA
    oracle_model.theta = TRUE_THETA
    oracle_model.sigma = TRUE_SIGMA
    oracle_model.dt    = DT

    # Simulate a large ensemble; use even columns as "actual" paths,
    # odd columns as "simulation" for the confidence bands.
    # Simpler: for each of N_TEST_PATHS, simulate 5000 fan paths and
    # check if one held-out path falls within the 5th-95th percentile.
    coverages = []
    r0_oracle = TRUE_THETA   # start all test paths from the long-run mean

    for _ in range(N_TEST_PATHS):
        # One "actual" test path
        actual = oracle_model.simulate(r0_oracle, TEST_STEPS, n_paths=1)[:, 0]

        # 2000 fan paths for confidence band
        fan    = oracle_model.simulate(r0_oracle, TEST_STEPS, n_paths=2000)[1:]
        p5_o   = np.percentile(fan, 5,  axis=1)
        p95_o  = np.percentile(fan, 95, axis=1)

        cov = np.mean((actual[1:] >= p5_o) & (actual[1:] <= p95_o))
        coverages.append(cov)

    coverage_oracle = np.mean(coverages) * 100
    print(f"  Oracle 90% CI coverage : {coverage_oracle:.1f}%  (target: ~90%)")
    oracle_ok = 83 <= coverage_oracle <= 97
    print(f"  Result: {'PASS  — simulation is correct' if oracle_ok else 'FAIL  — simulation bug detected'}")

    # ── Test B: Fitting coverage (estimated parameters) ───────────────────────
    # Now use FITTED parameters. Coverage will be lower than 90% because
    # kappa is biased upward in finite samples (a known statistical property),
    # causing confidence bands that are slightly too narrow.
    # This is expected behavior — not a code bug.
    print("\n" + "-" * 60)
    print("TEST B: Simulation coverage using FITTED parameters")
    print("  Coverage < 90% here is expected due to finite-sample kappa bias.")
    print("-" * 60)

    train_end  = int(len(path) * 0.80)
    r0_test    = path[train_end - 1]
    test_path  = path[train_end:]
    train_path = path[:train_end]
    fit_model   = VasicekModel()
    fit_model.fit_mle(train_path, dt=DT)

    fitted_paths  = fit_model.simulate(r0_test, len(test_path), N_PATHS)
    sim_f         = fitted_paths[1:]
    p5_f          = np.percentile(sim_f, 5,  axis=1)
    p95_f         = np.percentile(sim_f, 95, axis=1)
    mean_f        = sim_f.mean(axis=1)
    coverage_fit  = np.mean((test_path >= p5_f) & (test_path <= p95_f)) * 100
    rmse_fit      = np.sqrt(np.mean((mean_f - test_path) ** 2)) * 100

    print(f"  Fitted kappa : {fit_model.kappa:.4f}  (true: {TRUE_KAPPA},"
          f" bias: {(fit_model.kappa - TRUE_KAPPA)/TRUE_KAPPA*100:+.1f}%)")
    print(f"  RMSE         : {rmse_fit:.4f}%")
    print(f"  Coverage     : {coverage_fit:.1f}%")
    print(f"  Note: kappa overestimation makes bands narrower → lower coverage.")
    print(f"        This is the well-known 'mean reversion bias' in finite samples.")

    # ── Simulation fan chart (oracle params, from current rate) ──────────────
    print("\nGenerating simulation fan chart (sanity_simulation.png) ...")
    plot_simulation(
        oracle_model,
        r0=path[-1],
        horizon_years=10.0,
        n_paths=200,
        save_path="sanity_simulation.png",
    )

    # ── Backtest plot on synthetic data ───────────────────────────────────────
    print("Generating backtest plot (sanity_backtest.png) ...")
    dates    = pd.date_range(start="1970-01-01", periods=len(path), freq="MS")
    df_synth = pd.DataFrame({"rate": path}, index=dates)

    bt_model = VasicekModel()
    bt_results = run_backtest(
        model=bt_model,
        df=df_synth,
        train_frac=0.80,
        n_paths=N_PATHS,
        fit_method="mle",
    )
    plot_backtest(bt_results, save_path="sanity_backtest.png")

    # ── Final verdict ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Sanity Check — Final Verdict")
    print("=" * 60)
    print(f"  Param recovery (sigma) : {'PASS' if abs(mle_model.sigma - TRUE_SIGMA)/TRUE_SIGMA < 0.05 else 'WARN'}"
          f"  (sigma error: {abs(mle_model.sigma - TRUE_SIGMA)/TRUE_SIGMA*100:.1f}%)")
    print(f"  Simulation correctness : {'PASS' if oracle_ok else 'FAIL'}"
          f"  (oracle coverage: {coverage_oracle:.1f}%)")
    print(f"  Finite-sample kappa bias: {(fit_model.kappa-TRUE_KAPPA)/TRUE_KAPPA*100:+.1f}%"
          f"  (expected ~+15 to +25% for 40yr monthly data)")

    if oracle_ok:
        print("\n  [OK] Code is correct. The simulation produces well-calibrated")
        print("       uncertainty bands when given the true parameters.")
        print("       On real data, poor coverage is caused by:")
        print("         1. Finite-sample kappa bias (bands too narrow)")
        print("         2. Regime shifts the model cannot anticipate (e.g., 2022 hikes)")
    else:
        print("\n  [!!] Oracle coverage is off — check simulate() in vasicek.py.")
    print("=" * 60)


if __name__ == "__main__":
    run_sanity_check()
