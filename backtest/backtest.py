"""
Backtesting the Vasicek model.

Strategy:
    1. Split historical data into train (80%) and test (20%).
    2. Fit the model on the training set only.
    3. Simulate n_paths forward from the last training rate.
    4. Compare simulated distribution against actual test rates.

Metrics reported:
    - RMSE          : root mean squared error of the mean forecast path
    - Coverage 90%  : fraction of actual rates inside the 5th–95th percentile band
                      a well-calibrated model should be close to 90%
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def run_backtest(
    model,
    df: pd.DataFrame,
    train_frac: float = 0.80,
    n_paths: int = 2000,
    fit_method: str = "mle",
) -> dict:
    """
    Fit the model on the training window and simulate over the test window.

    Args:
        model      : VasicekModel instance (unfitted)
        df         : DataFrame with a 'rate' column and DatetimeIndex
        train_frac : fraction of data used for training
        n_paths    : Monte Carlo paths for simulation
        fit_method : 'mle' or 'ols'

    Returns:
        dict with results, metrics, and simulation paths
    """
    rates = df["rate"].values
    dates = df.index

    split      = int(len(rates) * train_frac)
    train_rates = rates[:split]
    test_rates  = rates[split:]
    train_dates = dates[:split]
    test_dates  = dates[split:]

    print(f"Train: {train_dates[0].date()} → {train_dates[-1].date()}  ({split} months)")
    print(f"Test : {test_dates[0].date()}  → {test_dates[-1].date()}   ({len(test_rates)} months)\n")

    # Fit on training data only
    if fit_method == "mle":
        model.fit_mle(train_rates)
    else:
        model.fit_ols(train_rates)
    model.summary()

    # Simulate forward from last training observation
    r0      = train_rates[-1]
    n_steps = len(test_rates)
    paths   = model.simulate(r0, n_steps, n_paths)  # (n_steps+1, n_paths)

    # Paths[0] = r0, paths[1:] = simulated test period
    sim        = paths[1:]                              # (n_steps, n_paths)
    mean_path  = sim.mean(axis=1)
    p5         = np.percentile(sim, 5,  axis=1)
    p25        = np.percentile(sim, 25, axis=1)
    p75        = np.percentile(sim, 75, axis=1)
    p95        = np.percentile(sim, 95, axis=1)

    # ── Metrics ──────────────────────────────────────────────────────────────
    rmse     = np.sqrt(np.mean((mean_path - test_rates) ** 2))
    coverage = np.mean((test_rates >= p5) & (test_rates <= p95))

    print(f"\nBacktest Metrics:")
    print(f"  RMSE (mean path vs actual) : {rmse * 100:.4f}%")
    print(f"  90% CI coverage            : {coverage * 100:.1f}%  (target: 90%)")

    return {
        "model"       : model,
        "train_rates" : train_rates,
        "train_dates" : train_dates,
        "test_rates"  : test_rates,
        "test_dates"  : test_dates,
        "paths"       : paths,
        "mean_path"   : mean_path,
        "p5"          : p5,
        "p25"         : p25,
        "p75"         : p75,
        "p95"         : p95,
        "rmse"        : rmse,
        "coverage_90" : coverage,
    }


def plot_backtest(results: dict, save_path: str = None) -> None:
    """Plot training history, simulated fan chart, and actual test path."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # ── Panel 1: Full history + fan chart ────────────────────────────────────
    ax = axes[0]

    train_dates = results["train_dates"]
    test_dates  = results["test_dates"]
    train_rates = results["train_rates"] * 100
    test_rates  = results["test_rates"]  * 100
    mean_path   = results["mean_path"]   * 100
    p5          = results["p5"]          * 100
    p25         = results["p25"]         * 100
    p75         = results["p75"]         * 100
    p95         = results["p95"]         * 100

    # Historical training data
    ax.plot(train_dates, train_rates, color="steelblue", lw=1.2, label="Historical (train)")

    # Simulation fan
    ax.fill_between(test_dates, p5,  p95, alpha=0.15, color="tomato", label="90% CI")
    ax.fill_between(test_dates, p25, p75, alpha=0.25, color="tomato", label="50% CI")
    ax.plot(test_dates, mean_path, color="tomato",   lw=1.5, ls="--", label="Mean forecast")
    ax.plot(test_dates, test_rates, color="black",   lw=1.5, label="Actual (test)")

    ax.axvline(train_dates[-1], color="gray", ls=":", lw=1.2, label="Train/test split")
    ax.set_title("Vasicek Model — Backtest Fan Chart", fontsize=13, fontweight="bold")
    ax.set_ylabel("Federal Funds Rate (%)")
    ax.legend(loc="upper left", fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(alpha=0.3)

    # ── Panel 2: Test period zoom ─────────────────────────────────────────────
    ax2 = axes[1]
    ax2.fill_between(test_dates, p5,  p95, alpha=0.15, color="tomato", label="90% CI")
    ax2.fill_between(test_dates, p25, p75, alpha=0.25, color="tomato", label="50% CI")
    ax2.plot(test_dates, mean_path,  color="tomato", lw=2,   ls="--", label="Mean forecast")
    ax2.plot(test_dates, test_rates, color="black",  lw=2,   label="Actual")

    rmse     = results["rmse"] * 100
    coverage = results["coverage_90"] * 100
    ax2.set_title(
        f"Test Period Zoom — RMSE: {rmse:.3f}%  |  90% CI Coverage: {coverage:.1f}%",
        fontsize=12,
    )
    ax2.set_ylabel("Federal Funds Rate (%)")
    ax2.set_xlabel("Date")
    ax2.legend(fontsize=9)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\nPlot saved to {save_path}")
    plt.show()


def plot_simulation(
    model,
    r0: float,
    horizon_years: float = 5.0,
    n_paths: int = 200,
    save_path: str = None,
) -> None:
    """
    Plot future simulation paths from a given starting rate.

    Args:
        model         : fitted VasicekModel
        r0            : starting rate (current rate)
        horizon_years : how many years to simulate forward
        n_paths       : number of paths to draw (keep small for visual clarity)
    """
    n_steps = int(horizon_years / model.dt)
    paths   = model.simulate(r0, n_steps, n_paths)

    t_axis  = np.arange(n_steps + 1) * model.dt  # in years

    fig, ax = plt.subplots(figsize=(12, 6))

    # Sample paths (semi-transparent)
    for i in range(min(n_paths, 100)):
        ax.plot(t_axis, paths[:, i] * 100, color="steelblue", alpha=0.08, lw=0.8)

    # Percentile bands
    p5  = np.percentile(paths, 5,  axis=1) * 100
    p95 = np.percentile(paths, 95, axis=1) * 100
    mean = paths.mean(axis=1) * 100

    ax.fill_between(t_axis, p5, p95, alpha=0.25, color="steelblue", label="90% CI")
    ax.plot(t_axis, mean,       color="navy",      lw=2, label="Mean path")
    ax.axhline(model.theta * 100, color="red", ls="--", lw=1.5,
               label=f"Long-run mean θ = {model.theta*100:.2f}%")

    ax.set_title(
        f"Vasicek Simulation — {horizon_years:.0f}-Year Forecast  "
        f"(r₀={r0*100:.2f}%, κ={model.kappa:.3f}, θ={model.theta*100:.2f}%, σ={model.sigma:.4f})",
        fontsize=12,
    )
    ax.set_xlabel("Years from now")
    ax.set_ylabel("Federal Funds Rate (%)")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    plt.show()
