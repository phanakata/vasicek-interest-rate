"""
Vasicek Interest Rate Model (1977)

The model describes the short rate r(t) as an Ornstein-Uhlenbeck process:

    dr(t) = κ(θ - r(t))dt + σ dW(t)

Parameters:
    κ (kappa) : mean reversion speed   — how fast rates snap back to θ
    θ (theta) : long-run mean           — the rate the process reverts to
    σ (sigma) : volatility              — magnitude of random shocks

Key properties:
    - Rates are normally distributed (can go negative — a known limitation)
    - Half-life of deviations: ln(2) / κ
    - Long-run variance: σ² / (2κ)

Two fitting methods are provided:
    1. OLS  : fast, closed-form, slightly biased for short samples
    2. MLE  : uses the exact conditional distribution, preferred in practice
"""

import numpy as np
from scipy.optimize import minimize


class VasicekModel:

    def __init__(self):
        self.kappa = None   # mean reversion speed
        self.theta = None   # long-run mean
        self.sigma = None   # volatility
        self.dt    = None   # time step (years); 1/12 for monthly data

    # ── Fitting ───────────────────────────────────────────────────────────────

    def fit_ols(self, rates: np.ndarray, dt: float = 1 / 12) -> "VasicekModel":
        """
        Fit parameters via OLS on the Euler-Maruyama discretization.

        Discretized SDE:
            Δr = κθ·dt − κ·r(t)·dt + σ√dt·ε
            Δr = A + B·r(t) + ε          ← simple linear regression

        Recovering parameters:
            κ = −B / dt
            θ = −A / B
            σ = std(residuals) / √dt
        """
        self.dt = dt
        r  = rates[:-1]        # r(t)
        dr = np.diff(rates)    # Δr = r(t+dt) − r(t)

        # OLS: [A, B] = argmin ||dr − A − B·r||²
        X      = np.column_stack([np.ones(len(r)), r])
        coeffs = np.linalg.lstsq(X, dr, rcond=None)[0]
        A, B   = coeffs

        self.kappa = -B / dt
        self.theta = -A / B
        self.sigma = np.std(dr - (A + B * r)) / np.sqrt(dt)

        return self

    def fit_mle(self, rates: np.ndarray, dt: float = 1 / 12) -> "VasicekModel":
        """
        Fit parameters via Maximum Likelihood Estimation.

        The Vasicek model has an exact solution. Given r(t), r(t+dt) is
        normally distributed with:

            mean = r(t)·e^(−κ·dt) + θ·(1 − e^(−κ·dt))
            var  = σ²·(1 − e^(−2κ·dt)) / (2κ)

        MLE maximizes the log-likelihood of all observed transitions.
        OLS estimates are used as the starting point for the optimizer.
        """
        self.dt = dt
        r_t  = rates[:-1]
        r_t1 = rates[1:]

        def neg_log_likelihood(params):
            kappa, theta, sigma = params
            if kappa <= 0 or sigma <= 0:
                return np.inf

            e_kdt = np.exp(-kappa * dt)
            mu    = r_t * e_kdt + theta * (1 - e_kdt)
            var   = sigma ** 2 * (1 - e_kdt ** 2) / (2 * kappa)

            return 0.5 * np.sum(np.log(2 * np.pi * var) + (r_t1 - mu) ** 2 / var)

        # Warm-start from OLS
        self.fit_ols(rates, dt)
        x0 = [self.kappa, self.theta, self.sigma]

        result = minimize(
            neg_log_likelihood, x0,
            method="Nelder-Mead",
            options={"maxiter": 50_000, "xatol": 1e-10, "fatol": 1e-10},
        )

        self.kappa, self.theta, self.sigma = np.abs(result.x)
        return self

    # ── Simulation ────────────────────────────────────────────────────────────

    def simulate(self, r0: float, n_steps: int, n_paths: int = 1000) -> np.ndarray:
        """
        Simulate n_paths forward using the exact conditional distribution.

        At each step:
            r(t+dt) | r(t) ~ N(μ, var)
            μ   = r(t)·e^(−κ·dt) + θ·(1 − e^(−κ·dt))
            var = σ²·(1 − e^(−2κ·dt)) / (2κ)

        Args:
            r0      : starting rate (last observed rate)
            n_steps : number of time steps to simulate
            n_paths : number of Monte Carlo paths

        Returns:
            paths : np.ndarray of shape (n_steps+1, n_paths)
                    paths[0] = r0 for all paths
        """
        dt    = self.dt
        e_kdt = np.exp(-self.kappa * dt)
        std   = np.sqrt(self.sigma ** 2 * (1 - e_kdt ** 2) / (2 * self.kappa))

        # Pre-generate all random shocks at once (vectorized, no Python loop)
        shocks = std * np.random.randn(n_steps, n_paths)
        drift  = self.theta * (1 - e_kdt)

        paths    = np.empty((n_steps + 1, n_paths))
        paths[0] = r0

        for t in range(n_steps):
            paths[t+1] = paths[t] * e_kdt + drift + shocks[t]

        return paths

    # ── Summary ───────────────────────────────────────────────────────────────

    def summary(self) -> None:
        print("=" * 40)
        print("Vasicek Model -- Fitted Parameters")
        print("=" * 40)
        print(f"  kappa (mean reversion speed) : {self.kappa:.4f}")
        print(f"  theta (long-run mean)         : {self.theta * 100:.4f}%")
        print(f"  sigma (volatility)            : {self.sigma:.4f}")
        print(f"  Half-life                     : {np.log(2) / self.kappa:.1f} years")
        print(f"  Long-run std                  : {(self.sigma / np.sqrt(2 * self.kappa)) * 100:.4f}%")
        print("=" * 40)
