# vasicek-interest-rate

A Python implementation of the Vasicek (1977) interest rate model applied to the US Federal Funds Rate.

**Pipeline:** fetch data → fit model → simulate future paths → backtest

---

## What is the Vasicek Model?

The Vasicek model describes how interest rates evolve over time:

```
dr(t) = κ(θ - r(t))dt + σ dW(t)
```

| Parameter | Symbol | Meaning |
|-----------|--------|---------|
| Mean reversion speed | κ (kappa) | How fast rates snap back toward the long-run mean. High κ = fast reversion |
| Long-run mean | θ (theta) | The rate that the process gravitates toward over time |
| Volatility | σ (sigma) | The magnitude of random shocks |
| Half-life | ln(2)/κ | How long it takes for a deviation from θ to shrink by half |

**Intuition:** if rates spike far above θ, the κ(θ - r) term pulls them back down.
If rates fall far below θ, it pushes them back up. The σ dW term adds randomness at every step.

**Known limitation:** the Vasicek model allows negative interest rates (since the distribution
is normal). This was considered a flaw pre-2008 but became less controversial after several
central banks adopted negative rate policies.

---

## Project Structure

```
vasicek-interest-rate/
├── main.py                  ← run this
├── requirements.txt
├── data/
│   └── fetch.py             # pulls FEDFUNDS from FRED
├── model/
│   └── vasicek.py           # VasicekModel: fit_ols(), fit_mle(), simulate()
└── backtest/
    └── backtest.py          # run_backtest(), plot_backtest(), plot_simulation()
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Get a free FRED API key

- Register at https://fred.stlouisfed.org/docs/api/api_key.html
- Takes ~1 minute, key is emailed immediately

### 3. Set your API key and run

```bash
export FRED_API_KEY="your_key_here"   # Mac/Linux
setx FRED_API_KEY "your_key_here"     # Windows

python main.py
```

---

## Outputs

Running `main.py` prints fitted parameters and backtest metrics, and saves two plots:

| File | Description |
|------|-------------|
| `simulation.png` | Monte Carlo fan chart — future rate paths from today |
| `backtest.png` | Backtest fan chart — model forecast vs what actually happened |

---

## How to Read the Fitted Parameters

Example output you might see:

```
========================================
Vasicek Model — Fitted Parameters
========================================
  κ  (mean reversion speed) :  0.1823
  θ  (long-run mean)        :  3.8200%
  σ  (volatility)           :  0.0145
  Half-life                 :  3.2 years
  Long-run std              :  2.40%
========================================
```

**Is κ reasonable?**

| κ value | Half-life | Interpretation |
|---------|-----------|----------------|
| < 0.05  | > 10 yrs  | Very slow reversion — rates wander for years |
| 0.1–0.3 | 2–6 yrs   | Typical for monthly Fed Funds data ✓ |
| > 0.5   | < 1.5 yrs | Very fast reversion — unusually mean-reverting |

**Is θ reasonable?**
It should be in the ballpark of the historical average rate of your data window.
For US Fed Funds Rate since 1994, expect θ ≈ 2–4%.

**Is σ reasonable?**
Monthly volatility of 0.01–0.02 (1–2% annualized) is typical for the Fed Funds Rate.

---

## How to Read the Backtest Results

```
Backtest Metrics:
  RMSE (mean path vs actual) :  1.2340%
  90% CI coverage            :  82.3%  (target: 90%)
```

### RMSE — Root Mean Squared Error of the mean forecast path

The average distance between the model's predicted mean rate and the actual rate,
in percentage points.

| RMSE | Interpretation |
|------|---------------|
| < 0.5% | Excellent — very accurate mean forecast |
| 0.5–1.5% | Good — reasonable for interest rate forecasting |
| 1.5–3.0% | Fair — model captures trend but misses magnitude |
| > 3.0% | Poor — model is far off; consider a longer training window |

**Important context:** interest rate forecasting is genuinely hard.
Professional economists with full macro models typically achieve RMSE of 1–2%
over a 1–2 year horizon. A Vasicek RMSE under 1.5% is competitive.

### 90% CI Coverage — How well-calibrated is the uncertainty?

This measures what fraction of actual rates fell inside the model's 90% confidence band.
A perfectly calibrated model gives exactly 90%.

| Coverage | Interpretation |
|----------|---------------|
| 85–95%   | Well-calibrated — the confidence bands are trustworthy ✓ |
| > 95%    | Overconfident — bands are too wide (model overstates uncertainty) |
| < 80%    | Underconfident — bands are too narrow (model understates uncertainty) |

**Why coverage matters more than RMSE for simulation:**
Even if the mean path is off (high RMSE), a well-calibrated 90% band is still useful —
it tells you the plausible range of future rates. This is what you'd use for
scenario analysis or risk management.

### Common failure modes and what they mean

| Symptom | Likely cause |
|---------|-------------|
| RMSE is fine but coverage < 75% | σ is underestimated — try a longer training window or MLE fitting |
| Coverage is fine but RMSE > 3% | The Fed made a large unexpected policy shift during the test period (e.g., 2022 rate hikes) |
| θ is far from current rates | The model was trained on a different rate regime — consider trimming the training data to a more recent window |
| κ ≈ 0 | Mean reversion is not detectable in your data window — rates may be trending |

---

## Tuning the Model

In `main.py` you can adjust:

```python
DATA_START  = "1994-01-01"  # earlier start = more data but mixes rate regimes
TRAIN_FRAC  = 0.80          # increase for more training data
N_PATHS     = 5000          # more paths = smoother confidence bands (slower)
FIT_METHOD  = "mle"         # 'mle' is more accurate; 'ols' is faster
HORIZON_YRS = 5.0           # years to simulate forward
```

**Tip:** if backtesting over a period that includes a major regime shift (e.g., 2022 rate hike
cycle), expect poor RMSE. That is not a model bug — it reflects that the Vasicek model assumes
a stable long-run mean and cannot predict structural policy changes.

---

## References

- Vasicek, O. (1977). *An equilibrium characterization of the term structure.*
  Journal of Financial Economics, 5(2), 177–188.
- Federal Reserve Bank of St. Louis, Federal Funds Effective Rate [FEDFUNDS],
  retrieved from FRED: https://fred.stlouisfed.org/series/FEDFUNDS
