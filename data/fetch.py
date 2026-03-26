"""
Fetch US interest rate data from FRED (Federal Reserve Economic Data).

Series used: FEDFUNDS — Federal Funds Effective Rate (monthly, %)
  - The overnight rate at which banks lend reserves to each other.
  - Primary policy rate controlled by the Fed.

Requires a free FRED API key:
    1. Register at https://fred.stlouisfed.org/docs/api/api_key.html
    2. Set it as an environment variable:
           Windows:  setx FRED_API_KEY "your_key_here"
           Mac/Linux: export FRED_API_KEY="your_key_here"
       Or pass it directly: fetch_fed_funds_rate(api_key="your_key_here")

Alternative series you can swap in:
    DGS3MO : 3-Month Treasury Constant Maturity Rate (daily)
    DGS10  : 10-Year Treasury Constant Maturity Rate (daily)
    TB3MS  : 3-Month Treasury Bill Secondary Market Rate (monthly)
"""

import os
import pandas as pd
from fredapi import Fred


def fetch_fed_funds_rate(
    start: str = "1994-01-01",
    end: str = None,
    api_key: str = None,
) -> pd.DataFrame:
    """
    Pull the Federal Funds Effective Rate from FRED.

    Returns a DataFrame with:
        index : DatetimeIndex (monthly)
        rate  : interest rate as a decimal (e.g. 0.05 = 5%)
    """
    key = api_key or os.environ.get("FRED_API_KEY")
    if not key:
        raise ValueError(
            "FRED API key not found.\n"
            "Get a free key at: https://fred.stlouisfed.org/docs/api/api_key.html\n"
            "Then set it: export FRED_API_KEY='your_key_here'\n"
            "Or pass it directly: fetch_fed_funds_rate(api_key='your_key_here')"
        )

    fred = Fred(api_key=key)
    print(f"Fetching FEDFUNDS from FRED ({start} → {end or 'today'}) ...")

    series = fred.get_series("FEDFUNDS", observation_start=start, observation_end=end)
    df = pd.DataFrame({"rate": series / 100.0})   # % → decimal
    df.index.name = "date"
    df = df.dropna()

    print(f"  {len(df)} monthly observations loaded.")
    return df
