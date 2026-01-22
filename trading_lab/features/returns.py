import numpy as np
import pandas as pd


def log_returns(prices: pd.Series) -> pd.Series:
    """
    Compute log returns from a price series.

    Log returns are preferred over simple returns in quantitative finance because:
    - They are time-additive
    - They behave better under compounding assumptions
    - They are more symmetric for statistical modeling

    Parameters
    ----------
    prices : pd.Series
        Price time series indexed by datetime.

    Returns
    -------
    pd.Series
        Log returns series, aligned and NaN-dropped.
    """
    # Compute log(P_t / P_{t-1})
    returns = np.log(prices / prices.shift(1))

    # Assign an explicit semantic name
    returns.name = f"{prices.name}_log_returns"

    # Drop first NaN introduced by shifting
    return returns.dropna()


def realized_volatility(returns: pd.Series, window: int = 1) -> pd.Series:
    """
    Compute realized volatility proxy from returns.

    Supports:
    - Absolute returns (window=1)
    - Rolling standard deviation (window > 1)

    This function provides a non-parametric proxy for volatility, commonly used
    to compare with model-based forecasts (e.g., GARCH).

    Parameters
    ----------
    returns : pd.Series
        Returns time series.
    window : int, optional
        Rolling window size. If 1, uses |returns| as an instantaneous proxy.

    Returns
    -------
    pd.Series
        Realized volatility series.
    """
    # Instantaneous realized volatility proxy
    if window == 1:
        vol = returns.abs()
    # Rolling volatility estimate
    else:
        vol = returns.rolling(window).std()

    vol.name = f"{returns.name}_realized_vol"
    return vol.dropna()


def annualize_volatility(vol: pd.Series, periods_per_year: int = 252) -> pd.Series:
    """
    Annualize a volatility series.

    Converts volatility measured at a given sampling frequency (e.g., daily)
    into an annualized metric using sqrt-time scaling:

        sigma_annual = sigma_period * sqrt(periods_per_year)

    Parameters
    ----------
    vol : pd.Series
        Volatility series at base frequency.
    periods_per_year : int, optional
        Number of periods per year (252 for daily trading days).

    Returns
    -------
    pd.Series
        Annualized volatility series.
    """
    out = vol * np.sqrt(periods_per_year)
    out.name = f"{vol.name}_ann"
    return out
