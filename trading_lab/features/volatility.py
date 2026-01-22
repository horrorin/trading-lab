import numpy as np
import pandas as pd


def realized_volatility_std(returns: pd.Series, window: int = 20) -> pd.Series:
    """
    Rolling realized volatility (standard deviation of returns).

    Parameters
    ----------
    returns : pd.Series
        Return series.
    window : int
        Rolling window size.

    Returns
    -------
    pd.Series
        Rolling volatility series (same units as returns).
    """
    vol = returns.rolling(window).std()
    vol.name = f"{returns.name}_rv_std_{window}"
    return vol.dropna()


def annualize_volatility(vol: pd.Series, periods_per_year: int = 252) -> pd.Series:
    """
    Annualize volatility via sqrt-time scaling.
    """
    out = vol * np.sqrt(periods_per_year)
    out.name = f"{vol.name}_ann"
    return out


def volatility_target_weights(
    vol: pd.Series, target_vol: float, eps: float = 1e-12
) -> pd.Series:
    """
    Compute volatility targeting weights: w_t = target_vol / vol_t.

    Commonly used as a risk overlay (position sizing) for a strategy's returns.

    Parameters
    ----------
    vol : pd.Series
        Volatility estimate series.
    target_vol : float
        Target volatility in the same units as `vol` (e.g., daily vol).
    eps : float
        Small constant to avoid division by zero.

    Returns
    -------
    pd.Series
        Weight series.
    """
    w = target_vol / (vol.clip(lower=eps))
    w.name = f"{vol.name}_vol_target_w"
    return w.dropna()
