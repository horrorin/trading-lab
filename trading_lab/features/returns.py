import numpy as np
import pandas as pd


def log_returns(prices: pd.Series) -> pd.Series:
    """
    Compute log returns from a price series.
    """
    returns = np.log(prices / prices.shift(1))
    returns.name = f"{prices.name}_log_returns"
    return returns.dropna()


def realized_volatility(returns: pd.Series, window: int = 1) -> pd.Series:
    """
    Realized volatility proxy based on absolute returns or rolling window.
    """
    if window == 1:
        vol = returns.abs()
    else:
        vol = returns.rolling(window).std()

    vol.name = f"{returns.name}_realized_vol"
    return vol.dropna()
