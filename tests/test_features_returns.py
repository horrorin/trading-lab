import numpy as np
import pandas as pd

from trading_lab.features.returns import (
    log_returns,
    realized_volatility,
    annualize_volatility,
)


def test_log_returns_matches_definition():
    # Prices: 100 -> 110 -> 121 (constant +10% simple return)
    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    prices = pd.Series([100.0, 110.0, 121.0], index=idx, name="PX")

    r = log_returns(prices)

    expected = np.log(prices / prices.shift(1)).dropna()
    assert r.index.equals(expected.index)
    assert np.allclose(r.values, expected.values)
    assert r.name == "PX_log_returns"


def test_realized_volatility_window_1_is_abs_returns():
    idx = pd.date_range("2020-01-01", periods=4, freq="D")
    returns = pd.Series([0.01, -0.02, 0.00, 0.03], index=idx, name="R")

    vol = realized_volatility(returns, window=1)

    expected = returns.abs().dropna()
    assert np.allclose(vol.values, expected.values)
    assert vol.name == "R_realized_vol"


def test_realized_volatility_rolling_std():
    idx = pd.date_range("2020-01-01", periods=6, freq="D")
    returns = pd.Series([0.01, -0.02, 0.00, 0.03, -0.01, 0.02], index=idx, name="R")

    vol = realized_volatility(returns, window=3)

    expected = returns.rolling(3).std().dropna()
    assert vol.index.equals(expected.index)
    assert np.allclose(vol.values, expected.values, equal_nan=False)
    assert vol.name == "R_realized_vol"


def test_annualize_volatility_sqrt_time_scaling():
    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    vol = pd.Series([0.01, 0.02, 0.03], index=idx, name="VOL")

    ann = annualize_volatility(vol, periods_per_year=252)

    expected = vol * np.sqrt(252)
    assert np.allclose(ann.values, expected.values)
    assert ann.name == "VOL_ann"
