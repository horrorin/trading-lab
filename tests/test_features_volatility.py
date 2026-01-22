import numpy as np
import pandas as pd

from trading_lab.features.volatility import (
    realized_volatility_std,
    annualize_volatility,
    volatility_target_weights,
)


def test_realized_volatility_std_rolling():
    idx = pd.date_range("2020-01-01", periods=10, freq="D")
    r = pd.Series(np.linspace(-0.05, 0.05, 10), index=idx, name="R")

    vol = realized_volatility_std(r, window=5)

    expected = r.rolling(5).std().dropna()
    assert vol.index.equals(expected.index)
    assert np.allclose(vol.values, expected.values)
    assert vol.name == "R_rv_std_5"


def test_annualize_volatility_sqrt_time_scaling():
    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    vol = pd.Series([0.01, 0.02, 0.03], index=idx, name="VOL")

    ann = annualize_volatility(vol, periods_per_year=252)

    expected = vol * np.sqrt(252)
    assert np.allclose(ann.values, expected.values)
    assert ann.name == "VOL_ann"


def test_volatility_target_weights_basic():
    idx = pd.date_range("2020-01-01", periods=4, freq="D")
    vol = pd.Series([0.01, 0.02, 0.04, 0.02], index=idx, name="VOL")
    target = 0.02

    w = volatility_target_weights(vol, target_vol=target)

    expected = target / vol
    assert np.allclose(w.values, expected.values)
    assert w.name == "VOL_vol_target_w"


def test_volatility_target_weights_handles_zero_with_clip():
    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    vol = pd.Series([0.01, 0.0, 0.02], index=idx, name="VOL")

    w = volatility_target_weights(vol, target_vol=0.02, eps=1e-6)

    # Middle value should be finite due to clipping
    assert np.isfinite(w.iloc[1])
