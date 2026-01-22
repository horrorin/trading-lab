import numpy as np
import pandas as pd

from trading_lab.features.normalization import zscore, clip_series


def test_zscore_rolling_properties():
    idx = pd.date_range("2020-01-01", periods=20, freq="D")
    s = pd.Series(np.arange(20, dtype=float), index=idx, name="X")

    z = zscore(s, window=10)

    # Same index as rolling window dropna
    assert z.index.min() == idx[9]
    assert z.name == "X_z_10"

    # Basic sanity: finite values
    assert np.isfinite(z.values).all()


def test_clip_series_preserves_name():
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    s = pd.Series([-2, -1, 0, 1, 2], index=idx, name="X").astype(float)

    out = clip_series(s, lower=-1.0, upper=1.0)
    assert out.name == "X"
    assert out.min() >= -1.0
    assert out.max() <= 1.0
