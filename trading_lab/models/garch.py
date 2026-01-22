import numpy as np
import pandas as pd
from arch import arch_model


def fit_garch(
    returns: pd.Series,
    p: int = 1,
    q: int = 1,
    dist: str = "t",
):
    """
    Fit a GARCH(p,q) model on return series (in percent).
    """
    r = returns * 100
    model = arch_model(r, mean="Zero", vol="GARCH", p=p, q=q, dist=dist)
    res = model.fit(disp="off")
    return res


def rolling_garch_forecast(
    returns: pd.Series,
    p: int,
    q: int,
    horizon: int = 1,
    test_size: int = 365 * 2,
):
    """
    Rolling one-step-ahead GARCH volatility forecast.
    """
    r = returns * 100
    forecasts = []
    index = r.index[-test_size:]

    for t in index:
        train = r.loc[:t].iloc[:-1]
        res = arch_model(train, mean="Zero", vol="GARCH", p=p, q=q, dist="t").fit(
            disp="off"
        )
        fc = res.forecast(horizon=horizon)
        sigma = np.sqrt(fc.variance.iloc[-1, horizon - 1])
        forecasts.append(sigma)

    return pd.Series(forecasts, index=index, name=f"GARCH({p},{q})_vol")
