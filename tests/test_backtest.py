"""Offline check of the backtest logic on synthetic prices (no internet needed).

Run:  python test_backtest.py
"""
import numpy as np
import pandas as pd

import backtest1 as backtest
from indicators import add_indicators, pullback_reclaim_signal


def fake_prices(seed, n=1000):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2021-01-01", periods=n)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.015, n)))
    open_ = close * (1 + rng.normal(0, 0.004, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.006, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.006, n)))
    vol = rng.integers(1_000_000, 5_000_000, n)
    return pd.DataFrame({"open": open_, "high": high, "low": low,
                         "close": close, "volume": vol}, index=idx)


def test_stop_target_logic():
    """A hand-built path: entry, then a bar that hits the target exactly."""
    df = fake_prices(1, 120)
    df = backtest.prepare(df)
    df["signal"] = False
    k = 30
    df.iloc[k, df.columns.get_loc("signal")] = True
    entry = df["open"].iloc[k + 1]
    risk = backtest.ATR_STOP_MULTIPLIER * df["ATR_14"].iloc[k]
    target = entry + backtest.REWARD_RISK_RATIO * risk
    # force the day after entry to trade through the target without gapping
    df.iloc[k + 2, df.columns.get_loc("open")] = entry
    df.iloc[k + 2, df.columns.get_loc("high")] = target + 1
    df.iloc[k + 2, df.columns.get_loc("low")] = entry - 0.1 * risk
    trades, _ = backtest.simulate(df, "TEST")
    assert trades[0]["exit_reason"] == "Target", trades[0]
    assert abs(trades[0]["exit"] - target) < 1e-9
    print("stop/target logic OK")


def test_no_lookahead():
    """Signals on day t must not change when future rows are removed."""
    df = add_indicators(fake_prices(2, 400))
    full = pullback_reclaim_signal(df)
    cut = pullback_reclaim_signal(add_indicators(fake_prices(2, 400).iloc[:300]))
    assert (full.iloc[:300].fillna(False) == cut.fillna(False)).all()
    print("no-lookahead OK")


def test_full_run():
    data = {f"FAKE{i}.NS": fake_prices(i) for i in range(10, 20)}
    t = backtest.run(list(data), loader=lambda tk: data[tk])
    assert t is not None and len(t) > 0
    assert (t["entry_date"] > t["signal_date"]).all()
    assert (t["exit_date"] >= t["entry_date"]).all()
    assert (t["position"] <= backtest.MAX_POSITION + 1e-12).all()
    print("full run OK")


if __name__ == "__main__":
    test_stop_target_logic()
    test_no_lookahead()
    test_full_run()