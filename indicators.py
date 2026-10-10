"""Shared indicator and data-loading code for the analyzer and the backtester.

Every indicator here is calculated only from data up to and including the
current row, so there is no lookahead. ATR, RSI and ADX all use Wilder
smoothing, so the analyzer and backtester now agree with each other.
"""
import numpy as np
import pandas as pd

WILDER = dict(alpha=1 / 14, adjust=False, min_periods=14)


def load_prices(ticker, start, end):
    """Download daily adjusted OHLCV from Yahoo Finance with clean columns."""
    import yfinance as yf

    df = yf.download(ticker, start=start, end=end, progress=False,
                     auto_adjust=True, actions=True)
    if df.empty:
        return None

    # Flatten only if the columns are multi-level; find the level that holds OHLCV.
    if isinstance(df.columns, pd.MultiIndex):
        needed = {"Open", "High", "Low", "Close", "Volume"}
        for level in range(df.columns.nlevels):
            labels = df.columns.get_level_values(level)
            if needed.issubset(set(labels)):
                df.columns = labels
                break

    df = df.drop(columns=[c for c in ("Dividends", "Stock Splits") if c in df.columns])
    df = df.rename(columns={"Open": "open", "High": "high", "Low": "low",
                            "Close": "close", "Volume": "volume"})
    df = df[["open", "high", "low", "close", "volume"]]
    df = df[df["volume"] > 0]  # drop non-trading rows
    return df


def add_indicators(df):
    """Return a copy of df with all indicators added."""
    df = df.copy()
    close = df["close"]

    df["SMA_20"] = close.rolling(20).mean()
    df["EMA_20"] = close.ewm(span=20, adjust=False).mean()

    # RSI (Wilder)
    delta = close.diff()
    avg_gain = delta.clip(lower=0).ewm(**WILDER).mean()
    avg_loss = (-delta.clip(upper=0)).ewm(**WILDER).mean()
    df["RSI"] = 100 - 100 / (1 + avg_gain / avg_loss)

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["Histogram"] = df["MACD"] - df["Signal"]

    # Bollinger bands
    mid = close.rolling(20).mean()
    sd = close.rolling(20).std()
    df["BB_up"] = mid + 2 * sd
    df["BB_low"] = mid - 2 * sd

    # Volume and VWAP (20-day rolling)
    df["vol_avg"] = df["volume"].rolling(20).mean()
    tp = (df["high"] + df["low"] + close) / 3
    df["VWAP"] = (tp * df["volume"]).rolling(20).sum() / df["volume"].rolling(20).sum()

    # ATR and ADX (Wilder)
    prev_close = close.shift(1)
    tr = pd.concat([df["high"] - df["low"],
                    (df["high"] - prev_close).abs(),
                    (df["low"] - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.ewm(**WILDER).mean()
    df["ATR_14"] = atr

    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    df["Plus_DI"] = 100 * plus_dm.ewm(**WILDER).mean() / atr
    df["Minus_DI"] = 100 * minus_dm.ewm(**WILDER).mean() / atr
    dx = 100 * (df["Plus_DI"] - df["Minus_DI"]).abs() / (df["Plus_DI"] + df["Minus_DI"])
    df["ADX"] = dx.ewm(**WILDER).mean()

    return df


def technical_score(df):
    """Six-point score, one value per row."""
    return ((df["close"] > df["SMA_20"]).astype(int)
            + (df["close"] > df["EMA_20"]).astype(int)
            + (df["close"] > df["VWAP"]).astype(int)
            + (df["RSI"] >= 50).astype(int)
            + (df["MACD"] > df["Signal"]).astype(int)
            + (df["volume"] > df["vol_avg"]).astype(int))


def pullback_reclaim_signal(df):
    """EMA pullback/reclaim: yesterday closed at/below EMA, today closes back above."""
    return ((df["EMA_20"] > df["SMA_20"])
            & (df["close"].shift(1) <= df["EMA_20"].shift(1))
            & (df["close"] > df["EMA_20"])
            & (df["RSI"] > 50)
            & (df["volume"] > df["vol_avg"]))