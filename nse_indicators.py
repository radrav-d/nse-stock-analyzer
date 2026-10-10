"""Technical analysis report for one NSE stock.

Run:  python nse_indicators.py            (analyses RELIANCE.NS)
      python nse_indicators.py TCS.NS     (analyses any other ticker)

All the indicator maths (SMA, EMA, RSI, MACD, VWAP, ATR, ADX) lives in
indicators.py, the same file the backtester uses. That means this report and
the backtest can never disagree about what an indicator says.
"""
import sys
from datetime import date, timedelta

import pandas as pd

from indicators import add_indicators, load_prices, technical_score

DEFAULT_TICKER = "RELIANCE.NS"
HISTORY_DAYS = 730   # download about 2 years so every indicator has time to warm up
REPORT_DAYS = 365    # the "last year" window used to count buy signals

# Every column the report needs. Rows missing any of these are dropped.
REQUIRED = ["SMA_20", "EMA_20", "RSI", "MACD", "Signal", "VWAP",
            "vol_avg", "ADX", "Plus_DI", "Minus_DI", "ATR_14"]


def get_data(ticker):
    """Download prices and add all indicators."""
    end = date.today() + timedelta(days=1)       # Yahoo's end date is exclusive
    start = end - timedelta(days=HISTORY_DAYS)
    df = load_prices(ticker, start.isoformat(), end.isoformat())
    if df is None or df.empty:
        raise SystemExit("Download failed - check internet or ticker symbol.")

    df = add_indicators(df).dropna(subset=REQUIRED)
    if len(df) < 30:
        raise SystemExit("Not enough price history to calculate the indicators.")
    return df


def momentum_label(rsi):
    if rsi > 70:
        return "Overbought"
    if rsi >= 50:
        return "Positive"
    if rsi >= 30:
        return "Weak"
    return "Oversold"


def adx_labels(adx, plus_di, minus_di):
    if plus_di > minus_di:
        direction = "Bullish"
    elif minus_di > plus_di:
        direction = "Bearish"
    else:
        direction = "Neutral"

    if adx >= 25:
        strength = "Strong trend"
    elif adx >= 20:
        strength = "Developing trend"
    else:
        strength = "Weak trend"
    return direction, strength


def bias_label(score):
    if score >= 5:
        return "Strong Bullish"
    if score == 4:
        return "Bullish"
    if score == 3:
        return "Neutral"
    if score == 2:
        return "Bearish"
    return "Strong Bearish"


def analyze(df):
    """Turn the indicator table into the values and labels shown in the report."""
    last = df.iloc[-1]
    close = last["close"]

    # The buy signal: uptrend + momentum not extreme + volume above average.
    signal = ((df["close"] > df["SMA_20"])
              & (df["RSI"] > 30) & (df["RSI"] < 70)
              & (df["volume"] > df["vol_avg"]))
    last_year = signal[signal.index > df.index[-1] - pd.Timedelta(days=REPORT_DAYS)]

    # The six yes/no checks behind the technical score.
    checks = {
        "Price > SMA20": bool(close > last["SMA_20"]),
        "Price > EMA20": bool(close > last["EMA_20"]),
        "Price > VWAP": bool(close > last["VWAP"]),
        "RSI >= 50": bool(last["RSI"] >= 50),
        "MACD Bullish": bool(last["MACD"] > last["Signal"]),
        "Volume > Average": bool(last["volume"] > last["vol_avg"]),
    }

    # The score itself comes from indicators.py (the same one the backtester
    # uses). This line double-checks it agrees with the six checks above.
    score = int(technical_score(df).iloc[-1])
    assert score == sum(checks.values()), "Score does not match the six checks"

    if close > last["SMA_20"] and close > last["EMA_20"]:
        trend = "Bullish"
    elif close < last["SMA_20"] and close < last["EMA_20"]:
        trend = "Bearish"
    else:
        trend = "Mixed"

    if last["MACD"] > last["Signal"]:
        macd = "Bullish"
    elif last["MACD"] < last["Signal"]:
        macd = "Bearish"
    else:
        macd = "Neutral"

    adx_direction, adx_status = adx_labels(last["ADX"], last["Plus_DI"], last["Minus_DI"])

    return {
        "date": df.index[-1].date(),
        "close": close,
        "sma": last["SMA_20"],
        "ema": last["EMA_20"],
        "vwap": last["VWAP"],
        "rsi": last["RSI"],
        "atr": last["ATR_14"],
        "adx": last["ADX"],
        "plus_di": last["Plus_DI"],
        "minus_di": last["Minus_DI"],
        "trend": trend,
        "momentum": momentum_label(last["RSI"]),
        "macd": macd,
        "volume": "Above average" if checks["Volume > Average"] else "Below average",
        "vwap_trend": "Bullish (above VWAP)" if checks["Price > VWAP"] else "Bearish (below VWAP)",
        "adx_direction": adx_direction,
        "adx_status": adx_status,
        "support": df["low"].tail(20).min(),
        "resistance": df["high"].tail(20).max(),
        "signal_today": bool(signal.iloc[-1]),
        "signals_year": int(last_year.sum()),
        "score": score,
        "bias": bias_label(score),
        "checks": checks,
    }


def print_report(ticker, r):
    line = "=" * 45
    dash = "-" * 45
    print(line)
    print(f"  {ticker} — Technical Analysis")
    print(f"  As of {r['date']}")
    print(line)

    print(f"  Close price      : Rs {r['close']:>8.2f}")
    print(f"  20-day SMA       : Rs {r['sma']:>8.2f}")
    print(f"  20-day EMA       : Rs {r['ema']:>8.2f}")
    print(f"  20-day VWAP      : Rs {r['vwap']:>8.2f}")
    print(f"  RSI (14)         : {r['rsi']:>8.1f}")
    print(f"  ATR (14)         : Rs {r['atr']:>8.2f}")
    print(dash)

    print(f"  Trend            : {r['trend']}")
    print(f"  Momentum         : {r['momentum']}")
    print(f"  MACD             : {r['macd']}")
    print(f"  Volume           : {r['volume']}")
    print(f"  VWAP trend       : {r['vwap_trend']}")
    print(f"  ADX (14)         : {r['adx']:>8.1f}")
    print(f"  ADX status       : {r['adx_status']}")
    print(f"  +DI (14)         : {r['plus_di']:>8.1f}")
    print(f"  -DI (14)         : {r['minus_di']:>8.1f}")
    print(f"  ADX direction    : {r['adx_direction']}")
    print(dash)

    print(f"  Support          : Rs {r['support']:>8.2f}")
    print(f"  Resistance       : Rs {r['resistance']:>8.2f}")
    print(dash)

    signal_text = "YES — trend + momentum + volume aligned" if r["signal_today"] else "No"
    print(f"  Buy signal today : {signal_text}")
    print(f"  Signals (1 year) : {r['signals_year']} days")
    print(f"  Technical Score  : {r['score']} / 6")
    print(f"  Overall Bias     : {r['bias']}")
    print(line)
    print("  Score Breakdown")
    for name, passed in r["checks"].items():
        print(f"  {name:<17}: {'Yes' if passed else 'No'}")


def main():
    ticker = sys.argv[1].upper() if len(sys.argv) > 1 else DEFAULT_TICKER
    df = get_data(ticker)
    print_report(ticker, analyze(df))

    name = ticker.replace(".NS", "").title()
    df.to_csv(f"{name}_Indicators.csv")
    print(f"Saved full indicator table to {name}_Indicators.csv")


if __name__ == "__main__":
    main()