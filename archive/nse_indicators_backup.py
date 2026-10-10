import numpy as np, pandas as pd
import yfinance as yf

TICKER = "RELIANCE.NS"
stock_name = TICKER.replace(".NS", "").title()
# Download 1 year of daily NSE data and clean the columns
df = yf.download(TICKER, period="1y", progress=False)
df.columns = [col[0] for col in df.columns]   # flatten
df = df.rename(columns={"Open":"open", "High":"high", "Low":"low",
                        "Close":"close", "Volume":"volume"})

if df.empty:
    raise SystemExit("Download failed — check internet or ticker symbol.")

# 1. SMA_20 
df["SMA_20"] = df["close"].rolling(20).mean()

# 2. EMA_20 
df["EMA_20"] = df["close"].ewm(span=20, adjust=False).mean()

# RSI (14) using Wilder-style smoothing I(edited rsi)
delta = df["close"].diff()

gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

avg_gain = gain.ewm(
    alpha=1 / 14,
    adjust=False,
    min_periods=14
).mean()

avg_loss = loss.ewm(
    alpha=1 / 14,
    adjust=False,
    min_periods=14
).mean()

rs = avg_gain / avg_loss

df["RSI"] = 100 - (100 / (1 + rs))

# 4. MACD + Signal 
ema12 = df["close"].ewm(span=12, adjust=False).mean()
ema26 = df["close"].ewm(span=26, adjust=False).mean()
df["MACD"]      = ema12 - ema26
df["Signal"]    = df["MACD"].ewm(span=9, adjust=False).mean()
df["Histogram"] = df["MACD"] - df["Signal"]

# 5. Bollinger BB_up / BB_low  
mid = df["close"].rolling(20).mean()
sd  = df["close"].rolling(20).std()
df["BB_up"]  = mid + 2*sd
df["BB_low"] = mid - 2*sd

# 6. ATR      
pc = df["close"].shift(1)     # previous close
tr = pd.concat([df["high"]-df["low"],
                (df["high"]-pc).abs(),
                (df["low"]-pc).abs()], axis=1).max(axis=1)
df["ATR"] = tr.rolling(14).mean()
# 7. ADX (14) with Wilder-style smoothing

up_move = df["high"].diff()
down_move = -df["low"].diff()

plus_dm = pd.Series(
    np.where(
        (up_move > down_move) & (up_move > 0),
        up_move,
        0
    ),
    index=df.index
)

minus_dm = pd.Series(
    np.where(
        (down_move > up_move) & (down_move > 0),
        down_move,
        0
    ),
    index=df.index
)

# Wilder-style smoothed True Range
tr14 = tr.ewm(
    alpha=1/14,
    adjust=False,
    min_periods=14
).mean()

plus_dm14 = plus_dm.ewm(
    alpha=1/14,
    adjust=False,
    min_periods=14
).mean()

minus_dm14 = minus_dm.ewm(
    alpha=1/14,
    adjust=False,
    min_periods=14
).mean()

# Directional Indicators
df["Plus_DI"] = 100 * plus_dm14 / tr14
df["Minus_DI"] = 100 * minus_dm14 / tr14

# DX
dx = (
    100
    * (df["Plus_DI"] - df["Minus_DI"]).abs()
    / (df["Plus_DI"] + df["Minus_DI"])
)

# ADX
df["ADX"] = dx.ewm(
    alpha=1/14,
    adjust=False,
    min_periods=14
).mean()
# 7. vol_avg 
df["vol_avg"]  = df["volume"].rolling(20).mean()
latest_volume = df["volume"].iloc[-1]
latest_vol_avg = df["vol_avg"].iloc[-1]

if latest_volume > latest_vol_avg:
    volume_status = "Above average"
else:
    volume_status = "Below average"

recent_support = df["low"].tail(20).min()
recent_resistance = df["high"].tail(20).max()
# 8. 20-day rolling VWAP
tp = (df["high"] + df["low"] + df["close"]) / 3
df["VWAP"] = (
    (tp * df["volume"]).rolling(20).sum()
    / df["volume"].rolling(20).sum()
)

# the combined signal 
df["signal"] = (
    (df["close"] > df["SMA_20"]) &        # trend: uptrend
    (df["RSI"] > 30) & (df["RSI"] < 70) & # momentum: not extreme
    (df["volume"] > df["vol_avg"])        # volume: conviction
)
fired_signal = df["signal"].sum() 

# Latest values edited 
last_close = df["close"].iloc[-1]
last_signal = df["signal"].iloc[-1]
last_rsi = df["RSI"].iloc[-1]
last_vwap = df["VWAP"].iloc[-1]
last_sma = df["SMA_20"].iloc[-1]
last_ema = df["EMA_20"].iloc[-1]
last_date = df.index[-1].date()
latest_adx = df["ADX"].iloc[-1]
latest_plus_di = df["Plus_DI"].iloc[-1]
latest_minus_di = df["Minus_DI"].iloc[-1]

if latest_plus_di > latest_minus_di:
    adx_direction = "Bullish"
elif latest_minus_di > latest_plus_di:
    adx_direction = "Bearish"
else:
    adx_direction = "Neutral"
if latest_adx >= 25:
    adx_status = "Strong trend"
elif latest_adx >= 20:
    adx_status = "Developing trend"
else:
    adx_status = "Weak trend"
# VWAP trend
vwap_trend = "Bullish (above VWAP)" if last_close > last_vwap else "Bearish (below VWAP)"

# Momentum
if last_rsi > 70:
    momentum = "Overbought"
elif last_rsi >= 50:
    momentum = "Positive"
elif last_rsi >= 30:
    momentum = "Weak"
else:
    momentum = "Oversold"

# MACD
latest_macd = df["MACD"].iloc[-1]
latest_macd_signal = df["Signal"].iloc[-1]

if latest_macd > latest_macd_signal:
    macd_status = "Bullish"
elif latest_macd < latest_macd_signal:
    macd_status = "Bearish"
else:
    macd_status = "Neutral"

# Overall trend
if last_close > last_sma and last_close > last_ema:
    overall_trend = "Bullish"
elif last_close < last_sma and last_close < last_ema:
    overall_trend = "Bearish"
else:
    overall_trend = "Mixed"

# Buy signal text
signal_text = "YES — trend + momentum + volume aligned" if last_signal else "No"

# Support and resistance
recent_support = df["low"].tail(20).min()
recent_resistance = df["high"].tail(20).max()

# Volume
latest_volume = df["volume"].iloc[-1]
latest_vol_avg = df["vol_avg"].iloc[-1]

if latest_volume > latest_vol_avg:
    volume_status = "Above average"
else:
    volume_status = "Below average"

# ---- Technical score ----
price_above_sma = last_close > last_sma
price_above_ema = last_close > last_ema
price_above_vwap = last_close > last_vwap
rsi_positive = last_rsi >= 50
macd_bullish = latest_macd > latest_macd_signal
volume_above_average = latest_volume > latest_vol_avg

technical_score = sum([
    price_above_sma,
    price_above_ema,
    price_above_vwap,
    rsi_positive,
    macd_bullish,
    volume_above_average
])

# Overall bias
if technical_score >= 5:
    overall_bias = "Strong Bullish"
elif technical_score == 4:
    overall_bias = "Bullish"
elif technical_score == 3:
    overall_bias = "Neutral"
elif technical_score == 2:
    overall_bias = "Bearish"
else:
    overall_bias = "Strong Bearish"

# ---- Formatted report ----
print("=" * 45)
print(f"  {TICKER} — Technical Analysis")
print(f"  As of {last_date}")
print("=" * 45)

print(f"  Close price      : Rs {last_close:>8.2f}")
print(f"  20-day SMA       : Rs {last_sma:>8.2f}")
print(f"  20-day EMA       : Rs {last_ema:>8.2f}")
print(f"  20-day VWAP      : Rs {last_vwap:>8.2f}")
print(f"  RSI (14)         : {last_rsi:>8.1f}")

print("-" * 45)

print(f"  Trend            : {overall_trend}")
print(f"  Momentum         : {momentum}")
print(f"  MACD             : {macd_status}")
print(f"  Volume           : {volume_status}")
print(f"  VWAP trend       : {vwap_trend}")
print(f"  ADX (14)         : {latest_adx:>8.1f}")
print(f"  ADX status       : {adx_status}")
print(f"  +DI (14)         : {latest_plus_di:>8.1f}")
print(f"  -DI (14)         : {latest_minus_di:>8.1f}")
print(f"  ADX direction    : {adx_direction}")
print("-" * 45)

print(f"  Support          : Rs {recent_support:>8.2f}")
print(f"  Resistance       : Rs {recent_resistance:>8.2f}")

print("-" * 45)

print(f"  Buy signal today : {signal_text}")
print(f"  Signals (1 year) : {fired_signal} days")
print(f"  Technical Score  : {technical_score} / 6") # added now 
print(f"  Overall Bias     : {overall_bias}")
print("=" * 45)
print("  Score Breakdown")
print(f"  Price > SMA20    : {'Yes' if price_above_sma else 'No'}")
print(f"  Price > EMA20    : {'Yes' if price_above_ema else 'No'}")
print(f"  Price > VWAP     : {'Yes' if price_above_vwap else 'No'}")
print(f"  RSI >= 50        : {'Yes' if rsi_positive else 'No'}")
print(f"  MACD Bullish     : {'Yes' if macd_bullish else 'No'}")
print(f"  Volume > Average : {'Yes' if volume_above_average else 'No'}")
df.to_csv(f"{stock_name}_Indicators.csv")
print(f"Saved full indicator table to {stock_name}_Indicators.csv")