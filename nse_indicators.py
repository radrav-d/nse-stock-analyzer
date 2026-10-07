import numpy as np, pandas as pd
import yfinance as yf

TICKER = "SAGILITY.NS"
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

# 3. RSI   
delta = df["close"].diff()                       # day-over-day change
gain  = delta.clip(lower=0).rolling(14).mean()   # average of up-moves
loss  = (-delta.clip(upper=0)).rolling(14).mean()# average of down-moves
rs    = gain / loss
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

# 7. vol_avg 
df["vol_avg"]  = df["volume"].rolling(20).mean()
latest_volume = df["volume"].iloc[-1]
latest_vol_avg = df["vol_avg"].iloc[-1]

if latest_volume > latest_vol_avg:
    volume_status = "Above average"
else:
    volume_status = "Below average"

print(f"  Volume           : {volume_status}")
recent_support = df["low"].tail(20).min()
recent_resistance = df["high"].tail(20).max()

print(f"  Support          : Rs {recent_support:8.2f}")
print(f"  Resistance       : Rs {recent_resistance:8.2f}")
# 8. VWAP    
tp = (df["high"] + df["low"] + df["close"]) / 3
df["VWAP"] = (tp * df["volume"]).cumsum() / df["volume"].cumsum() # Note: cumulative VWAP over the full period (anchored VWAP); intraday VWAP would reset daily.

# the combined signal 
df["signal"] = (
    (df["close"] > df["SMA_20"]) &        # trend: uptrend
    (df["RSI"] > 30) & (df["RSI"] < 70) & # momentum: not extreme
    (df["volume"] > df["vol_avg"])        # volume: conviction
)
fired_signal = df["signal"].sum() 

# ---- Latest values ----
last_close  = df["close"].iloc[-1]
last_signal = df["signal"].iloc[-1]
last_rsi    = df["RSI"].iloc[-1]
last_vwap   = df["VWAP"].iloc[-1]
last_sma    = df["SMA_20"].iloc[-1]
last_date   = df.index[-1].date()

trend = "Bullish (above VWAP)" if last_close > last_vwap else "Bearish (below VWAP)"
signal_text = "YES — trend + momentum + volume aligned" if last_signal else "No"

# ---- Formatted report ----
print("=" * 45)
print(f"  {TICKER} — Technical Analysis")
print(f"  As of {last_date}")
print("=" * 45)
print(f"  Close price      : Rs {last_close:>8.2f}")
print(f"  20-day SMA       : Rs {last_sma:>8.2f}")
print(f"  VWAP             : Rs {last_vwap:>8.2f}")
print(f"  RSI (14)         : {last_rsi:>8.1f}")
print("-" * 45)
print(f"  VWAP trend       : {trend}")
print(f"  Buy signal today : {signal_text}")
print(f"  Signals (1 year) : {fired_signal} days")
print("=" * 45)
latest_close = df["close"].iloc[-1]
latest_sma = df["SMA_20"].iloc[-1]
latest_ema = df["EMA_20"].iloc[-1]
latest_rsi = df["RSI"].iloc[-1]

if latest_rsi > 70:
    momentum = "Overbought"
elif latest_rsi >= 50:
    momentum = "Positive"
elif latest_rsi >= 30:
    momentum = "Weak"
else:
    momentum = "Oversold"

print(f"  Momentum         : {momentum}")
latest_macd = df["MACD"].iloc[-1]
latest_macd_signal = df["Signal"].iloc[-1]

if latest_macd > latest_macd_signal:
    macd_status = "Bullish"
elif latest_macd < latest_macd_signal:
    macd_status = "Bearish"
else:
    macd_status = "Neutral"

print(f"  MACD             : {macd_status}")
if latest_close > latest_sma and latest_close > latest_ema:
    trend = "Bullish"
elif latest_close < latest_sma and latest_close < latest_ema:
    trend = "Bearish"
else:
    trend = "Mixed"
print(f"  Trend            : {trend}")

df.to_csv(f"{stock_name}_Indicators.csv")
print(f"Saved full indicator table to {stock_name}_Indicators.csv")
