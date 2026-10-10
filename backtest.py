import pandas as pd
import yfinance as yf

TICKER = "SUZLON.NS"
# Research assumption: 0.10% total round-trip trading cost.
ROUND_TRIP_COST = 0.001

# ---------------------------------------------------------
# Download historical data
# ---------------------------------------------------------
# Download historical data
df = yf.download(
    TICKER,
    period="3y",
    progress=False,
    auto_adjust=True,
    actions=True,
    repair=True
)

if df.empty:
    raise SystemExit(
        "Download failed — check ticker or internet connection."
    )

# Flatten columns only if they are actually multi-level.
# Find the level containing Open, High, Low, Close and Volume.
if isinstance(df.columns, pd.MultiIndex):
    required_columns = {"Open", "High", "Low", "Close", "Volume"}

    for level in range(df.columns.nlevels):
        labels = df.columns.get_level_values(level)

        if required_columns.issubset(set(labels)):
            df.columns = labels
            break

# Inspect corporate actions after normalizing column names
action_cols = [
    col for col in ["Dividends", "Stock Splits"]
    if col in df.columns
]

print("\nCorporate actions returned by Yahoo Finance:")

if action_cols:
    action_mask = df[action_cols].fillna(0).ne(0).any(axis=1)
    actions = df.loc[action_mask, action_cols]

    print(actions if not actions.empty else "No actions reported")
    df = df.drop(columns=action_cols)
else:
    print("No corporate-action columns returned")

# Standardize names for the rest of the program
df = df.rename(columns={
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Volume": "volume"
})
# Check adjusted prices around corporate actions
print("\nCorporate-action price continuity check:")

for event_date in ["2024-04-04", "2026-03-09"]:
    start = pd.Timestamp(event_date) - pd.Timedelta(days=4)
    end = pd.Timestamp(event_date) + pd.Timedelta(days=4)

    window = df.loc[start:end, ["close"]]

    print(f"\nAround {event_date}:")
    print(window.to_string())
# Verify the columns before calculating indicators
required = {"open", "high", "low", "close", "volume"}
missing = required - set(df.columns)

if missing:
    raise SystemExit(
        f"Missing required price columns: {sorted(missing)}. "
        f"Available columns: {list(df.columns)}"
    )

if df.empty:
    raise SystemExit(
        "Download failed — check ticker or internet connection."
    )

# ---------------------------------------------------------
# Calculate indicators
# ---------------------------------------------------------

# 1. Trend indicators
df["SMA_20"] = df["close"].rolling(20).mean()
df["EMA_20"] = df["close"].ewm(span=20, adjust=False).mean()

# RSI (14) using Wilder-style smoothing
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

# 3. MACD
ema12 = df["close"].ewm(span=12, adjust=False).mean()
ema26 = df["close"].ewm(span=26, adjust=False).mean()

df["MACD"] = ema12 - ema26
df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

# 4. Volume average
df["vol_avg"] = df["volume"].rolling(20).mean()

# 5. 20-day rolling VWAP
tp = (df["high"] + df["low"] + df["close"]) / 3

df["VWAP"] = (
    (tp * df["volume"]).rolling(20).sum()
    / df["volume"].rolling(20).sum()
)
# 6. ATR + ADX (14) using Wilder-style smoothing

previous_close = df["close"].shift(1)

tr = pd.concat([
    df["high"] - df["low"],
    (df["high"] - previous_close).abs(),
    (df["low"] - previous_close).abs()
], axis=1).max(axis=1)

atr14 = tr.ewm(
    alpha=1 / 14,
    adjust=False,
    min_periods=14
).mean()
df["ATR_14"] = atr14
up_move = df["high"].diff()
down_move = -df["low"].diff()

plus_dm = pd.Series(
    0.0,
    index=df.index
)

minus_dm = pd.Series(
    0.0,
    index=df.index
)

plus_dm[
    (up_move > down_move) & (up_move > 0)
] = up_move[
    (up_move > down_move) & (up_move > 0)
]

minus_dm[
    (down_move > up_move) & (down_move > 0)
] = down_move[
    (down_move > up_move) & (down_move > 0)
]

plus_dm14 = plus_dm.ewm(
    alpha=1 / 14,
    adjust=False,
    min_periods=14
).mean()

minus_dm14 = minus_dm.ewm(
    alpha=1 / 14,
    adjust=False,
    min_periods=14
).mean()

df["Plus_DI"] = 100 * plus_dm14 / atr14
df["Minus_DI"] = 100 * minus_dm14 / atr14

dx = (
    100
    * (df["Plus_DI"] - df["Minus_DI"]).abs()
    / (df["Plus_DI"] + df["Minus_DI"])
)

df["ADX"] = dx.ewm(
    alpha=1 / 14,
    adjust=False,
    min_periods=14
).mean()
# ---------------------------------------------------------
# Remove rows where indicators are not ready
# ---------------------------------------------------------

df = df.dropna()
# ADX trend classification

df["adx_direction"] = "Neutral"

df.loc[
    df["Plus_DI"] > df["Minus_DI"],
    "adx_direction"
] = "Bullish"

df.loc[
    df["Minus_DI"] > df["Plus_DI"],
    "adx_direction"
] = "Bearish"

df["adx_strength"] = "Weak"

df.loc[
    df["ADX"] >= 20,
    "adx_strength"
] = "Developing"

df.loc[
    df["ADX"] >= 25,
    "adx_strength"
] = "Strong"


# Six-point technical score
# ---------------------------------------------------------

df["technical_score"] = (
    (df["close"] > df["SMA_20"]).astype(int)
    + (df["close"] > df["EMA_20"]).astype(int)
    + (df["close"] > df["VWAP"]).astype(int)
    + (df["RSI"] >= 50).astype(int)
    + (df["MACD"] > df["Signal"]).astype(int)
    + (df["volume"] > df["vol_avg"]).astype(int)
)


# ---------------------------------------------------------
# FORWARD-RETURN ANALYSIS AND STRUCTURED REPORTING
# ---------------------------------------------------------

# Entry: next trading day's open.
# Exit: close five trading rows after the signal date.
df["entry_price"] = df["open"].shift(-1)
df["exit_price"] = df["close"].shift(-5)

df["gross_return"] = df["exit_price"] / df["entry_price"] - 1
df["net_return"] = df["gross_return"] - ROUND_TRIP_COST

# Only observations with a complete forward return are evaluated.
test = df.loc[df["net_return"].notna()].copy()

def summarize(data):
    """Summarize completed forward-return observations."""
    returns = data["net_return"].dropna()

    if returns.empty:
        return {
            "Observations": 0,
            "Win rate %": float("nan"),
            "Average %": float("nan"),
            "Median %": float("nan"),
            "Best %": float("nan"),
            "Worst %": float("nan"),
        }

    return {
        "Observations": len(returns),
        "Win rate %": (returns > 0).mean() * 100,
        "Average %": returns.mean() * 100,
        "Median %": returns.median() * 100,
        "Best %": returns.max() * 100,
        "Worst %": returns.min() * 100,
    }


def print_table(title, rows):
    """Print a consistently formatted report table."""
    print(f"\n{'=' * 76}")
    print(title)
    print("=" * 76)

    if not rows:
        print("No qualifying observations.")
        return

    table = pd.DataFrame(rows)

    if "Observations" in table.columns:
        table["Observations"] = table["Observations"].astype(int)

    print(table.to_string(
        index=False,
        formatters={
            col: (lambda x: f"{x:.2f}")
            for col in [
                "Win rate %",
                "Average %",
                "Median %",
                "Best %",
                "Worst %",
            ]
            if col in table.columns
        }
    ))


print("\n" + "=" * 76)
print(f"{TICKER} | HISTORICAL RESEARCH REPORT")
print("=" * 76)
print(f"Dataset: {df.index[0].date()} to {df.index[-1].date()}")
print(f"Indicator-ready observations: {len(df)}")
print(f"Completed forward-return observations: {len(test)}")
print("Forward horizon: 5 trading rows from signal date")
print(f"Assumed round-trip cost: {ROUND_TRIP_COST * 100:.2f}%")

# 1. Score distribution and performance
score_rows = []

for score in range(7):
    group = test.loc[test["technical_score"] == score]
    row = {"Score": score}
    row.update(summarize(group))
    score_rows.append(row)

print_table("1. TECHNICAL SCORE PERFORMANCE", score_rows)

# 2. ADX strength regimes
regimes = [
    ("Weak", test["ADX"] < 20),
    ("Developing", (test["ADX"] >= 20) & (test["ADX"] < 25)),
    ("Strong", test["ADX"] >= 25),
]

regime_rows = []

for name, mask in regimes:
    row = {"ADX regime": name}
    row.update(summarize(test.loc[mask]))
    regime_rows.append(row)

print_table("2. ADX REGIME PERFORMANCE", regime_rows)

# 3. ADX direction and strength
bullish = (
    (test["ADX"] >= 25)
    & (test["Plus_DI"] > test["Minus_DI"])
)

bearish = (
    (test["ADX"] >= 25)
    & (test["Minus_DI"] > test["Plus_DI"])
)

direction_rows = []

for name, mask in [
    ("Strong bullish", bullish),
    ("Strong bearish", bearish),
]:
    row = {"Direction": name}
    row.update(summarize(test.loc[mask]))
    direction_rows.append(row)

print_table("3. STRONG-TREND DIRECTION PERFORMANCE", direction_rows)

# 4. Combined-condition analysis
# Compare individual filters with increasingly specific setups.
setups = [
    ("Score >= 4", test["technical_score"] >= 4),

    ("Strong ADX", test["ADX"] >= 25),

    ("Strong bullish ADX", bullish),

    (
        "Score >= 4 + strong bullish ADX",
        (test["technical_score"] >= 4) & bullish,
    ),

    (
        "Score 5 + strong bullish ADX",
        (test["technical_score"] == 5) & bullish,
    ),

    (
        "Score 6 + strong bullish ADX",
        (test["technical_score"] == 6) & bullish,
    ),
]

combination_rows = []

for name, mask in setups:
    row = {"Setup": name}
    row.update(summarize(test.loc[mask]))
    combination_rows.append(row)

print_table("4. COMBINED-CONDITION ANALYSIS", combination_rows)

# 5. Annual performance of the strong bullish regime
annual_rows = []

annual_test = test.loc[bullish].copy()

for year, group in annual_test.groupby(annual_test.index.year):
    row = {"Year": year}
    row.update(summarize(group))
    annual_rows.append(row)

print_table("5. STRONG BULLISH ADX — YEARLY PERFORMANCE", annual_rows)

# 6. Explicit research limitations
print("\n" + "=" * 76)
print("INTERPRETATION NOTES")
print("=" * 76)
print("- Forward-return observations overlap; they are not independent trades.")
print("- Historical average returns are not forecasts.")
print("- Costs exclude any additional slippage or market impact.")
print("- Combined filters require out-of-sample validation.")
print("- ATR strategy results are reported separately below.")
#---------------------------------------------------------
# EMA pullback / reclaim: ATR position-based backtest
# ---------------------------------------------------------

previous_close = df["close"].shift(1)
previous_ema = df["EMA_20"].shift(1)

df["pullback_reclaim"] = (
    (df["EMA_20"] > df["SMA_20"]) &
    (previous_close <= previous_ema) &
    (df["close"] > df["EMA_20"]) &
    (df["RSI"] > 50) &
    (df["volume"] > df["vol_avg"])
)

ATR_STOP_MULTIPLIER = 1.5
REWARD_RISK_RATIO = 2.0
MAX_HOLD_DAYS = 10

trade_records = []
i = 0

while i < len(df) - 1:

    if not bool(df["pullback_reclaim"].iloc[i]):
        i += 1
        continue

    signal_i = i
    entry_i = signal_i + 1

    # Need enough remaining data for a complete holding window.
    if entry_i + MAX_HOLD_DAYS - 1 >= len(df):
        break

    entry_price = float(df["open"].iloc[entry_i])
    atr_value = float(df["ATR_14"].iloc[signal_i])

    if (
        not pd.notna(entry_price)
        or not pd.notna(atr_value)
        or entry_price <= 0
        or atr_value <= 0
    ):
        i += 1
        continue

    # ATR is taken from the signal date, before entry.
    initial_risk = ATR_STOP_MULTIPLIER * atr_value
    stop_price = entry_price - initial_risk
    target_price = entry_price + (
        REWARD_RISK_RATIO * initial_risk
    )

    exit_i = entry_i + MAX_HOLD_DAYS - 1
    exit_price = float(df["close"].iloc[exit_i])
    exit_reason = "Time exit"

    # Simulate each daily candle while the position is open.
    for j in range(entry_i, exit_i + 1):

        bar = df.iloc[j]

        day_open = float(bar["open"])
        day_high = float(bar["high"])
        day_low = float(bar["low"])

        # Handle gaps at the market open first.
        if day_open <= stop_price:
            exit_i = j
            exit_price = day_open
            exit_reason = "Stop gap" if day_open < stop_price else "Stop"
            break

        if day_open >= target_price:
            exit_i = j
            exit_price = day_open
            exit_reason = "Target gap" if day_open > target_price else "Target"
            break

        stop_hit = day_low <= stop_price
        target_hit = day_high >= target_price

        # Daily candles cannot reveal which level was hit first.
        if stop_hit and target_hit:
            exit_i = j
            exit_price = stop_price
            exit_reason = "Both touched; stop assumed first"
            break

        if stop_hit:
            exit_i = j
            exit_price = stop_price
            exit_reason = "Stop"
            break

        if target_hit:
            exit_i = j
            exit_price = target_price
            exit_reason = "Target"
            break

    gross_return = exit_price / entry_price - 1
    net_return = gross_return - ROUND_TRIP_COST

    trade_records.append({
        "signal_date": df.index[signal_i].date(),
        "entry_date": df.index[entry_i].date(),
        "exit_date": df.index[exit_i].date(),
        "entry": entry_price,
        "stop": stop_price,
        "target": target_price,
        "exit": exit_price,
        "exit_reason": exit_reason,
        "net_return": net_return,
    })

    # Resume scanning on the exit date. Any new signal there
    # can only enter on the following trading day's open.
    i = exit_i

# ---------------------------------------------------------
# Strategy results
# ---------------------------------------------------------

print("\nEMA pullback/reclaim — ATR exit strategy:")

if trade_records:

    trades_df = pd.DataFrame(trade_records)
    returns = trades_df["net_return"]

    wins = returns[returns > 0]
    losses = returns[returns < 0]

    win_rate = (returns > 0).mean() * 100
    avg_return = returns.mean() * 100
    median_return = returns.median() * 100

    gross_profit = wins.sum()
    gross_loss = abs(losses.sum())

    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
        profit_factor_text = f"{profit_factor:.2f}"
    else:
        profit_factor_text = "N/A (no losing trades)"

    equity = (1 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1
    max_drawdown = drawdown.min() * 100

    print(f"Trades          : {len(trades_df)}")
    print(f"Win rate        : {win_rate:.1f}%")
    print(f"Average return  : {avg_return:.2f}%")
    print(f"Median return   : {median_return:.2f}%")
    print(f"Profit factor   : {profit_factor_text}")
    print(f"Max drawdown    : {max_drawdown:.2f}%")
    print(f"Compounded return: {(equity.iloc[-1] - 1) * 100:.2f}%")

    print("\nIndividual trades:")

    for _, trade in trades_df.iterrows():
        print(
            f"Signal {trade['signal_date']} | "
            f"Entry {trade['entry_date']} @ {trade['entry']:.2f} | "
            f"Stop {trade['stop']:.2f} | "
            f"Target {trade['target']:.2f} | "
            f"Exit {trade['exit_date']} @ {trade['exit']:.2f} | "
            f"{trade['exit_reason']} | "
            f"Net {trade['net_return'] * 100:.2f}%"
        )

else:
    print("No qualifying trades.")

# ---------------------------------------------------------
# BUY-AND-HOLD BENCHMARK
# Compare against holding the stock over the strategy period
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("BUY-AND-HOLD BENCHMARK")
print("=" * 60)

if trade_records:
    first_trade = trades_df.iloc[0]
    last_trade = trades_df.iloc[-1]

    start_date = pd.Timestamp(first_trade["entry_date"])
    end_date = pd.Timestamp(last_trade["exit_date"])

    # The benchmark starts at the same first entry price
    # and ends at the last exit date.
    benchmark_start_price = float(first_trade["entry"])
    benchmark_end_price = float(
        df.loc[:end_date, "close"].iloc[-1]
    )

    benchmark_return = (
        benchmark_end_price / benchmark_start_price - 1
    ) * 100

    strategy_return = (
        (1 + trades_df["net_return"]).prod() - 1
    ) * 100

    print(f"Ticker: {TICKER}")
    print(f"Start date: {start_date.date()}")
    print(f"End date:   {end_date.date()}")
    print(f"Benchmark starting price: {benchmark_start_price:.2f}")
    print(f"Benchmark ending price:   {benchmark_end_price:.2f}")
    print()
    print(f"Strategy compounded return: {strategy_return:.2f}%")
    print(f"Buy-and-hold return:        {benchmark_return:.2f}%")
    print(
        f"Strategy minus benchmark:   "
        f"{strategy_return - benchmark_return:+.2f} percentage points"
    )

    print(
        "\nNote: The benchmark uses Yahoo Finance's "
        "auto-adjusted prices. Strategy costs are included; "
        "the benchmark does not deduct trading costs."
    )

else:
    print("Benchmark unavailable: no completed strategy trades.")

# ---------------------------------------------------------
# DATA AND TRADE-DATE SANITY CHECK
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("DATA SANITY CHECK")
print("=" * 60)

print("\nFirst and last trading dates:")
print(df.index.min())
print(df.index.max())

print("\nWeekend dates in the price data:")
weekend_rows = df[df.index.dayofweek >= 5]
print(weekend_rows[["open", "high", "low", "close"]].head(10))

if weekend_rows.empty:
    print("No weekend rows found.")

print("\nTrade entry and exit dates:")
if "trades_df" in globals() and not trades_df.empty:
    for _, trade in trades_df.iterrows():
        entry_date = pd.Timestamp(trade["entry_date"])
        exit_date = pd.Timestamp(trade["exit_date"])

        print(
            f"Entry: {entry_date.date()} "
            f"(weekday={entry_date.day_name()}) | "
            f"Exit: {exit_date.date()} "
            f"(weekday={exit_date.day_name()})"
        )
else:
    print("No trade records available to inspect.")
# ---------------------------------------------------------
# Dataset information
# ---------------------------------------------------------

print("\nDataset:")
print(f"Loaded {len(df)} usable trading days")
print(f"From {df.index[0].date()} to {df.index[-1].date()}")