"""Multi-ticker backtest of the EMA pullback/reclaim strategy with ATR exits.

Run:  python backtest.py

Everything you would normally tweak is in the CONFIG block below.
Trades from every ticker are pooled so you get hundreds of trades, not eight.
"""
import os
import re

import numpy as np
import pandas as pd

from indicators import add_indicators, load_prices, pullback_reclaim_signal

# ============================== CONFIG ==============================
TICKERS = [
    "RELIANCE.NS", "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "AXISBANK.NS",
    "KOTAKBANK.NS", "TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS",
    "BHARTIARTL.NS", "ITC.NS", "HINDUNILVR.NS", "LT.NS", "MARUTI.NS",
    "M&M.NS", "TITAN.NS", "BAJFINANCE.NS", "SUNPHARMA.NS", "DRREDDY.NS",
    "CIPLA.NS", "NTPC.NS", "POWERGRID.NS", "ONGC.NS", "COALINDIA.NS",
    "BPCL.NS", "TATASTEEL.NS", "JSWSTEEL.NS", "HINDALCO.NS", "ADANIPORTS.NS",
    "ULTRACEMCO.NS", "ASIANPAINT.NS", "SUZLON.NS",
]

START = "2021-01-01"          # fixed dates so results are reproducible
END = "2026-10-01"
SPLIT_DATE = "2024-07-01"     # entries before = in-sample, on/after = out-of-sample

ROUND_TRIP_COST = 0.003       # 0.30% brokerage + STT + charges + slippage (check your broker)
COST_SENSITIVITY = [0.001, 0.003, 0.005]

ATR_STOP_MULTIPLIER = 1.5
REWARD_RISK_RATIO = 2.0
MAX_HOLD_DAYS = 10

RISK_PER_TRADE = 0.01         # risk 1% of equity per trade
MAX_POSITION = 1.0            # never more than 100% of equity in one trade (no leverage)

CACHE_DIR = "price_cache"
# ====================================================================


def get_prices(ticker):
    """Download once, then reuse the cached CSV so reruns are fast and identical."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", ticker)
    path = os.path.join(CACHE_DIR, f"{safe}_{START}_{END}.csv")
    if os.path.exists(path):
        return pd.read_csv(path, index_col=0, parse_dates=True)
    df = load_prices(ticker, START, END)
    if df is not None and not df.empty:
        df.to_csv(path)
    return df


def prepare(df):
    """Add indicators and the entry signal; drop rows where indicators aren't ready."""
    df = add_indicators(df)
    df["signal"] = pullback_reclaim_signal(df)
    needed = ["SMA_20", "EMA_20", "RSI", "vol_avg", "ATR_14", "ADX", "VWAP"]
    return df.dropna(subset=needed)


def simulate(df, ticker):
    """Walk the data once, taking non-overlapping trades. Returns (trades, info)."""
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    atr = df["ATR_14"].to_numpy(float)
    sig = df["signal"].to_numpy(bool)
    dates = df.index
    n = len(df)

    trades, days_in_trade, dropped_incomplete = [], 0, 0
    i = 0
    while i < n - 1:
        if not sig[i]:
            i += 1
            continue

        entry_i = i + 1
        last_i = entry_i + MAX_HOLD_DAYS - 1
        if last_i >= n:                       # not enough data left for a full window
            dropped_incomplete += 1
            break

        entry = o[entry_i]
        risk = ATR_STOP_MULTIPLIER * atr[i]   # ATR known on signal date, before entry
        if not (entry > 0 and risk > 0):
            i += 1
            continue
        stop = entry - risk
        target = entry + REWARD_RISK_RATIO * risk

        exit_i, exit_price, reason = last_i, c[last_i], "Time exit"
        for j in range(entry_i, last_i + 1):
            # Gaps can only happen after the entry bar (we enter at its open).
            if j > entry_i and o[j] <= stop:
                exit_i, exit_price, reason = j, o[j], "Stop (gap)"
                break
            if j > entry_i and o[j] >= target:
                exit_i, exit_price, reason = j, o[j], "Target (gap)"
                break
            stop_hit, target_hit = l[j] <= stop, h[j] >= target
            if stop_hit:                      # includes "both hit": assume stop first
                exit_i, exit_price = j, stop
                reason = "Stop+Target same day" if target_hit else "Stop"
                break
            if target_hit:
                exit_i, exit_price, reason = j, target, "Target"
                break

        gross = exit_price / entry - 1
        trades.append({
            "ticker": ticker,
            "signal_date": dates[i], "entry_date": dates[entry_i], "exit_date": dates[exit_i],
            "entry": entry, "stop": stop, "target": target, "exit": exit_price,
            "exit_reason": reason, "hold_days": exit_i - entry_i + 1,
            "gross_return": gross, "risk_frac": risk / entry,
        })
        days_in_trade += exit_i - entry_i + 1
        i = exit_i                            # a new signal on the exit day enters next open

    first_open = o[0]
    info = {
        "bars": n,
        "exposure": days_in_trade / n if n else np.nan,
        "buy_hold": c[-1] / first_open - 1 if n else np.nan,
        "dropped_incomplete": dropped_incomplete,
    }
    return trades, info


def apply_costs(trades, cost):
    """Add net return, R-multiple and sized portfolio impact for a given cost."""
    t = trades.copy()
    t["net_return"] = t["gross_return"] - cost
    t["R"] = t["net_return"] / t["risk_frac"]
    t["position"] = (RISK_PER_TRADE / t["risk_frac"]).clip(upper=MAX_POSITION)
    t["equity_impact"] = t["position"] * t["net_return"]
    return t


def stats(t):
    """Summary statistics for a set of trades (already cost-adjusted)."""
    if t.empty:
        return {k: np.nan for k in
                ["Trades", "Win %", "Avg ret %", "Avg R", "Avg win R", "Avg loss R", "Profit factor"]}
    wins, losses = t[t["net_return"] > 0], t[t["net_return"] < 0]
    gl = abs(losses["net_return"].sum())
    return {
        "Trades": len(t),
        "Win %": (t["net_return"] > 0).mean() * 100,
        "Avg ret %": t["net_return"].mean() * 100,
        "Avg R": t["R"].mean(),
        "Avg win R": wins["R"].mean() if len(wins) else np.nan,
        "Avg loss R": losses["R"].mean() if len(losses) else np.nan,
        "Profit factor": wins["net_return"].sum() / gl if gl > 0 else np.inf,
    }


def show(title, rows, index=None):
    print(f"\n{'=' * 90}\n{title}\n{'=' * 90}")
    table = pd.DataFrame(rows)
    if index and index in table.columns:
        table = table.set_index(index)
    with pd.option_context("display.float_format", "{:.2f}".format, "display.width", 200):
        print(table.to_string())


def pooled_equity(t):
    """Approximate pooled equity curve: trades sorted by exit date, compounded one after
    another. Ignores that trades in different tickers overlap in time, so treat the
    drawdown as a rough guide only."""
    t = t.sort_values("exit_date")
    equity = (1 + t["equity_impact"]).cumprod()
    return equity, (equity / equity.cummax() - 1).min() * 100


def run(tickers=None, loader=get_prices):
    tickers = tickers or TICKERS
    all_trades, per_ticker, failed = [], {}, []

    for tk in tickers:
        raw = loader(tk)
        if raw is None or len(raw) < 100:
            failed.append(tk)
            continue
        df = prepare(raw)
        trades, info = simulate(df, tk)
        per_ticker[tk] = info
        all_trades += trades

    if not all_trades:
        print("No trades produced.")
        return None

    base = pd.DataFrame(all_trades)
    t = apply_costs(base, ROUND_TRIP_COST)

    print(f"Period {START} to {END} | cost {ROUND_TRIP_COST * 100:.2f}% round trip | "
          f"stop {ATR_STOP_MULTIPLIER} ATR | target {REWARD_RISK_RATIO}R | max hold {MAX_HOLD_DAYS}d")
    print(f"Tickers tested: {len(per_ticker)}" + (f" | failed to load: {failed}" if failed else ""))
    be = 1 / (1 + REWARD_RISK_RATIO) * 100
    print(f"Break-even win rate at {REWARD_RISK_RATIO}R before costs: about {be:.1f}%")

    # 1. Per ticker
    rows = []
    for tk, g in t.groupby("ticker"):
        s, info = stats(g), per_ticker[tk]
        ret100 = ((1 + g.sort_values("exit_date")["net_return"]).prod() - 1) * 100
        rows.append({"Ticker": tk, "Trades": s["Trades"], "Win %": s["Win %"],
                     "Avg R": s["Avg R"], "Profit factor": s["Profit factor"],
                     "Strategy % (100% cap)": ret100, "Buy&hold %": info["buy_hold"] * 100,
                     "Time in trade %": info["exposure"] * 100})
    show("1. PER TICKER (small samples: don't read too much into any single row)", rows, "Ticker")

    # 2. Pooled
    pooled = stats(t)
    _, mdd = pooled_equity(t)
    sized_total = (pooled_equity(t)[0].iloc[-1] - 1) * 100
    show("2. POOLED RESULTS (all tickers)", [pooled])
    print(f"\nPooled sized equity ({RISK_PER_TRADE * 100:.0f}% risk/trade, approx.): "
          f"{sized_total:.1f}% total, rough max drawdown {mdd:.1f}%")
    print(f"Expectancy: {pooled['Avg R']:.3f} R per trade "
          f"(positive = edge after costs, but check the sample size above)")

    # 3. Exit reasons
    rows = []
    for reason, g in t.groupby("exit_reason"):
        rows.append({"Exit reason": reason, "Trades": len(g), "Share %": len(g) / len(t) * 100,
                     "Avg R": g["R"].mean(), "Avg hold (days)": g["hold_days"].mean()})
    show("3. HOW TRADES END", rows, "Exit reason")

    # 4. In-sample vs out-of-sample
    split = pd.Timestamp(SPLIT_DATE)
    rows = []
    for name, g in [(f"In-sample (entry < {SPLIT_DATE})", t[t["entry_date"] < split]),
                    (f"Out-of-sample (entry >= {SPLIT_DATE})", t[t["entry_date"] >= split])]:
        rows.append({"Period": name, **stats(g)})
    show("4. IN-SAMPLE VS OUT-OF-SAMPLE (tune only on the first; judge on the second)", rows, "Period")

    # 5. Cost sensitivity
    rows = []
    for cost in COST_SENSITIVITY:
        rows.append({"Round-trip cost %": cost * 100, **stats(apply_costs(base, cost))})
    show("5. COST SENSITIVITY", rows, "Round-trip cost %")

    # 6. Yearly
    rows = []
    for year, g in t.groupby(t["entry_date"].dt.year):
        rows.append({"Entry year": year, **stats(g)})
    show("6. BY ENTRY YEAR", rows, "Entry year")

    t.to_csv("trades.csv", index=False)
    print("\nSaved every trade to trades.csv")
    print("Notes: trades in different tickers overlap in time, buy&hold ignores costs, and "
          "daily bars can't show whether stop or target hit first (stop assumed).")
    return t


if __name__ == "__main__":
    run()