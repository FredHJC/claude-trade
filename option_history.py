"""Pull daily price history of individual option contracts (yfinance) and measure which ones multiplied.

Real-data base rate: take every listed contract of one expiry, look at its close on a start date ~1 month
earlier, and track the best close / final close afterwards.

Usage: python option_history.py --expiry 2026-11-20 --start 2026-10-20 SPY QQQ NVDA ...
Writes data/opt_hist_<expiry>.parquet (daily bars) and data/opt_mult_<expiry>.csv (per-contract multiples)
"""
import argparse, time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import yfinance as yf


def contract_hist(sym):
    """Daily bars of one option contract. Backs off for a minute whenever Yahoo answers 429."""
    for attempt in range(4):
        try:
            time.sleep(0.25)
            h = yf.Ticker(sym).history(period="3mo", auto_adjust=False)
            if h is None or h.empty:
                return None
            h = h[["Open", "High", "Low", "Close", "Volume"]].copy()
            h.index = h.index.tz_localize(None).normalize()
            h["contract"] = sym
            return h.reset_index().rename(columns={"index": "Date"})
        except Exception as ex:
            time.sleep(60 if "RateLimit" in type(ex).__name__ else 2 * (attempt + 1))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("symbols", nargs="+")
    ap.add_argument("--expiry", required=True)
    ap.add_argument("--start", required=True, help="entry date (first bar on/after this date is the entry)")
    ap.add_argument("--band", type=float, default=0.35, help="keep strikes within +-band of spot")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--min-oi", type=int, default=200)
    a = ap.parse_args()

    rows = []
    for s in a.symbols:
        try:
            t = yf.Ticker(s)
            ch = t.option_chain(a.expiry)
            spot = ch.underlying.get("regularMarketPrice")
            for side, df in (("C", ch.calls), ("P", ch.puts)):
                d = df[(df.strike.between(spot * (1 - a.band), spot * (1 + a.band))) & (df.openInterest.fillna(0) >= a.min_oi)]
                rows += [dict(sym=s, side=side, strike=k, contract=c, spot_now=spot)
                         for k, c in zip(d.strike, d.contractSymbol)]
        except Exception as ex:
            print(s, "chain error", repr(ex)[:100])
    L = pd.DataFrame(rows)
    print(f"{len(L)} contracts to fetch", flush=True)
    with ThreadPoolExecutor(a.workers) as ex:
        hist = [h for h in ex.map(contract_hist, L.contract) if h is not None]
    Hh = pd.concat(hist, ignore_index=True)
    Hh.to_parquet(f"data/opt_hist_{a.expiry}.parquet", index=False)

    start = pd.Timestamp(a.start)
    out = []
    for c, g in Hh.groupby("contract"):
        g = g[g.Date >= start].sort_values("Date")
        if len(g) < 2 or g.Date.iloc[0] > start + pd.Timedelta(days=4):
            continue
        e = g.iloc[0]
        if not (e.Close > 0) or e.Volume <= 0:  # need a real trade on the entry day
            continue
        later = g.iloc[1:]
        ipk = later.Close.to_numpy().argmax()
        out.append(dict(contract=c, entry_date=e.Date.date(), entry=e.Close, peak=later.Close.iloc[ipk],
                        peak_date=later.Date.iloc[ipk].date(), last=later.Close.iloc[-1],
                        last_date=later.Date.iloc[-1].date(), peak_mult=later.Close.iloc[ipk] / e.Close,
                        last_mult=later.Close.iloc[-1] / e.Close))
    R = L.merge(pd.DataFrame(out), on="contract")
    R.to_csv(f"data/opt_mult_{a.expiry}.csv", index=False)
    print(f"bars={len(Hh)} contracts_with_entry={len(R)}")


if __name__ == "__main__":
    main()
