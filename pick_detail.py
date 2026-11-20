"""For chosen contracts: yfinance price history of the contract itself + the spot level needed for M x by date.

Usage: python pick_detail.py [--mult 40] TSLA261218C00480000 NVDA261218C00200000 ...
"""
import argparse, re, time
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import brentq

from scan_40x import bs_price, implied_vol, n_sessions

ap = argparse.ArgumentParser()
ap.add_argument("contracts", nargs="+")
ap.add_argument("--mult", type=float, default=40.0)
a = ap.parse_args()

C = pd.read_parquet("data/chains.parquet").set_index("contractSymbol")
H = pd.read_parquet("data/history.parquet").dropna(subset=["Close"])
asof = H.Date.max().strftime("%Y-%m-%d")
rows, hist = [], []
for c in a.contracts:
    q = C.loc[c]
    sym, is_call = q.sym, q.side == "C"
    und = H[H.sym == sym].set_index("Date").Close
    S0, K, ask, bid = und.iloc[-1], q.strike, q.ask, q.bid
    h = n_sessions(asof, q.expiry)
    iv = float(implied_vol(np.array([0.5 * (bid + ask)]), S0, np.array([K]), h / 252, is_call)[0])
    target = a.mult * ask
    need = {}
    for left in (15, 10, 5, 0):  # sessions remaining when the target is hit
        if left >= h:
            continue
        if left == 0:
            s = K + target if is_call else K - target
        else:
            f = lambda s: float(bs_price(s, K, left / 252, iv, is_call)) - target
            s = brentq(f, S0, S0 * 10) if is_call else brentq(f, S0 * 0.01, S0)
        need[f"need_{left}left"] = f"{s:.1f} ({s / S0 - 1:+.0%})"
    time.sleep(0.5)
    oh = yf.Ticker(c).history(period="6mo", auto_adjust=False)
    oh.index = oh.index.tz_localize(None).normalize()
    oh = oh[oh.Volume > 0]
    hist.append(oh.assign(contract=c)[["Close", "Volume", "contract"]])
    m1 = oh.loc[:pd.Timestamp(asof) - pd.Timedelta(days=30)]
    rows.append(dict(contract=c, spot=round(S0, 2), bid=bid, ask=ask, iv=f"{iv:.0%}", oi=int(q.openInterest),
                     target=round(target, 2), **need,
                     px_1m_ago=round(m1.Close.iloc[-1], 2) if len(m1) else np.nan,
                     und_1m_ago=round(und.loc[:m1.index[-1]].iloc[-1], 2) if len(m1) else np.nan,
                     hi_6m=round(oh.Close.max(), 2), hi_date=oh.Close.idxmax().date(), lo_6m=round(oh.Close.min(), 2),
                     last=round(oh.Close.iloc[-1], 2), n_days=len(oh)))
pd.concat(hist).reset_index(names="Date").to_csv("data/pick_history.csv", index=False)
pd.set_option("display.width", 280)
print(pd.DataFrame(rows).to_string(index=False))
