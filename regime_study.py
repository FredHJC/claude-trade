"""Regime-conditional study: how did the underlying behave after states that look like today's?

State of a symbol on a date = (drawdown from 252-session high, 21-session return). We keep the historical
dates whose state lies in a band around today's state, then replay the following h sessions and reprice
today's option chain along those paths (sticky-strike Black-Scholes) to count how often a contract bought
at today's ask would have been worth >= M x.

Usage: python regime_study.py [--mult 40] SPY QQQ SMH ...
"""
import argparse, json
import numpy as np
import pandas as pd
import yfinance as yf

from scan_40x import bs_price, implied_vol, n_sessions, touch_stats

DEFAULT = ["SPY", "QQQ", "IWM", "SMH", "SOXL", "TQQQ", "NVDA", "AMD", "TSLA", "MU", "AVGO", "UVXY"]


def load_history(sym):
    """Longest available daily history: data/history_max.parquet cache -> Yahoo 'max' -> 10y cache."""
    import os
    path = "data/history_max.parquet"
    cache = pd.read_parquet(path) if os.path.exists(path) else pd.DataFrame(columns=["Date", "Close", "sym"])
    g = cache[cache.sym == sym]
    if g.empty:
        try:
            h = yf.Ticker(sym).history(period="max", auto_adjust=True).dropna(subset=["Close"])
            g = pd.DataFrame({"Date": h.index.tz_localize(None), "Close": h.Close.to_numpy(), "sym": sym})
            pd.concat([cache, g], ignore_index=True).to_parquet(path, index=False)
        except Exception as ex:
            print(f"{sym}: max-history fetch failed ({type(ex).__name__}); falling back to 10y cache")
            g = pd.read_parquet("data/history.parquet").dropna(subset=["Close"])
            g = g[g.sym == sym][["Date", "Close", "sym"]]
            return pd.DatetimeIndex(g.Date), g.Close.to_numpy(), "10y"
    return pd.DatetimeIndex(g.Date), g.Close.to_numpy(), "max"


def states(close):
    c = pd.Series(close)
    return (c / c.rolling(252, min_periods=120).max() - 1).to_numpy(), (c / c.shift(21) - 1).to_numpy()


def fwd_paths(close, idx, h, atm_iv=None):
    """Gross-return paths over the h sessions after each index. With atm_iv, each window's log returns are
    rescaled from its own prior-60-session realized vol to atm_iv (clipped to 0.5x..2x)."""
    idx = idx[idx + h < len(close)]
    lr = np.diff(np.log(close))  # lr[j] = return of session j+1
    win = np.stack([lr[i:i + h] for i in idx])
    if atm_iv is not None:
        vol = pd.Series(lr).rolling(60).std().to_numpy() * np.sqrt(252)
        win = win * np.clip(atm_iv / np.maximum(vol[idx - 1], 1e-6), 0.5, 2.0)[:, None]
    return np.exp(np.cumsum(win, axis=1)), idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("symbols", nargs="*", default=DEFAULT)
    ap.add_argument("--mult", type=float, default=40.0)
    ap.add_argument("--lo", type=float, default=0.6, help="band: state between lo x and hi x today's values")
    ap.add_argument("--hi", type=float, default=1.8)
    a = ap.parse_args()
    M = a.mult

    meta = json.load(open("data/meta.json"))["meta"]
    C = pd.read_parquet("data/chains.parquet")
    summary, picks = [], []
    for sym in a.symbols:
        dates, close, src = load_history(sym)
        asof = dates[-1].strftime("%Y-%m-%d")
        dd, mom = states(close)
        dd0, mom0 = dd[-1], mom[-1]
        if not (dd0 < -0.03 and mom0 < -0.03):
            print(f"{sym}: not in a drawdown state (dd={dd0:.1%}, 21d={mom0:.1%}) - skipped")
            continue
        in_band = (dd <= a.lo * dd0) & (dd >= a.hi * dd0) & (mom <= a.lo * mom0) & (mom >= a.hi * mom0)
        all_idx = np.arange(252, len(close))
        reg_idx = np.where(in_band)[0]
        reg_idx = reg_idx[reg_idx < len(close) - 1]
        S0 = meta[sym]["spot"]

        for h in (10, 20):
            P_all, _ = fwd_paths(close, all_idx, h)
            P_reg, used = fwd_paths(close, reg_idx, h)
            episodes = 1 + int((np.diff(used) > 20).sum()) if len(used) else 0
            for name, P in (("all", P_all), ("regime", P_reg)):
                r = P[:, -1] - 1
                summary.append(dict(sym=sym, hist=f"{dates[0].year}-", h=h, set=name, n=len(P), episodes=episodes if name == "regime" else np.nan,
                                    dd_now=dd0, mom21_now=mom0, mean=r.mean(), median=np.median(r),
                                    p05=np.quantile(r, .05), p95=np.quantile(r, .95),
                                    up10=(P.max(1) >= 1.10).mean(), up20=(P.max(1) >= 1.20).mean(),
                                    dn10=(P.min(1) <= 0.90).mean(), dn20=(P.min(1) <= 0.80).mean()))

        # reprice today's chain along the regime paths
        for expiry, G in C[C.sym == sym].groupby("expiry"):
            h = n_sessions(asof, expiry)
            if h < 4:
                continue
            P_reg, used = fwd_paths(close, reg_idx, h)
            P_all, _ = fwd_paths(close, all_idx, h)
            if len(P_reg) < 30:
                continue
            q = G[(G.bid > 0) & (G.ask > G.bid)]
            near = pd.concat([q[(q.side == sd) & ((q.strike > S0) if sd == "C" else (q.strike < S0))]
                              .assign(d=lambda x: (x.strike - S0).abs()).nsmallest(2, "d") for sd in "CP"])
            atm_iv = float(np.mean([implied_vol(np.array([0.5 * (b + k)]), S0, np.array([x]), h / 252, sd == "C")[0]
                                    for b, k, x, sd in zip(near.bid, near.ask, near.strike, near.side)]))
            P_regv, _ = fwd_paths(close, reg_idx, h, atm_iv)
            episodes = 1 + int((np.diff(used) > 20).sum())
            for side in "CP":
                is_call = side == "C"
                g = G[(G.side == side) & (G.bid > 0) & (G.ask >= 0.03) & ((G.ask - G.bid) / G.ask <= 0.6)
                      & (G.openInterest >= 100) & ((G.strike > S0) if is_call else (G.strike < S0))]
                if g.empty:
                    continue
                K, bid, ask = (g[c].to_numpy(float) for c in ("strike", "bid", "ask"))
                iv = implied_vol(0.5 * (bid + ask), S0, K, h / 252, is_call)
                pe, pt, ev = touch_stats(P_reg, S0, K, iv, ask, is_call, M)
                _, pt_all, _ = touch_stats(P_all, S0, K, iv, ask, is_call, M)
                pev, ptv, evv = touch_stats(P_regv, S0, K, iv, ask, is_call, M)
                picks.append(pd.DataFrame(dict(sym=sym, side=side, expiry=expiry, h=h, strike=K, spot=S0, otm=K / S0 - 1,
                                               bid=bid, ask=ask, iv=iv, oi=g.openInterest.to_numpy(), n_reg=len(P_reg),
                                               episodes=episodes, atm_iv=atm_iv, p_reg_exp=pe, p_reg_touch=pt, ev_reg=ev,
                                               p_regv_exp=pev, p_regv_touch=ptv, ev_regv=evv, p_all_touch=pt_all,
                                               contract=g.contractSymbol.to_numpy())))

    S = pd.DataFrame(summary)
    S.to_csv("data/regime_summary.csv", index=False)
    Pk = pd.concat(picks, ignore_index=True)
    Pk.to_csv(f"data/regime_picks_{M:g}.csv", index=False)
    pd.set_option("display.width", 250)
    pct = lambda x: f"{x:.1%}"
    print(S.to_string(index=False, formatters={c: pct for c in
                                                ["dd_now", "mom21_now", "mean", "median", "p05", "p95", "up10", "up20", "dn10", "dn20"]}))


if __name__ == "__main__":
    main()
