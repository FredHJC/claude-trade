"""Scan option chains for contracts / vertical spreads that could return M-times (default 40x) by expiry.

For every liquid OTM contract we estimate the probability of reaching M x the ask:
  p_rn        risk-neutral (market-implied) probability that the option is worth >= M x ask AT expiry
  p_raw_*     frequency in the underlying's own 10y history (overlapping h-session windows)
  p_fhs_*     same windows, but each window's returns are rescaled from its prior-60d realized vol
              to today's ATM implied vol (historical tail *shape*, market-implied vol *level*)
  *_exp       40x at expiry (intrinsic value only)
  *_touch     40x at ANY daily close before/at expiry (Black-Scholes repricing, sticky-strike IV)
  ev_*        expected payoff multiple of "sell at M x if touched, else hold to expiry"

Usage: python scan_40x.py [--mult 40] [--min-ask 0.03]
Reads data/*.parquet + data/meta.json (see fetch_data.py); writes data/scan_naked_<M>.csv, data/scan_vertical_<M>.csv
"""
import argparse, json
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy.special import ndtr

HOLIDAYS = ["2026-11-26", "2026-12-25"]
LOOKBACK = 60  # sessions of realized vol used to standardise each historical window


def n_sessions(asof, expiry):
    return int(np.busday_count(np.datetime64(asof, "D") + np.timedelta64(1, "D"), np.datetime64(expiry, "D") + np.timedelta64(1, "D"), holidays=HOLIDAYS))


def bs_price(S, K, T, iv, is_call):
    sd = iv * np.sqrt(np.maximum(T, 1e-9))
    d1 = (np.log(S / K) + 0.5 * sd * sd) / sd
    call = S * ndtr(d1) - K * ndtr(d1 - sd)
    return call if is_call else call - S + K  # put via parity, r = q = 0


def implied_vol(price, S, K, T, is_call):
    lo, hi = np.full(price.shape, 0.01), np.full(price.shape, 8.0)
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        up = bs_price(S, K, T, mid, is_call) < price
        lo, hi = np.where(up, mid, lo), np.where(up, hi, mid)
    return 0.5 * (lo + hi)


def build_paths(close, h, atm_iv):
    """Return (raw, fhs) cumulative gross-return paths of shape (W, h) from overlapping history windows."""
    r = np.diff(np.log(close))
    if len(r) < LOOKBACK + h + 250:
        return None
    win = sliding_window_view(r, h)
    vol = pd.Series(r).rolling(LOOKBACK).std().to_numpy() * np.sqrt(252)
    idx = np.arange(LOOKBACK, len(r) - h + 1)
    win, prior = win[idx], vol[idx - 1]
    scale = np.clip(atm_iv / np.maximum(prior, 1e-6), 0.5, 2.0)
    return np.exp(np.cumsum(win, axis=1)), np.exp(np.cumsum(win * scale[:, None], axis=1))


def touch_stats(paths, S0, K, iv, ask, is_call, M, chunk=40):
    """P(option value >= M*ask at expiry), P(... at any daily close), EV multiple of the exit rule."""
    W, h = paths.shape
    p_exp, p_touch, ev = (np.zeros(len(K)) for _ in range(3))
    t_rem = (h - 1 - np.arange(h - 1)) / 252.0
    for a in range(0, len(K), chunk):
        k, v, c = K[a:a + chunk], iv[a:a + chunk], ask[a:a + chunk]
        ST = S0 * paths[:, -1][:, None]
        intr = np.maximum(ST - k, 0) if is_call else np.maximum(k - ST, 0)
        target = M * c
        hit_exp = intr >= target
        hit = hit_exp.copy()
        if h > 1:
            V = bs_price(S0 * paths[:, :-1, None], k[None, None, :], t_rem[None, :, None], v[None, None, :], is_call)
            hit |= (V >= target).any(axis=1)
        p_exp[a:a + chunk], p_touch[a:a + chunk] = hit_exp.mean(0), hit.mean(0)
        ev[a:a + chunk] = np.where(hit, target, intr).mean(0) / c
    return p_exp, p_touch, ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mult", type=float, default=40.0)
    ap.add_argument("--min-ask", type=float, default=0.03)
    ap.add_argument("--min-oi", type=int, default=100)
    ap.add_argument("--max-spread", type=float, default=0.6, help="(ask-bid)/ask upper bound")
    a = ap.parse_args()
    M = a.mult

    meta = json.load(open("data/meta.json"))["meta"]
    H = pd.read_parquet("data/history.parquet").dropna(subset=["Close"])
    C = pd.read_parquet("data/chains.parquet")
    asof = H.Date.max().strftime("%Y-%m-%d")
    fresh = pd.Timestamp(asof) - pd.Timedelta(days=1)  # traded within the last two sessions
    closes = {s: g.sort_values("Date").Close.to_numpy() for s, g in H.groupby("sym")}

    naked, vert = [], []
    for (sym, expiry), G in C.groupby(["sym", "expiry"]):
        m = meta.get(sym, {})
        S0, h = m.get("spot"), n_sessions(asof, expiry)
        if not S0 or h < 1 or sym not in closes:
            continue
        T = h / 252.0
        et = m.get("earningsTimestamp")
        earn = pd.Timestamp(et, unit="s").strftime("%Y-%m-%d") if et else ""
        earn_in = bool(earn) and asof < earn <= expiry

        q = G[(G.bid > 0) & (G.ask > G.bid)].copy()
        q["otm"] = np.where(q.side == "C", q.strike > S0, q.strike < S0)
        q = q[q.otm]
        if len(q) < 6:
            continue
        q["mid"] = 0.5 * (q.bid + q.ask)
        for side in ("C", "P"):
            sel = q.side == side
            q.loc[sel, "iv"] = implied_vol(q.loc[sel, "mid"].to_numpy(), S0, q.loc[sel, "strike"].to_numpy(), T, side == "C")
        # ATM vol = average own-IV of the two nearest OTM strikes on each side
        near = pd.concat([q[q.side == s].assign(d=lambda x: (x.strike - S0).abs()).nsmallest(2, "d") for s in "CP"])
        atm_iv = float(near.iv.mean())
        if not (0.03 < atm_iv < 4):
            continue
        paths = build_paths(closes[sym], h, atm_iv)
        if paths is None:
            continue
        raw, fhs = paths
        term_raw, term_fhs = np.sort(raw[:, -1]), np.sort(fhs[:, -1])

        for side in ("C", "P"):
            is_call = side == "C"
            g = q[q.side == side].sort_values("strike", ascending=is_call).reset_index(drop=True)
            if len(g) < 3:
                continue
            K, bid, ask, iv = (g[c].to_numpy(float) for c in ("strike", "bid", "ask", "iv"))
            sgn = 1.0 if is_call else -1.0
            mid = 0.5 * (bid + ask)
            # smile used for tail probabilities: only quotes that are not minimum-tick noise
            rel = (mid >= 0.05) & ((ask - bid) / ask <= 0.6)
            if rel.sum() < 2:
                rel = np.ones(len(K), bool)
            order = np.argsort(K[rel])
            ks, ivs = K[rel][order], iv[rel][order]

            def tail_probs(S_star):
                """P(S_T beyond S_star): risk-neutral, raw history, FHS history.

                Risk-neutral = lognormal with smile-interpolated IV, capped by the model-free bound
                P(S_T beyond S*) <= price(K') / |S* - K'| for every quoted strike K' on the near side of S*."""
                S_star = np.maximum(S_star, 1e-6)
                v = np.interp(S_star, ks, ivs) * np.sqrt(T)
                d2 = (np.log(S0 / S_star) - 0.5 * v * v) / v
                gap = sgn * (S_star[:, None] - K[None, :])
                bound = np.where(gap > 0, mid[None, :] / np.maximum(gap, 1e-9), np.inf).min(axis=1)
                x = S_star / S0
                if is_call:
                    return np.minimum(ndtr(d2), bound), 1 - np.searchsorted(term_raw, x, "left") / len(term_raw), \
                        1 - np.searchsorted(term_fhs, x, "left") / len(term_fhs)
                return np.minimum(ndtr(-d2), bound), np.searchsorted(term_raw, x, "right") / len(term_raw), \
                    np.searchsorted(term_fhs, x, "right") / len(term_fhs)

            # ---- naked long options -------------------------------------------------------------
            liq = ((g.ask >= a.min_ask) & ((g.ask - g.bid) / g.ask <= a.max_spread)
                   & ((g.openInterest >= a.min_oi) | (g.volume >= a.min_oi)) & (g.lastTradeDate >= fresh)).to_numpy()
            if liq.any():
                i = np.where(liq)[0]
                S_star = K[i] + sgn * M * ask[i]
                p_rn, _, _ = tail_probs(S_star)
                pe_r, pt_r, ev_r = touch_stats(raw, S0, K[i], iv[i], ask[i], is_call, M)
                pe_f, pt_f, ev_f = touch_stats(fhs, S0, K[i], iv[i], ask[i], is_call, M)
                naked.append(pd.DataFrame(dict(
                    sym=sym, grp=m.get("group"), side=side, expiry=expiry, h=h, strike=K[i], spot=S0,
                    otm=K[i] / S0 - 1, bid=bid[i], ask=ask[i], spread=(ask[i] - bid[i]) / ask[i], iv=iv[i], atm_iv=atm_iv,
                    oi=g.openInterest.to_numpy()[i], vol=g.volume.to_numpy()[i], need_exp=S_star / S0 - 1,
                    p_rn=p_rn, p_raw_exp=pe_r, p_raw_touch=pt_r, p_fhs_exp=pe_f, p_fhs_touch=pt_f,
                    ev_raw=ev_r, ev_fhs=ev_f, n_win=len(raw), earn=earn, earn_in=earn_in,
                    contract=g.contractSymbol.to_numpy()[i])))

            # ---- vertical debit spreads (synthetic digital): long K_i, short further-OTM K_j -----
            ok = ((g.openInterest >= a.min_oi) & (g.lastTradeDate >= fresh)).to_numpy()
            ii, jj = np.triu_indices(len(g), 1)
            keep = ok[ii] & ok[jj]
            ii, jj = ii[keep], jj[keep]
            if len(ii):
                debit = ask[ii] - bid[jj]  # pay the ask, sell the bid
                debit_mid = mid[ii] - mid[jj]
                width = np.abs(K[jj] - K[ii])
                good = ((debit >= 0.02) & (width / np.maximum(debit, 1e-9) >= M)
                        & (debit_mid >= 0.02) & (debit <= 3 * debit_mid))  # drop crossed / minimum-tick noise
                ii, jj, debit, debit_mid, width = ii[good], jj[good], debit[good], debit_mid[good], width[good]
            if len(ii):
                S_star = K[ii] + sgn * M * debit
                p_rn, p_r, p_f = tail_probs(S_star)
                p_rn = np.minimum(p_rn, debit_mid / (M * debit))  # Markov bound on the spread itself
                d = pd.DataFrame(dict(
                    sym=sym, grp=m.get("group"), side=side, expiry=expiry, h=h, spot=S0, k_long=K[ii], k_short=K[jj],
                    debit=debit, debit_mid=debit_mid, width=width,
                    max_mult=width / debit, need_exp=S_star / S0 - 1, p_rn=p_rn, p_raw_exp=p_r, p_fhs_exp=p_f,
                    atm_iv=atm_iv, n_win=len(raw), earn=earn, earn_in=earn_in))
                vert.append(d.sort_values("p_fhs_exp", ascending=False).head(5))  # best few per chain

    N = pd.concat(naked, ignore_index=True)
    V = pd.concat(vert, ignore_index=True) if vert else pd.DataFrame()
    tag = f"{M:g}"
    N.to_csv(f"data/scan_naked_{tag}.csv", index=False)
    V.to_csv(f"data/scan_vertical_{tag}.csv", index=False)
    print(f"asof={asof} M={M:g} naked={len(N)} contracts over {N.sym.nunique()} symbols; verticals={len(V)}")


if __name__ == "__main__":
    main()
