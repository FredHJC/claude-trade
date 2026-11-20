"""Fetch underlying history, option chains (<= ~1 month out) and earnings dates via yfinance.

Usage: python fetch_data.py [--max-expiry 2026-12-18]
Writes: data/history.parquet, data/chains.parquet, data/meta.json
"""
import argparse, json, sys, time, datetime as dt
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import yfinance as yf

UNIVERSE = {
    # broad index / sector / macro ETFs
    "index": ["SPY", "QQQ", "IWM", "DIA", "SMH", "XBI", "KRE", "XLF", "XLE", "ARKK", "EEM", "FXI", "KWEB",
              "TLT", "HYG", "GLD", "SLV", "GDX", "USO", "UNG", "IBIT", "ETHA"],
    # leveraged / volatility products
    "levered": ["TQQQ", "SQQQ", "SOXL", "SOXS", "UVXY", "VXX", "TSLL", "NVDL", "MSTU"],
    # mega caps
    "mega": ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "AMD", "NFLX", "ORCL", "CRM",
             "ADBE", "INTC", "MU", "QCOM", "TSM", "BABA", "PDD", "JPM", "BAC", "XOM", "UNH", "LLY", "BA", "DIS"],
    # high-vol / retail favourites
    "highvol": ["MSTR", "COIN", "PLTR", "HOOD", "SMCI", "GME", "AMC", "RIVN", "LCID", "NIO", "SOFI", "UPST",
                "AFRM", "RBLX", "SNOW", "CRWD", "SHOP", "UBER", "DKNG", "ROKU", "MARA", "RIOT", "CVNA", "APP",
                "RDDT", "ARM", "IONQ", "RGTI", "QBTS", "OKLO", "SMR", "ASTS", "RKLB", "CRWV", "NBIS", "HIMS",
                "CLSK", "IREN", "SOUN", "BBAI", "ACHR", "JOBY", "LUNR", "OPEN", "CHWY", "DJT", "CELH", "ENPH",
                "MRNA", "NVAX", "SNAP", "PINS", "U", "AI", "PATH", "DELL", "MRVL", "LULU", "DOCU", "MDB", "ZS",
                "OKTA", "S", "GTLB", "ULTA", "DG", "DLTR", "FIVE", "COST", "FDX", "NKE", "CCL", "LEN", "KMX",
                "ANF", "AEO", "BBY", "M", "KSS", "BURL", "WDAY", "ZM", "ADSK", "INTU", "NTAP", "PSTG", "CIEN",
                "HPQ", "HPE", "IOT", "RBRK", "NTNX", "VSCO", "SIG", "TOL", "JBL", "ACN", "DRI", "GIS", "KR"],
}


def fetch_one(sym: str, max_expiry: str):
    out = {"sym": sym, "hist": None, "chains": [], "meta": {}, "err": None}
    try:
        t = yf.Ticker(sym)
        h = t.history(period="10y", auto_adjust=True)
        if h is None or len(h) < 60:
            out["err"] = "no history"
            return out
        h = h[["Open", "High", "Low", "Close", "Volume"]].copy()
        h.index = h.index.tz_localize(None)
        h["sym"] = sym
        out["hist"] = h
        exps = [e for e in (t.options or []) if e <= max_expiry]
        spot = None
        for e in exps:
            for attempt in range(3):
                try:
                    ch = t.option_chain(e)
                    break
                except Exception as ex:  # rate limit etc.
                    time.sleep(1.5 * (attempt + 1))
                    ch = None
            if ch is None:
                continue
            if spot is None and getattr(ch, "underlying", None):
                u = ch.underlying
                spot = u.get("regularMarketPrice")
                out["meta"].update({k: u.get(k) for k in (
                    "regularMarketPrice", "regularMarketPreviousClose", "postMarketPrice", "regularMarketTime",
                    "fiftyDayAverage", "twoHundredDayAverage", "fiftyTwoWeekHigh", "fiftyTwoWeekLow",
                    "shortName", "quoteType", "marketCap", "earningsTimestamp", "earningsTimestampStart",
                    "earningsTimestampEnd", "isEarningsDateEstimate")})
            for side, df in (("C", ch.calls), ("P", ch.puts)):
                if df is None or df.empty:
                    continue
                d = df[["contractSymbol", "lastTradeDate", "strike", "lastPrice", "bid", "ask", "volume",
                        "openInterest", "impliedVolatility"]].copy()
                d["side"], d["expiry"], d["sym"] = side, e, sym
                out["chains"].append(d)
        out["meta"]["spot"] = spot if spot else float(h["Close"].iloc[-1])
    except Exception as ex:
        out["err"] = repr(ex)[:200]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-expiry", default="2026-12-18")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()

    syms = sorted({s for v in UNIVERSE.values() for s in v})
    group = {s: g for g, v in UNIVERSE.items() for s in v}
    hist, chains, meta, errs = [], [], {}, {}
    with ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(fetch_one, s, a.max_expiry): s for s in syms}
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            if r["err"]:
                errs[r["sym"]] = r["err"]
            if r["hist"] is not None:
                hist.append(r["hist"])
            chains.extend(r["chains"])
            if r["meta"]:
                meta[r["sym"]] = {**r["meta"], "group": group[r["sym"]]}
            print(f"[{i}/{len(syms)}] {r['sym']}: chains={len(r['chains'])} err={r['err']}", flush=True)

    H = pd.concat(hist).reset_index().rename(columns={"index": "Date"})
    C = pd.concat(chains, ignore_index=True)
    C["lastTradeDate"] = pd.to_datetime(C["lastTradeDate"], utc=True).dt.tz_localize(None)
    for col in ("volume", "openInterest"):
        C[col] = pd.to_numeric(C[col], errors="coerce").fillna(0).astype("int64")
    H.to_parquet("data/history.parquet", index=False)
    C.to_parquet("data/chains.parquet", index=False)
    json.dump({"fetched_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "max_expiry": a.max_expiry,
               "meta": meta, "errors": errs}, open("data/meta.json", "w"), indent=1, default=str)
    print(f"history rows={len(H)} syms={H.sym.nunique()} | chain rows={len(C)} syms={C.sym.nunique()} | errors={errs}")


if __name__ == "__main__":
    sys.exit(main())
