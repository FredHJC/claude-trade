"""Merge scan_40x + regime_study outputs into one robust shortlist.

robust = the LOWEST of all historical touch-probability estimates available for a contract
(own 10y raw, own 10y vol-scaled, regime raw, regime vol-scaled). A contract only ranks high
if every way of reading history agrees. p_rn (market-implied) is shown as the anchor.
"""
import sys
import numpy as np
import pandas as pd

M = sys.argv[1] if len(sys.argv) > 1 else "40"
N = pd.read_csv(f"data/scan_naked_{M}.csv")
R = pd.read_csv(f"data/regime_picks_{M}.csv")[["contract", "n_reg", "episodes", "p_reg_touch", "p_regv_touch"]]
D = N.merge(R, on="contract", how="left")
est = ["p_raw_touch", "p_fhs_touch", "p_reg_touch", "p_regv_touch"]
D["robust"] = D[est].min(axis=1, skipna=True)
D["hi"] = D[est].max(axis=1, skipna=True)
ok = (D.oi >= 1000) & (D.spread <= 0.25) & (D.n_win >= 1500) & (D.h >= 5) & (D.p_rn >= 0.008)
ok &= D.episodes.isna() | (D.episodes >= 12)
S = D[ok].sort_values("robust", ascending=False)
S.to_csv(f"data/shortlist_{M}.csv", index=False)

pd.set_option("display.width", 280); pd.set_option("display.max_rows", 200)
P2, P1 = "{:.2%}".format, "{:.1%}".format
F = {**{c: P2 for c in ["p_rn", "robust", "hi"] + est}, **{c: P1 for c in ["otm", "spread", "iv", "need_exp"]}}
cols = ["sym", "side", "expiry", "strike", "spot", "otm", "bid", "ask", "spread", "iv", "oi", "need_exp", "p_rn"] + est + ["robust", "hi", "episodes", "earn_in"]
best = S.groupby(["sym", "side"]).head(1)
print("== best contract per (symbol, side), ranked by robust ==")
print(best.head(25)[cols].to_string(index=False, formatters=F))
print("\n== calls vs puts: median / 90th pct of robust across liquid contracts ==")
print(D[ok].groupby("side").robust.describe(percentiles=[.5, .9, .99]).round(4).to_string())
print("\n== best PUT per symbol ==")
print(best[best.side == "P"].head(10)[cols].to_string(index=False, formatters=F))
