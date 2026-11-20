# claude-trade

Hypothetical research only: which listed options could return **40x within one month**, and how likely that is.
Not investment advice. All data comes from Yahoo Finance via `yfinance` (15-minute-delayed quotes, daily bars).

## Pipeline

| step | script | what it does |
|---|---|---|
| 1 | `fetch_data.py` | 10y daily history + every option chain expiring within ~1 month for ~155 tickers → `data/` |
| 2 | `scan_40x.py` | for each liquid OTM contract (and vertical spread): move needed for 40x, market-implied probability, historical probability (raw and vol-scaled) of reaching 40x at expiry or at any daily close |
| 3 | `regime_study.py` | longest-available history; keeps only dates whose drawdown / 21-day return look like today's and replays the next sessions through today's chain |
| 4 | `shortlist.py` | merges 2 + 3; `robust` = the lowest of all historical estimates |
| 5 | `option_history.py` | price history of individual contracts of one expiry → how many actually multiplied |
| 6 | `pick_detail.py` | contract price history + spot level needed for 40x by date, for chosen contracts |

```bash
pip install -r requirements.txt
python fetch_data.py && python scan_40x.py && python regime_study.py $(python -c "import json;print(' '.join(json.load(open('data/meta.json'))['meta']))")
python shortlist.py
python option_history.py --expiry 2026-11-20 --start 2026-10-20 SPY QQQ NVDA TSLA
python pick_detail.py TSLA261218C00480000 NVDA261218C00200000
```

Yahoo rate-limits (HTTP 429) after a few thousand requests; the scripts back off, but keep concurrency low.

## Reading the numbers

- `p_rn` – market-implied (risk-neutral) probability of 40x at expiry. Under fair pricing **no** strategy can exceed 1/40 = 2.5%.
- `p_raw_*`, `p_fhs_*`, `p_reg_*`, `p_regv_*` – frequencies in history (own 10y, vol-scaled, regime-conditional, regime + vol-scaled).
  They are biased upward for calls: survivorship (today's liquid names are past winners), positive drift, overlapping windows,
  sticky-strike IV (overstates calls in rallies, understates puts in crashes).
- `robust` – minimum of the available historical estimates.

`results/2026-11-19/` holds the snapshot produced from the 2026-11-19 close.
