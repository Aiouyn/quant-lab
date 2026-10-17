# Data parity: every research dataset must be reproducible in real time

**Rule.** A strategy may only be promoted if every dataset it reads meets three conditions:
- it has a live source with the same fields;
- research joins it by **availability time**, not label time, through `desk/data/availability.py`;
- live-vs-archive agreement is verified.

**Tools:**
- `desk.live.parity_recorder` records the live sources in the archive schemas, stamping each row with its receive time. Output goes to `results/v3/parity/` and to Drive under `QUANT_DATA/live/`.
- `python -m desk.data.parity <date>` compares a recorded day with the archive once the archive is published (D+1). Reports go to `results/v3/parity/reports/`.

## Verified so far (2026-10-02)

| Dataset | Live source | Result |
|---|---|---|
| **HL l2Book** (archive ~5.4 s, 20 levels) | WS `l2Book` | 2026-10-01, BTC, 15,002 archive snapshots vs the live 1 Hz recorder: best bid/ask prices identical (72% exact at the next live second; the rest is the book moving in between), sizes identical ×100 (the old recorder's contract scaling). Receipt ~0.5–1.0 s after the book timestamp. **OK.** |
| **Binance klines 1m** | REST klines | 2026-10-01 SOLUSDT, 1,440 bars: close, volume and taker-buy volume **100% identical**. **OK.** |
| **Binance metrics** (OI, long/short, taker ratio) | REST `/futures/data/*` 5m | 2026-10-01, 3 symbols × 288 rows. OI: **100% identical** once realigned. Ratios: differences around 1e-4 (the API rounds to 4 decimals). **Trap found:** the archive's `create_time` = API timestamp **− 5 min** for OI and long/short ratios, and = API timestamp for the taker ratio. Read naively, the archive leaks about 5 minutes. Fixed in the recorder (long format with archive-aligned `create_time`) and in `availability.METRIC_LAG_S`. |
| **Carry panel** (Binance 1h klines/premium) | REST premiumIndex / klines | **Trap found:** coin files keyed Binance bars by **open** time, a 1 h look-ahead against the HL columns. Fixed in `carry_data.align_bar_close` (applied by `load_panel`). Effect on carry v2: maker (its real execution) validation Sharpe 1.85 → **2.82**, from less basis noise. Its *taker* robustness variant was inflated: 1.14 → 0.13, and validation basis PnL went from +$21/day to −$5/day. |
| **HL asset_ctxs** | REST `metaAndAssetCtxs` | 2026-10-01 BTC, 80 overlapping minutes: an archive row for minute T equals the live mid at T+3 s (88% exact; 0% drift). The recorder now polls at T+3 s; research lag is 5 s. **OK.** |
| **Live carry paper inputs** | HL + Binance premiumIndex | **Two bugs fixed.** (1) `bn_rate_last` live was Binance's *predicted* rate, while history uses the last *paid* rate. (2) `bn_interval_h` live was hard-coded to 8, but 42 of 79 coins settle every 4 h, so their live Binance rate was halved in the spread. The session was restarted (`20261002_192826_0438`). It is flat, as is the backtest: v2 has held no positions for 90 days except one, a month ago. |

## Availability lags applied in research (`desk/data/availability.py`)

| Dataset | Label | Available after the label (live-measured, rounded up) |
|---|---|---|
| HL l2Book | book `time` | 2 s |
| HL asset_ctxs | minute T | 5 s: an archive row is a snapshot at ~T+3 s (BTC mid_px 88% exact against the live recorder at +3 s, 2026-10-01) |
| Binance klines 1m | open_time | 63 s |
| Binance aggTrades 1 s aggregate | row k = trades in [k−1 s, k) | 1 s |
| Binance bookDepth | timestamp | 2 s |
| Binance metrics | archive create_time | 600 s (OI, long/short ratios), 660 s (taker ratio). Probe saw 304–453 s and 390–512 s; the recorder keeps measuring `first_seen_lag_s` |

## Coverage caveats

- **HL asset_ctxs archive is truncated from 2026-09 onward:** each day holds only its first hours. The live recording is the source of truth for recent days.
- **Binance bookDepth ±1..5%:** the live REST depth of 1,000 levels does not reach ±5% on deep books (BTC ≈ ±0.12%, 1000PEPE ≈ ±2.8%). Rows carry `complete`; research may use only the bands that are complete live.
- **HL node_fills liquidation flags:** not in the public live stream. Use only live-inferable fields.
- **Binance futures WS:** the market streams moved to `wss://fstream.binance.com/market/stream`; the old `/stream` path silently hangs.

## Still to verify (automatically, once the archive publishes the recorded days)

- HL asset_ctxs: label convention and field equality (from 2026-10-03).
- HL l2Book: exact book-version join on 20 coins.
- Binance aggTrades: WS completeness against the archive.
- Binance bookDepth: snapshot equality.

Run `python -m desk.data.parity 2026-10-03` on 2026-10-04 or later.
