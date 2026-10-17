# Research log — 2026-10-03 night (autonomous)

Protocol for every family: design on dev only, pre-register (`results/v3/preregistered/`), evaluate validation ONCE, then holdout ONCE.
Costs are Hyperliquid base tier: taker 4.5 bps per side, plus the spread, plus funding.

| # | Family | Data | Result | Verdict |
|---|---|---|---|---|
| 1 | Cross-sectional factors on 60 perps (low-vol, 7-day reversal, volume shock, 1 h reversal) | hourly carry panel | Rank ICs are real (low-vol 0.05, t 6–7) but **mean** returns are right-skewed: high-vol coins have median −56 bps/day and mean +21. Every portfolio loses before costs. | REJECTED |
| 2 | Minute "oracle snapback" (fade HL mid vs oracle) | HL asset_ctxs, all perps | Apparent +35–65 bps gross was **spread**: the mid of an empty book. With an entry gate on the real spread, gross falls to ~11 bps. | REJECTED |
| 3 | HL→Binance convergence, taker | HL L2 0.5 s + Binance aggTrades 1 s | SOL: dislocations are already closed when a 2 s order arrives (gross ≈ −1 bps). | REJECTED |
| 4 | Resting-order dislocation capture (maker) | same | Adverse selection. SOL −11 bps/fill, HYPE −14 to −60 (HL leads HYPE), PUMP −15 to −108; win rate 7–30%. | REJECTED |
| 5 | Spot cash-and-carry (short HL perp, long Binance spot, hysteresis on trailing funding) | carry panel | Dev +21.6%/yr. **Val +7.1% (maker) / +5.4% (taker), both halves positive: PASSED.** **Holdout −2.8% / −11.6%: FAILED** (funding too low in 2026-04..10). | FAILED (regime-dependent) |
| 6 | Time-series momentum, long-only (20/60/120 d) | daily | Dev Sharpe 1.35 vs buy-and-hold 1.04. Val Sharpe −0.22: FAILED, but max drawdown 11% vs 44% for buy-and-hold. | FAILED (a risk reducer, not alpha) |
| 7 | Funding sign-flip → receive the new rate (prior project F05), thr 3e-5, hold 8 h | HL asset_ctxs, 193 coins | Dev +76 bps/trade (t 3.0). Val +52 (t 1.85), both halves positive, **but negative excluding the top-5 days**: 92% of trades are longs after market-wide crashes (2024-04-13, 2024-08-05, 2025-02-03…). Concentration gate FAILED. Holdout contaminated (the family was discovered there). | **REJECTED** after a clean OOS stress test: Binance 2020-01..2023-05 (pre-registered, never seen, LUNA/FTX included), n=5,594, mean +4 bps (t 0.5), median −3, ex-top-5-days −12, 2022 −1.7. A crash-rebound lottery with no edge. The forward paper (`desk.live.flip_paper`) stays on as a monitor only, expectation ~0. |
| 8 | Binance positioning (OI, retail/top-trader long-short, taker ratio), cross-sectional, availability-lagged | Binance metrics 5m, 67 coins | Best of 14 configurations: contrarian to retail account long bias (z-score vs own 30 d), weekly, inverse-vol. Dev Sharpe 1.30 (reproduced at 1.44 by the eval script). **Val Sharpe −1.45**, both halves negative. OI-7d has a strong IC (t 5.2) but its portfolio is ~0 (the same skew problem). | REJECTED |
| 9 | New HL perp listings (all perps incl. delisted, 197 listings) | HL ctx daily | Short at day+1: dev −15%/trade at 14 d (t −2.6; median +, but pumps of +200–430%). Long (pre-registered): val **−24%/trade** (t −3.2, median −34%). Direction flips with the alt/meme cycle; a BTC 60 d regime switch fails even in-sample (t 0.5 / −0.3). | REJECTED (pure regime) |

## Structural conclusions

- **Speed:** with ~2 s latency and 9 bps taker round trip, Hyperliquid intraday is efficient. HFT arbitrageurs close cross-venue gaps in under 2 s, and resting quotes get picked off.
- **Funding premia:** carry against Binance perps and against spot are real but **regime-dependent**, and compressed since mid-2025. Keep them in paper so they can be deployed when funding reopens.
- **Payoff shape:** what survives in this data is lumpy event behaviour (crash rebounds). Judge such strategies with tail-aware statistics (median, excluding top days), never only the mean.

## Data and infrastructure added

- `desk/data/hl_ctx.py`: minute store of every HL perp since 2023-05, delisted coins included (D:/quant_lab/hlctx, 2.4 GB).
- `desk/data/xvenue.py`: 1 s HL L2 + Binance grid (results/v3/xvenue_store). `harvest.lz4_frames` reads **all** frames; a coin-day was previously read as 1 hour.
- Google Drive duplicate folders were merged (`rclone dedupe`), and the harvest now pre-creates folders.
- The feature store moved to D: (junction). D: and C: are shared with the operator's games and other projects; check free space before large writes.

## Regime monitor (the carry sleeves wait for funding to come back)

`python -m desk.live.regime` (running with `--loop` every 6 h, history in `results/v3/regime_history.jsonl`, latest in `results/v3/regime_monitor.json`).
It shows, per coin, the trailing-504 h HL-vs-Binance funding spread (carry v2 enters at 0.3 bps/h) and the trailing HL funding (spot-carry entry 0.2 bps/h).

Snapshot 2026-10-03 03:5x UTC:
- carry v2: no coin open; the closest are AERO, NIL and GRASS at 0.24–0.25 bps/h.
- Median HL funding is back to ~0.17 bps/h (≈15%/yr), and 20+ coins meet the spot-carry entry. The funding regime is warming up again after the low 2026-04..09 window.

## Forward paper: spot_carry_v1 (started 2026-10-03 03:21 local, session 20261003_032123_7b84)

`python -m desk.live.spot_carry_paper` applies the pre-registered v1 rule live: short HL perp + long Binance spot, maker fills one hour later, $100k gross.
The first cycle targeted 12 coins.
It passed validation and failed the holdout, which fell in a low-funding regime. This test checks whether it pays now that the funding regime is warming.
Decision rule: after 30 days, compare realized net carry against the validation rate (+7%/yr). Deploy only if it is positive and the regime monitor stays open.
- Automatic review: every 6 h, `results/v3/forward_review.json` applies the rule fixed before any forward result. After ≥ 30 days the sleeve is a deploy candidate if its yearly return on $100k gross is > 0 and ≥ 3.5% (half the validation rate) and its max drawdown is < 3%. Otherwise it fails. The dashboard (:8900) shows the paper sessions and a funding-regime card.
- **Capital reality of spot carry (amended rule, 05:3x, before any forward result):** the short perp is on HL and its spot hedge on Binance, so the short needs ~100% margin. Over dev+val, the worst 24 h run-up is p99 +72% (max +93%) and the whole-hold run-up is p90 +187% (WIF +1,329%). Capital is therefore about 2× gross, and the validation +7.1%/yr on gross is only ~3.5%/yr on capital, below a stablecoin or T-bill yield. Deploy only if the return on capital is ≥ 5%/yr.
- The same constraint applies to carry v2 (HL perp vs Binance perp): each leg is margined on its own venue, so a large move needs margin on the losing side. Count capital as ~2× gross there too (or ≥ 1.5× with daily cross-venue rebalancing) before comparing yields.

## The one robust result: HL-native cash-and-carry with portfolio margin (BTC + HYPE)

Hyperliquid portfolio margin (beta) lets a spot balance collateralize a short perp in the same account. BTC (LTV 0.5) and HYPE (LTV 0.65) are eligible; the account needs > $10k. Spot and perp PnL offset each other, so capital is about 1.15× notional instead of 2× cross-venue. Source: https://hyperliquid.gitbook.io/hyperliquid-docs/trading/portfolio-margin

Backtest 2023-06..2026-10, **always on, no parameters**, net of ~10 bps entry/exit, on 1.15× capital: **every half-year is positive for both coins.**

| | Range of half-years | Current half |
|---|---|---|
| BTC | +2.4%/yr (2026H1) .. +26% | +7.0% |
| HYPE | +7.2% .. +102% | +8.4% |

This is a market-neutral yield, not alpha. It beats T-bills and stablecoin yields modestly, and idle USDC in the same account also earns interest.

Risks:
- Hyperliquid venue and contract risk.
- Liquidation path dependency (10-min TWAP, oracle timing).
- Beta caps.
- Funding can turn negative for weeks (BTC: 30% of hours in 2026H1).

Forward paper `python -m desk.live.hl_carry_paper` started 2026-10-03 (BTC + HYPE, $50k each). Verdict after 30 days in `forward_review.json`: deploy candidate if ≥ 5%/yr on capital.

### Deployment sizing (if the 30-day forward verdict is positive)

Liquidation stress under portfolio margin (LTV haircut on spot; maintenance margin assumed 3%, not stated in the docs):

| | Buffer 0% | Buffer 15% | Buffer 30% | Worst historical 30 d run-up | p99 30 d run-up |
|---|---|---|---|---|---|
| BTC (LTV 0.5) | +89% | **+117%** | +145% | +63% | +48% |
| HYPE (LTV 0.65) | +163% | **+203%** | +242% | +94% | +81% |

Plan:
- 1× short perp against 1× spot, plus a 15% USDC buffer (capital 1.15× notional).
- Re-hedge monthly, or whenever |price move since the last hedge| > 40%: sell the spot gain to USDC, or top it up.
- Start small (for example $10–20k, above the $10k eligibility floor), split BTC/HYPE 50/50.
- Use maker (post-only) entries.
- Close a coin if its trailing 30-day funding turns negative and stays below −5%/yr.
- Never exceed the beta caps.
