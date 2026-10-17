# Research agenda: where edges can still exist at our speed

## The diagnosis (measured, RESEARCH_LOG_2026-10-03.md)

With ~2 s latency and ~9 bps round-trip taker costs, intraday crypto is efficient for us:
- Cross-venue gaps close before our orders arrive.
- Resting quotes get picked off.
- Price-pattern factors are lottery-skewed.
- Funding premia exist, but only in some regimes.

So the lab should not out-race professionals. It should go where speed does not matter, and where we can **bear a risk others avoid** or **exploit a behavioural/structural bias** that persists because it is costly, slow or unattractive for large players to arbitrage.

## Where to look, and why

| # | Idea | Why it can persist | Desk | How to test it here |
|---|---|---|---|---|
| 1 | Cross-asset time-series momentum, vol-targeted, equal risk per asset class | Slow information diffusion, hedging demand; documented across 100+ years and asset classes | macro | `macro_panel`; trend 1–12 m; inverse-vol; per-class risk budget |
| 2 | Risk parity plus trend overlay | Leverage aversion makes low-risk assets earn more per unit of risk | macro | inverse-vol / covariance weights, gross ≤ 2, correlation-regime switch designed on dev only |
| 3 | Country and sector cross-sectional momentum, crash-protected | Under-reaction to macro news; crash risk is what keeps it paid | macro | 6–12 m skip-1 m ranks, vol scaling, market-state gate |
| 4 | Calendar flows: turn-of-month, pre-holiday, overnight vs intraday | Mechanical flows (payroll, rebalancing), inventory risk | macro | `event_study` on SPY/QQQ/IWM by calendar day; `adj_open` for overnight |
| 5 | Volatility term structure and credit-spread regimes | Variance risk premium is compensation for crash risk | macro | VIX/VIX3M, VVIX, MOVE, HYG/IEF trend → equity/bond exposure |
| 6 | Defensive equity (low vol, quality) vs high beta | Betting-against-beta: leverage constraints | macro | USMV/SPLV/QUAL vs SPY; sector low-vol rotation |
| 7 | Curve and real yields → duration ladder and gold | Term premium; real-rate sensitivity | macro | ^TNX−^IRX slope → TLT/IEF/IEI/SHY; real yields → GLD/TIP |
| 8 | Cross-asset lead-lag (credit → equities, dollar → EM, oil → CAD, copper → AUD) | Segmented investors, slow cross-market attention | macro | pre-specified economic pairs only, 1–5 d horizons |
| 9 | Favorite-longshot bias and calibration | Retail-heavy market; longshots overpriced in betting/option markets for decades | **pred** | price bucket × time to resolution vs realized frequency; trade the miscalibrated buckets net of the 1¢ spread |
| 10 | Overreaction to news jumps, drift into resolution | Thin books, attention-driven flows, capital locked until resolution | pred | `event_study` on >10¢ jumps by hours_to_end; near-certain contracts in the last days |
| 11 | Crypto implied-volatility premium as a timing signal | Variance risk premium in DVOL vs realized | perps | dvol vs 30 d realized vol → exposure scaling |
| 12 | Vol-managed, beta-neutral residual momentum in perps | Volatility management raises momentum Sharpe; residuals avoid the market crash leg | perps | weekly, liquidity-filtered, never naked-short the high-vol tail |
| 13 | US macro shocks propagating into crypto outside US hours | Attention and time-zone segmentation | perps | spx_ret_1d / vix / dxy (usable next day 01:00 UTC) → next-day perps |
| 14 | Funding harvested only when the trend agrees; HL-native cash-and-carry | Funding pays crowded leverage; capital-efficient only in portfolio margin | perps / carry | maker execution, per-coin caps |
| 15 | Combining weak signals (walk-forward ensembles) | Diversification across low-correlated weak edges | all | `walk_forward` inside `signal()`; causality test catches leaks |
| 16 | A drawdown-controlled directional crypto sleeve | Trend + vol targeting cut bear drawdowns (44% → 11% measured) | perps | judge against vol-matched buy-and-hold after funding |

## What makes a result trustworthy here

- **Design:** on dev only.
- **Evidence:** report the mean, the median and the mean **excluding the 5 best days**.
- **Promotion path:**
  1. sealed validation, deflated against every validation run;
  2. one-shot holdout;
  3. **forward holdout** on data nobody has seen, rescored every 6 h, with a verdict at 60 days.
- **Live trading:** only after a forward confirmation, through a venue connector the operator decides to build:
  - Hyperliquid for perps;
  - a broker such as IBKR for ETFs/FX;
  - Polymarket on-chain for prediction markets.
