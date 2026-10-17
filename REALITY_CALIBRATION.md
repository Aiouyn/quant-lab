# Reality calibration — what real Hyperliquid execution taught the simulator

Sources:
- The operator's earlier project, `HYPERLIQUID_DATA_MODEL_1`: its live execution ledger (`output/exec_ledger.db`, 105 real IOC fills) and its live-vs-backtest reports (`output/S4B_LIVE_VS_BT_*.md`).
- Its warehouse, `HYPERLIQUID_TRACKER/synced/master_hyperliquid.db` (6.3 GB): BBO ticks every 2 s, cross-venue snapshots, funding.

## Measured on real fills (S4b meme fade, PUMP/FARTCOIN, ~$10 IOC taker orders)

| Quantity | Measured | Simulator before | Simulator now |
|---|---|---|---|
| Taker fee | 4.5 bps | 4.5 bps | 4.5 bps |
| Decision → fill latency | 1.2 s median, 2.5 s mean | 1 s | **2 s** (`CostModel.latency_s`) |
| Slippage vs decision mid | 1.3 bps median, 3.3 p90 | 0.5 bps + spread + depth | + **0.05 bps per bps of 60 s move** (`vol_slip_coef`) |
| Execution drag in fast moves | −11 bps per round trip on ~1% moves | not modelled | modelled by `vol_slip_coef` |
| IOC orders not filled | ~24% of attempts (no match at limit) | always filled | **gate `missed_entries_25%_profitable`** |

### Live vs research: −17.5 bps per trade over 51 live round trips (research and paper said +11.6)

| Component | Cost | What it means |
|---|---|---|
| Regime | −22 bps | The live window was not the research sample. |
| Selection | −31 bps | The events live actually took were worse than the backtest average. |
| Execution / timing | −11 bps | IOC orders plus catch-up: 21 of 22 entries came ~124 s after the signal bar. |

**Lesson:** a paper/research edge of about +10 bps on fast meme moves does not survive live execution.

## Hyperliquid spreads (BBO every 2 s, 2026-08-09 → 2026-10-02, median bps)

| BTC | SOL | HYPE | ETH | ZEC | XPL | FARTCOIN | PUMP |
|---|---|---|---|---|---|---|---|
| 0.13 | 0.13 | 0.13 | 0.41 | 0.65 | 1.29 | 2.14 | 2.76 |

The carry engine's per-leg slippage assumption (1 bp majors, 3 bps alts, plus impact) is conservative except for meme coins.

## Carry basis model validated

The model approximates the HL-vs-Binance price spread by the premium spread. It was compared with the real basis (HL BBO mid vs Binance mark, aligned to within 3 s, about 550 hourly points):

| | BTC | ETH | SOL |
|---|---|---|---|
| Correlation of basis level (real vs model) | 0.91 | 0.86 | 0.78 |
| Hourly change std, real vs model (bps) | 0.92 vs 1.64 | 1.28 vs 1.98 | 1.13 vs 1.62 |

The model tracks the basis **level**, which drives multi-day carry PnL. Its hourly noise is higher than reality, so it overstates basis risk.

Note: the warehouse table `cross_venue_snapshots.hl_mark` is stale (it updates every few minutes) and must not be used for basis without BBO alignment.

## The prior project's research verdicts (26 sessions; do not repeat)

**Killed after fees:**
- whale-direction copy trading
- exotic features (entropy, skew, correlation breaks, VPIN)
- pure ML direction
- slow risk premia
- market making on a book-spread proxy

**Survived research/paper but failed or is unproven live:**
- S4b meme panic fade (failed live, see above)
- EU→US session reversal S19 (research n = 463)
- funding-flip receive F05 (n = 361, EV +49 bps)
- whale-build fade F03 (n = 644, EV +44 bps)
