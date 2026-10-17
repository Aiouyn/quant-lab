"""Forward review of the live paper sleeves against their pre-registered expectations (written every 6 h by the
regime loop to results/v3/forward_review.json).  Decision rule fixed on 2026-10-03, BEFORE any forward result:
  spot_carry_v1: after >= 30 days, DEPLOY-CANDIDATE if annualized net return on $100k gross > 0 and >= half the
  validation rate (+7.1%/yr -> >= 3.5%/yr) and max drawdown < 3% of gross; otherwise NOT-YET / FAIL.
AMENDED 2026-10-03 05:3x (before any forward result except entry fees, disclosed): the short perp sits on HL while
its spot hedge sits on Binance, so the short needs ~100% margin (dev+val holds: worst 24 h run-up p99 72%, max 93%,
whole-hold run-up p90 +187%) -> capital = 2 x gross. The verdict now also requires return on TOTAL capital
(apr_on_gross / 2) >= 5%/yr, i.e. beating a stablecoin / T-bill yield; otherwise the sleeve is not worth deploying.
hl_carry (HL-native portfolio-margin cash-and-carry, BTC+HYPE, added 2026-10-03): same 30-day horizon, deploy
candidate if return on capital (1.15 x $100k) >= 5%/yr and max drawdown < 3%.
"""
from __future__ import annotations

import json
import time

VAL_APR = 0.071
GROSS = 100_000.0


def review(settings):
    out = {'updated': time.time(), 'sleeves': []}
    for d in sorted((settings.home / 'paper').glob('*/session.json')):
        s = json.loads(d.read_text())
        if s.get('kind') not in ('spot_carry', 'hl_carry'):
            continue
        capital = 100_000.0 * (1.15 if s['kind'] == 'hl_carry' else 2.0)       # portfolio margin vs cross-venue
        eq = [json.loads(l) for l in (d.parent / 'equity.jsonl').read_text().splitlines() if l.strip()] \
            if (d.parent / 'equity.jsonl').exists() else []
        if not eq:
            continue
        days = (eq[-1]['ts'] - s['started']) / 86400
        e = [x['equity'] for x in eq]
        peak, dd = -1e18, 0.0
        for v in e:
            peak = max(peak, v); dd = max(dd, peak - v)
        apr = e[-1] / GROSS / max(days, 1e-9) * 365
        apr_cap = e[-1] / capital / max(days, 1e-9) * 365
        verdict = 'RUNNING (<30 d)' if days < 30 else \
            ('DEPLOY-CANDIDATE' if apr_cap >= 0.05 and dd / GROSS < 0.03 and (s['kind'] == 'hl_carry' or apr >= VAL_APR / 2) else 'FAIL')
        out['sleeves'].append({'session': s['id'], 'target': s['target'], 'days': round(days, 2), 'equity_usd': round(e[-1], 2),
                               'funding_usd': round(eq[-1]['funding'], 2), 'fees_usd': round(eq[-1]['fees'], 2),
                               'basis_usd': round(e[-1] - eq[-1]['funding'] + eq[-1]['fees'], 2),
                               'apr_on_gross': round(apr, 4), 'apr_on_capital': round(apr_cap, 4), 'max_dd_usd': round(dd, 2), 'validation_apr': VAL_APR,
                               'verdict': verdict})
    (settings.home / 'forward_review.json').write_text(json.dumps(out, indent=1))
    return out


if __name__ == '__main__':
    from ..config import SETTINGS
    print(json.dumps(review(SETTINGS), indent=1))
