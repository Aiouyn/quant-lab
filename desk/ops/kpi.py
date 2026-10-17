"""Firm performance report: what does each dollar of LLM budget produce?   python -m desk.ops.kpi [hours]

Cost uses the MiMo token-plan credit rates (flash: cache hit 2, miss 100, output 200 credits/token; pro: 2.5/300/600)
and the plan price per credit (Standard: $16 / 11e9 credits, overridable with KPI_USD_PER_CREDIT).
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter

RATES = {'mimo-v2.6-flash': (2, 100, 200), 'mimo-v2.6-pro': (2.5, 300, 600)}
USD_PER_CREDIT = float(os.environ.get('KPI_USD_PER_CREDIT', 16 / 11e9))


def report(settings, hours=1.0):
    from ..company.ledger import Ledger
    L = Ledger(settings.ledger)
    t0 = time.time() - hours * 3600
    req = L.q("SELECT model, COUNT(*) n, SUM(prompt_tokens) p, SUM(COALESCE(cached_tokens,0)) c, SUM(completion_tokens) o, "
              "AVG(seconds) s, SUM(status<>'ok') err FROM llm_requests WHERE created >= ? GROUP BY model", (t0,))
    credits = 0.0
    tok = {'requests': 0, 'prompt': 0, 'cached': 0, 'output': 0, 'errors': 0}
    for r in req:
        hit, miss, out = RATES.get(r['model'] or '', RATES['mimo-v2.6-flash'])
        p, c, o = r['p'] or 0, r['c'] or 0, r['o'] or 0
        credits += c * hit + (p - c) * miss + o * out
        tok['requests'] += r['n']; tok['prompt'] += p; tok['cached'] += c; tok['output'] += o; tok['errors'] += r['err'] or 0
    usd = credits * USD_PER_CREDIT
    ev = L.q('SELECT agent, kind, payload FROM events WHERE created >= ?', (t0,))
    kinds = Counter(e['kind'] for e in ev)
    tools = Counter()
    tool_err = Counter()
    py = Counter()
    py_err = Counter()
    py_wait = []
    for e in ev:
        try:
            pl = json.loads(e['payload'] or '{}')
        except Exception:
            pl = {}
        if e['kind'] == 'TOOL':
            tools[pl.get('tool')] += 1
            if pl.get('error'):
                tool_err[pl.get('tool')] += 1
        if e['kind'] == 'RUN_PYTHON':
            py[pl.get('status', '?') == 'ok'] += 1
            py_wait.append(float(pl.get('queued_s') or 0))
            if pl.get('status') != 'ok' and pl.get('error'):
                py_err[pl['error'].split(':')[0][:60]] += 1
    out = {
        'window_h': hours,
        'llm': {**tok, 'cache_hit_rate': round(tok['cached'] / max(tok['prompt'], 1), 3),
                'credits': round(credits / 1e6, 1), 'usd': round(usd, 3), 'usd_per_hour': round(usd / hours, 3),
                'plan_hours_left_at_this_rate': None},
        'per_agent': [{'agent': r['agent'], 'requests': r['n'], 'usd': round(((r['c'] or 0) * 2 + ((r['p'] or 0) - (r['c'] or 0)) * 100
                                                                            + (r['o'] or 0) * 200) * USD_PER_CREDIT, 4),
                       'tool_calls': sum(1 for e in ev if e['agent'] == r['agent'] and e['kind'] == 'TOOL'),
                       'run_python_failed': sum(1 for e in ev if e['agent'] == r['agent'] and e['kind'] == 'RUN_PYTHON'
                                                and '"status": "ok"' not in (e['payload'] or ''))}
                      for r in L.q("SELECT agent, COUNT(*) n, SUM(prompt_tokens) p, SUM(COALESCE(cached_tokens,0)) c, "
                                   "SUM(completion_tokens) o FROM llm_requests WHERE created >= ? AND status='ok' GROUP BY agent "
                                   "ORDER BY p DESC", (t0,))],
        'sessions_ended': kinds.get('SESSION_END', 0),
        'tool_calls': dict(tools.most_common()), 'tool_errors': dict(tool_err),
        'run_python': {'ok': py[True], 'failed': py[False], 'failure_rate': round(py[False] / max(py[True] + py[False], 1), 3),
                       'top_errors': dict(py_err.most_common(6)),
                       # time runs spend queued for a sandbox slot / free RAM: a stalled firm shows up here first
                       'mean_queued_s': round(sum(py_wait) / max(len(py_wait), 1), 1), 'max_queued_s': round(max(py_wait or [0]), 1)},
        'artifacts': {
            'factors': L.q('SELECT resolution desk, status, COUNT(*) n FROM factors WHERE created >= ? GROUP BY 1,2', (t0,)),
            'strategies': L.q('SELECT status, COUNT(*) n FROM strategies WHERE created >= ? GROUP BY 1', (t0,)),
            'lessons': L.q('SELECT COUNT(*) n FROM lessons WHERE created >= ?', (t0,))[0]['n'],
            'board_posts': L.q("SELECT COUNT(*) n FROM board WHERE created >= ? AND author <> 'lab-director'", (t0,))[0]['n'],
            'dev_backtests': kinds.get('DEV_BACKTEST', 0), 'trials_logged': L.q('SELECT COUNT(*) n FROM trials WHERE created >= ?', (t0,))[0]['n'],
        },
        'pipeline_events': {k: v for k, v in kinds.items() if k in ('DEV_PASS', 'DEV_REJECT', 'EVAL_RETRY_MEMORY', 'EVAL_INFRA_ERROR', 'VAL_PASS', 'VAL_REJECT', 'APPROVED',
                                                                     'HOLDOUT_REJECT', 'FORWARD_SCORED', 'SUBMITTED')},
        'errors': {k: v for k, v in kinds.items() if 'ERROR' in k or 'QUOTA' in k},
    }
    a = out['artifacts']
    n_art = sum(x['n'] for x in a['factors']) + sum(x['n'] for x in a['strategies']) + a['lessons'] + a['dev_backtests']
    out['yield'] = {'artifacts': n_art, 'usd_per_artifact': round(usd / n_art, 3) if n_art else None,
                    'artifacts_per_million_tokens': round(n_art / max((tok['prompt'] + tok['output']) / 1e6, 1e-9), 3)}
    return out


if __name__ == '__main__':
    from ..config import SETTINGS
    print(json.dumps(report(SETTINGS, float(sys.argv[1]) if len(sys.argv) > 1 else 1.0), indent=1, default=str))
