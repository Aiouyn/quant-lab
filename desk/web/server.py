"""Control-room dashboard: python -m desk serve --port 8900 (or 0.0.0.0:3000 via app.py)"""
from __future__ import annotations

import json
import time
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs

HTML = Path(__file__).with_name('index.html')


def state(settings):
    from ..company.ledger import Ledger
    from ..company.rsi import rsi_snapshot
    from ..live.monitor import paper_status
    L = Ledger(settings.ledger)
    st = L.status()
    st['run'] = L.get('run')
    st['now'] = time.time()
    st['memo'] = L.latest_memo('desk')
    st['digest'] = L.latest_memo('digest')
    st['board'] = L.board_list('active', limit=40)
    facs = L.factors(25)
    st['factors'] = [{'id': f['id'], 'name': f['name'], 'author': f['author'], 'pod': f['pod'], 'res': f['resolution'],
                      'score': round(f['score'] or 0, 2), 'best_h': (f['report'] or {}).get('best_horizon_s'),
                      't': (f['report'] or {}).get('best_t'), 'cost_cov': (f['report'] or {}).get('best_cost_coverage'),
                      'novel': ((f['report'] or {}).get('novelty') or {}).get('max_abs_corr')} for f in facs]
    strats = []
    for s in L.strategies(limit=60):
        full = L.strategy(s['id'])
        dev = (full.get('dev_report') or {}).get('summary') or {}
        frep = full.get('forward_report') or {}
        strats.append({**s, 'dev_sharpe': dev.get('sharpe'), 'dev_trades': dev.get('trades'),
                       'gates': full.get('gate_reasons'),
                       'val_sharpe': ((full.get('val_report') or {}).get('summary') or {}).get('sharpe'),
                       'holdout_sharpe': ((full.get('holdout_report') or {}).get('summary') or {}).get('sharpe'),
                       'forward_sharpe': ((frep.get('summary') or {}).get('sharpe')),
                       'forward_verdict': frep.get('verdict')})
    st['strategy_list'] = strats
    st['events'] = L.q("SELECT created, agent, kind, substr(payload,1,400) AS payload FROM events WHERE kind NOT IN ('TOOL') "
                       "ORDER BY id DESC LIMIT 60")
    st['lesson_list'] = L.q("SELECT id, author, claim, confidence, created FROM lessons WHERE status='active' ORDER BY created DESC LIMIT 25")
    st['tokens_by_hour'] = L.q("SELECT CAST((created - (SELECT MIN(created) FROM llm_requests))/600 AS INT) AS bucket, "
                               "SUM(prompt_tokens+completion_tokens) AS tokens, COUNT(*) AS n FROM llm_requests GROUP BY bucket ORDER BY bucket")
    st['errors'] = L.q("SELECT status, COUNT(*) n FROM llm_requests WHERE status<>'ok' GROUP BY status")
    st['rsi'] = rsi_snapshot(L, settings=settings)
    try:
        st['paper'] = paper_status(settings)
        rm = settings.home / 'regime_monitor.json'
        if rm.exists():
            r = json.loads(rm.read_text())
            st['regime'] = {k: r.get(k) for k in ('updated', 'carry_v2_open', 'spot_carry_open', 'median_abs_spread_bps_h',
                                                  'median_hl_funding_bps_h')} | {'closest': r.get('rows', [])[:5]}
    except Exception as e:
        st['paper'] = {'error': repr(e)}
    return st


def lookup_item(settings, item_id: str):
    from ..company.ledger import Ledger
    from ..company.rsi import primitive_lift_leaderboard, CONTRIB_DIR
    L = Ledger(settings.ledger)
    i = (item_id or '').strip()
    if i.startswith('S-'):
        return L.strategy(i)
    if i.startswith('P-'):
        r = L.q('SELECT * FROM portfolios WHERE id=?', (i,))
        if not r:
            return None
        p = r[0]
        for k in ('members', 'weights', 'report'):
            try:
                p[k] = json.loads(p[k]) if p.get(k) else p.get(k)
            except (TypeError, ValueError):
                pass
        return p
    if i.startswith('F-'):
        r = L.q('SELECT * FROM factors WHERE id=?', (i,))
        if r and r[0].get('report'):
            try:
                r[0]['report'] = json.loads(r[0]['report'])
            except (TypeError, ValueError):
                pass
        return r[0] if r else None
    for prefix, table in (('H-', 'dynamic_hints'), ('L-', 'lessons')):
        if i.startswith(prefix):
            r = L.q(f'SELECT * FROM {table} WHERE id=?', (i,))
            return r[0] if r else None
    if i.startswith('prim:'):
        pname = i.split(':', 1)[1]
        meta = next((p for p in primitive_lift_leaderboard(L) if p['primitive'] == pname), {'primitive': pname})
        src = CONTRIB_DIR / f'{pname}.py'
        if src.exists():
            meta['source_code'] = src.read_text(encoding='utf-8')
        return meta
    return L.board_item(i) or (L.q('SELECT * FROM agents WHERE name=?', (i,)) or [None])[0]


def serve(settings, port=8900, host='127.0.0.1'):
    class ReusableHTTPServer(ThreadingHTTPServer):
        allow_reuse_address = True

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype='application/json'):
            b = body.encode('utf-8') if isinstance(body, str) else body
            self.send_response(code)
            self.send_header('Content-Type', ctype)
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Length', str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            u = urlparse(self.path)
            try:
                if u.path in ('/', '/index.html'):
                    return self._send(200, HTML.read_text(encoding='utf-8'), 'text/html; charset=utf-8')
                if u.path == '/api/state':
                    return self._send(200, json.dumps(state(settings), default=str))
                if u.path == '/api/rsi':
                    from ..company.ledger import Ledger
                    from ..company.rsi import rsi_snapshot
                    return self._send(200, json.dumps(rsi_snapshot(Ledger(settings.ledger), settings=settings), default=str))
                if u.path == '/api/item':
                    i = parse_qs(u.query).get('id', [''])[0]
                    return self._send(200, json.dumps(lookup_item(settings, i), default=str))
                self._send(404, 'not found', 'text/plain')
            except Exception as e:
                self._send(500, json.dumps({'error': repr(e)}))

        def do_POST(self):
            # The dashboard is read-only: the retired RSI action endpoints ran synthetic loops that wrote into the
            # ledger. Operator actions go through the CLI.
            self._send(405, json.dumps({'error': 'the control room is read-only; use python -m desk ... instead'}))

    httpd = ReusableHTTPServer((host, int(port)), H)
    print(f'Desk OS control room: http://{host}:{port}', flush=True)
    httpd.serve_forever()
