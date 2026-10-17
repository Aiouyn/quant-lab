"""Control-room dashboard: python -m desk serve --port 8900"""
from __future__ import annotations

import json
import time
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs

HTML = Path(__file__).with_name('index.html')


def state(settings):
    from ..company.ledger import Ledger
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
        strats.append({**s, 'dev_sharpe': dev.get('sharpe'), 'dev_trades': dev.get('trades'),
                       'gates': full.get('gate_reasons'),
                       'val_sharpe': ((full.get('val_report') or {}).get('summary') or {}).get('sharpe'),
                       'holdout_sharpe': ((full.get('holdout_report') or {}).get('summary') or {}).get('sharpe')})
    st['strategy_list'] = strats
    st['events'] = L.q("SELECT created, agent, kind, substr(payload,1,400) AS payload FROM events WHERE kind NOT IN ('TOOL') "
                       "ORDER BY id DESC LIMIT 60")
    st['lesson_list'] = L.q("SELECT id, author, claim, confidence, created FROM lessons WHERE status='active' ORDER BY created DESC LIMIT 25")
    st['tokens_by_hour'] = L.q("SELECT CAST((created - (SELECT MIN(created) FROM llm_requests))/600 AS INT) AS bucket, "
                               "SUM(prompt_tokens+completion_tokens) AS tokens, COUNT(*) AS n FROM llm_requests GROUP BY bucket ORDER BY bucket")
    st['errors'] = L.q("SELECT status, COUNT(*) n FROM llm_requests WHERE status<>'ok' GROUP BY status")
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


def serve(settings, port=8900):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype):
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
                if u.path == '/':
                    return self._send(200, HTML.read_text(encoding='utf-8'), 'text/html; charset=utf-8')
                if u.path == '/api/state':
                    return self._send(200, json.dumps(state(settings), default=str), 'application/json')
                if u.path == '/api/item':
                    from ..company.ledger import Ledger
                    L = Ledger(settings.ledger)
                    i = parse_qs(u.query).get('id', [''])[0]
                    obj = L.strategy(i) if i.startswith('S-') else (L.q('SELECT * FROM factors WHERE id=?', (i,)) or [None])[0] \
                        if i.startswith('F-') else L.board_item(i) or (L.q('SELECT * FROM agents WHERE name=?', (i,)) or [None])[0]
                    return self._send(200, json.dumps(obj, default=str), 'application/json')
                self._send(404, 'not found', 'text/plain')
            except Exception as e:
                self._send(500, json.dumps({'error': repr(e)}), 'application/json')

    httpd = ThreadingHTTPServer(('127.0.0.1', port), H)
    print(f'Desk OS control room: http://127.0.0.1:{port}', flush=True)
    httpd.serve_forever()
