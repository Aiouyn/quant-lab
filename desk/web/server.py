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
    from ..company.rsi import ensure_seeded, rsi_snapshot
    from ..live.monitor import paper_status
    L = ensure_seeded(settings, Ledger(settings.ledger))
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
    from ..company.rsi import primitive_lift_leaderboard, compute_treasury_state, CONTRIB_DIR
    L = Ledger(settings.ledger)
    i = (item_id or '').strip()
    if i.startswith('S-'):
        obj = L.strategy(i)
        if obj:
            lin = L.q('SELECT * FROM lineages WHERE child_id=?', (i,))
            if lin:
                obj['lineage'] = lin[0]
        return obj
    if i.startswith('P-'):
        r = L.q('SELECT * FROM portfolios WHERE id=?', (i,))
        if r:
            p = r[0]
            for k in ('members', 'weights', 'report'):
                if p.get(k):
                    try:
                        p[k] = json.loads(p[k])
                    except Exception:
                        pass
            return p
        return None
    if i == 'treasury':
        return compute_treasury_state(L, settings=settings)
    if i == 'harness_audit':
        from ..data.live_flux import audit_harness_and_provenance
        return L.get('harness_audit') or audit_harness_and_provenance(settings, ledger=L)
    if i == 'live_flux':
        from ..data.live_flux import sync_live_flux
        return L.get('live_flux_state') or sync_live_flux(settings, ledger=L)
    if i == 'dry_run':
        from ..data.live_flux import run_dry_run_readiness
        return L.get('dry_run_readiness') or run_dry_run_readiness(settings, ledger=L)
    if i == 'execution_attribution':
        from ..data.live_flux import ingest_live_forward_bars_and_attribute
        return L.get('execution_attribution') or ingest_live_forward_bars_and_attribute(settings, ledger=L)
    if i == 'counterfactual_ablation':
        from ..company.rsi import run_counterfactual_ablation_and_replace
        return L.get('counterfactual_ablation') or run_counterfactual_ablation_and_replace(L, settings=settings, use_llm=False)
    if i == 'adversarial_redteam':
        from ..company.rsi import run_adversarial_redteam_and_cross_pollinate
        return L.get('adversarial_redteam') or run_adversarial_redteam_and_cross_pollinate(L, settings=settings, use_llm=False)
    if i == 'readiness_certification':
        from ..company.rsi import run_readiness_and_falsification_audit
        return L.get('readiness_certification') or run_readiness_and_falsification_audit(L, settings=settings, use_llm=False)
    if i == 'open_ended_rsi':
        from ..company.rsi import run_open_ended_frontier_rsi
        return L.get('open_ended_rsi') or run_open_ended_frontier_rsi(L, settings=settings, use_llm=False)
    if i.startswith('R-'):
        return L.reasoning_episode(i)
    if i.startswith('F-'):
        r = L.q('SELECT * FROM factors WHERE id=?', (i,))
        if r and r[0].get('report'):
            try:
                r[0]['report'] = json.loads(r[0]['report'])
            except Exception:
                pass
        return r[0] if r else None
    if i.startswith('H-'):
        r = L.q('SELECT * FROM dynamic_hints WHERE id=?', (i,))
        return r[0] if r else None
    if i.startswith('C-'):
        r = L.q('SELECT * FROM charter_versions WHERE id=?', (i,))
        return r[0] if r else None
    if i.startswith('L-'):
        r = L.q('SELECT * FROM lessons WHERE id=?', (i,))
        return r[0] if r else None
    if i.startswith('RSI-'):
        r = L.q('SELECT * FROM rsi_runs WHERE id=?', (i,))
        return r[0] if r else None
    if i.startswith('prim:'):
        pname = i.split(':', 1)[1]
        meta = next((p for p in primitive_lift_leaderboard(L) if p['primitive'] == pname), {'primitive': pname})
        src_path = CONTRIB_DIR / f'{pname}.py'
        if src_path.exists():
            meta['source_code'] = src_path.read_text(encoding='utf-8')
        return meta
    return L.board_item(i) or (L.q('SELECT * FROM agents WHERE name=?', (i,)) or [None])[0]


def serve(settings, port=3000, host='0.0.0.0'):
    import threading

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
            from ..company.ledger import Ledger
            from ..company.rsi import ensure_seeded, rsi_snapshot
            u = urlparse(self.path)
            try:
                if u.path in ('/', '/index.html'):
                    return self._send(200, HTML.read_text(encoding='utf-8'), 'text/html; charset=utf-8')
                if u.path == '/api/state':
                    return self._send(200, json.dumps(state(settings), default=str))
                if u.path == '/api/rsi':
                    L = ensure_seeded(settings, Ledger(settings.ledger))
                    return self._send(200, json.dumps(rsi_snapshot(L, settings=settings), default=str))
                if u.path == '/api/item':
                    i = parse_qs(u.query).get('id', [''])[0]
                    return self._send(200, json.dumps(lookup_item(settings, i), default=str))
                self._send(404, 'not found', 'text/plain')
            except Exception as e:
                self._send(500, json.dumps({'error': repr(e)}))

        def do_POST(self):
            from ..company.ledger import Ledger
            from ..company.rsi import (
                ensure_seeded, rsi_snapshot, compile_error_hints, evolve_population,
                evolve_primitives, synthesize_rsi_tool_and_operator,
                run_counterfactual_ablation_and_replace, run_adversarial_redteam_and_cross_pollinate,
                run_readiness_and_falsification_audit, run_open_ended_frontier_rsi,
                run_rsi_campaign, propose_charter_patch, evaluate_rsi_drill, validate_and_promote_population,
                start_rsi_daemon, stop_rsi_daemon, run_tournament_funnel, run_forward_survival_audit,
            )
            u = urlparse(self.path)
            try:
                length = int(self.headers.get('Content-Length') or 0)
                raw = self.rfile.read(length).decode('utf-8') if length > 0 else '{}'
                payload = json.loads(raw or '{}')
                if u.path == '/api/rsi/action':
                    L = ensure_seeded(settings, Ledger(settings.ledger))
                    act = payload.get('action', 'evolve')
                    desk = payload.get('desk') or None
                    use_llm = bool(payload.get('use_llm', True))
                    if act == 'compile_hints':
                        res = compile_error_hints(L, author='operator-ui')
                    elif act == 'evolve_primitives':
                        res = evolve_primitives(L, settings=settings, use_llm=use_llm, author='toolsmith-ui')
                    elif act == 'synthesize_rsi_tool':
                        res = synthesize_rsi_tool_and_operator(L, settings=settings, use_llm=use_llm, author='toolsmith-ui')
                    elif act == 'ablation_replace':
                        res = run_counterfactual_ablation_and_replace(L, settings=settings, use_llm=use_llm, author='ablation-ui')
                    elif act == 'redteam_immunize':
                        res = run_adversarial_redteam_and_cross_pollinate(L, settings=settings, use_llm=use_llm, author='redteam-ui')
                    elif act == 'readiness_audit':
                        res = run_readiness_and_falsification_audit(L, settings=settings, use_llm=use_llm, author='readiness-ui')
                    elif act == 'open_ended_rsi':
                        res = run_open_ended_frontier_rsi(L, settings=settings, use_llm=use_llm, author='open-ended-ui')
                    elif act == 'ingest_and_attribute':
                        from ..data.live_flux import ingest_live_forward_bars_and_attribute
                        res = ingest_live_forward_bars_and_attribute(settings, ledger=L, author='execution-ui', log_episode=True)
                    elif act == 'build_store':
                        from ..data.store_builder import ensure_store_panels
                        res = ensure_store_panels(settings, force=bool(payload.get('force', False)), try_remote=True)
                    elif act == 'audit_harness':
                        from ..data.live_flux import audit_harness_and_provenance
                        res = audit_harness_and_provenance(settings, ledger=L, author='auditor-lead')
                    elif act == 'sync_live_flux':
                        from ..data.live_flux import sync_live_flux
                        res = sync_live_flux(settings, ledger=L)
                    elif act == 'dry_run_readiness':
                        from ..data.live_flux import run_dry_run_readiness
                        res = run_dry_run_readiness(settings, ledger=L, author='auditor-lead')
                    elif act == 'step_paper_tick':
                        from ..data.live_flux import step_live_paper_session
                        res = step_live_paper_session(settings, ledger=L)
                    elif act == 'validate_and_promote':
                        res = validate_and_promote_population(
                            L, settings=settings, desk=desk,
                            max_candidates=int(payload.get('max_candidates', 6)),
                            author='truth-machine-ui',
                        )
                    elif act == 'run_funnel':
                        desks = payload.get('desks') or ([desk] if desk else ['macro', 'perps', 'carry', 'pred'])
                        res = run_tournament_funnel(
                            L, settings=settings,
                            n_agents=int(payload.get('n_agents', 50)),
                            desks=desks,
                            use_llm=use_llm,
                            author='funnel-ui',
                        )
                    elif act == 'forward_audit':
                        res = run_forward_survival_audit(
                            L, settings=settings,
                            inject_decay_id=payload.get('inject_decay_id') or None,
                            author='forward-audit-ui',
                        )
                    elif act == 'start_daemon':
                        res = start_rsi_daemon(
                            settings=settings,
                            interval_s=float(payload.get('interval_s', 12.0)),
                            use_llm=use_llm,
                        )
                    elif act == 'stop_daemon':
                        res = stop_rsi_daemon(settings=settings)
                    elif act == 'run_campaign':
                        gens = int(payload.get('generations', 2))
                        desks = payload.get('desks') or ([desk] if desk else ['macro', 'perps', 'carry', 'pred'])
                        res = run_rsi_campaign(L, settings=settings, generations=gens, desks=desks,
                                               max_children_per_gen=int(payload.get('max_children', 2)), use_llm=use_llm)
                    elif act == 'drill_benchmark':
                        cand = propose_charter_patch(L, role=payload.get('role', 'researcher'), pod=desk)
                        res = {'charter': cand, 'drill_gate': evaluate_rsi_drill(L, settings=settings, charter_id=cand['id'])}
                    else:
                        res = evolve_population(L, settings=settings, desk=desk,
                                                max_children=int(payload.get('max_children', 2)),
                                                author='rsi-evolver', use_llm=use_llm)
                    return self._send(200, json.dumps({'ok': True, 'action': act, 'result': res, 'rsi': rsi_snapshot(L, settings=settings)}, default=str))
                self._send(404, json.dumps({'error': 'unknown endpoint'}))
            except Exception as e:
                self._send(500, json.dumps({'error': repr(e)}))

    httpd = ReusableHTTPServer((host, int(port)), H)

    def _bg_seed():
        try:
            from ..company.ledger import Ledger
            from ..company.rsi import ensure_seeded, start_rsi_daemon
            ensure_seeded(settings, Ledger(settings.ledger))
            start_rsi_daemon(settings=settings, interval_s=8.0, use_llm=False)
        except Exception:
            pass

    threading.Thread(target=_bg_seed, daemon=True).start()
    print(f'Desk OS control room: http://{host}:{port}', flush=True)
    httpd.serve_forever()
