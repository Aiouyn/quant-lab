"""python -m desk <command>"""
from __future__ import annotations

import argparse
import json
import sys


def _print(x):
    print(json.dumps(x, indent=1, default=str) if not isinstance(x, str) else x, flush=True)


DRILLS = ['planted', 'null', 'perps-planted', 'perps-null', 'macro-planted', 'macro-null']


def main(argv=None):
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    p = argparse.ArgumentParser(prog='desk', description='Desk OS v3 - autonomous quant research company')
    sub = p.add_subparsers(dest='cmd', required=True)
    d = sub.add_parser('data', help='build the feature store (trades, funding, L2 reconstruction)')
    d.add_argument('--workers', type=int, default=10)
    d.add_argument('--limit', type=int)
    d.add_argument('--skip-trades', action='store_true')
    d.add_argument('--overwrite', action='store_true')
    sub.add_parser('splits', help='show the frozen dev/val/holdout/forward split')
    cd = sub.add_parser('carry-data', help='download/build the cross-venue funding panel (Hyperliquid vs Binance)')
    cd.add_argument('--coins', type=int, default=30); cd.add_argument('--workers', type=int, default=3)
    cd.add_argument('--bybit', action='store_true', help='also build Hyperliquid-vs-Bybit pairs (COIN@BY)')
    tp = sub.add_parser('tape', help='build the price-level OKX trade tape (maker / limit-order fill model)')
    tp.add_argument('--workers', type=int, default=4)

    r = sub.add_parser('run', help='run the research company')
    r.add_argument('--agents', type=int, default=16)
    r.add_argument('--hours', type=float, default=8.0)
    r.add_argument('--concurrency', type=int)
    r.add_argument('--drill', choices=DRILLS, help='run on a synthetic drill market instead of real data')
    r.add_argument('--tokens-per-hour', type=int, help='optional hard cap on LLM tokens per rolling hour')
    sub.add_parser('stop', help='ask the running company to stop gracefully')
    stt = sub.add_parser('status', help='company status, KPIs and leaderboard'); stt.add_argument('--drill', choices=DRILLS)
    s = sub.add_parser('serve', help='control-room dashboard'); s.add_argument('--port', type=int, default=8900)
    s.add_argument('--host', default='127.0.0.1', help='bind address (default 127.0.0.1; the dashboard has no auth)')
    s.add_argument('--drill', choices=DRILLS)
    sub.choices['stop'].add_argument('--drill', choices=DRILLS)
    pp = sub.add_parser('paper', help='Hyperliquid paper trading for a bundle or portfolio')
    pp.add_argument('target'); pp.add_argument('--hours', type=float, default=24 * 7)
    pp.add_argument('--features', choices=['okx', 'hl'], default='okx', help='live feature source (okx = research-identical)')
    pf = sub.add_parser('portfolio', help='build a portfolio from approved strategies')
    pf.add_argument('--max-members', type=int, default=8)
    rec = sub.add_parser('record', help='record Hyperliquid public L2/trades/asset ctx')
    rec.add_argument('--hours', type=float, default=24)
    rec.add_argument('--coins', default='', help='comma list for the multi-asset recorder (e.g. BTC,ETH,SOL,HYPE)')
    rec.add_argument('--archive', default='', help='directory to archive compacted days (e.g. a Google Drive folder)')
    rs = sub.add_parser('rescore-factors', help='re-score registered factors under the current scoring rule')
    rs.add_argument('--drill', choices=DRILLS)
    sub.add_parser('forward', help='ingest new archive days and score approved strategies on the rolling forward holdout')
    v = sub.add_parser('validate', help='run validation pipeline on a submitted strategy'); v.add_argument('strategy_id')
    dr = sub.add_parser('drill', help='build or score drill markets (measure the firm itself)')
    dr.add_argument('kind', choices=['planted', 'null', 'scoreboard']); dr.add_argument('--report', action='store_true')
    dr.add_argument('--days', type=int, default=90)
    dr.add_argument('--desk', choices=['l2', 'perps', 'macro'], default='l2')
    rsi_p = sub.add_parser('rsi', help='RSI telemetry: status, scoreboard, compile-hints, build-store, quarantine '
                                       '(move rows written by the retired RSI engine out of the ledger)')
    rsi_p.add_argument('action', nargs='?', default='status',
                       choices=['status', 'scoreboard', 'compile-hints', 'build-store', 'quarantine'])
    rsi_p.add_argument('--apply', action='store_true', help='quarantine: actually move rows (default is a dry run)')
    pr = sub.add_parser('process', help='process changes for agent charters: list, propose, trial (paired drill A/B), '
                                        'retire. A change goes live only when a trial accepts it.')
    pr.add_argument('action', nargs='?', default='list', choices=['list', 'propose', 'trial', 'retire', 'loop', 'prune', 'purge-errors'])
    pr.add_argument('variant_id', nargs='?', help='trial/retire: variant id (trial defaults to the oldest proposal)')
    pr.add_argument('--role'); pr.add_argument('--pod'); pr.add_argument('--patch'); pr.add_argument('--rationale')
    pr.add_argument('--desk', choices=['perps', 'macro'], default='perps')
    pr.add_argument('--pairs', type=int, default=3); pr.add_argument('--hours', type=float, default=2.0)
    pr.add_argument('--agents', type=int, default=8)
    pr.add_argument('--serial', action='store_true', help='trial: run the 3 drills of each pair one after another')
    op = sub.add_parser('operator', help='post an operator directive shown first in every agent brief (no text: show the current one)')
    op.add_argument('text', nargs='?')
    cm = sub.add_parser('costs', help='review cost-model proposals from agents (loosening needs a human)')
    cm.add_argument('action', nargs='?', default='list', choices=['list', 'approve', 'reject'])
    cm.add_argument('proposal_id', nargs='?', type=int)
    kn = sub.add_parser('kernel', help='evaluator kernel manifest: verify, or seal after reviewing a kernel change')
    kn.add_argument('action', nargs='?', default='verify', choices=['verify', 'seal', 'guard'])
    a = p.parse_args(argv)

    from .config import SETTINGS
    import dataclasses
    s = SETTINGS.ensure()
    if getattr(a, 'drill', None) and a.cmd in ('status', 'serve', 'stop', 'rescore-factors'):
        s = dataclasses.replace(s, home=s.home / 'drills' / a.drill)
    if a.cmd == 'data':
        from .data.ingest import archive_inventory, ingest
        from .data import external
        inv = archive_inventory(s.data_dir)
        dates = [r['date'] for r in inv][: a.limit] if a.limit else [r['date'] for r in inv]
        if not a.skip_trades:
            _print(f'trades: aggregating OKX public prints for {len(dates)} days')
            _print(external.build_trades(dates, s.store))
            from .data.ingest import day_start_ms
            try:
                f = external.fetch_hl_funding(s.store, day_start_ms(dates[0]) - 86_400_000)
                _print(f'hyperliquid funding: {len(f)} hourly rates')
            except Exception as e:
                _print(f'hyperliquid funding failed: {e!r}')
            try:
                f = external.fetch_okx_funding(s.store)
                _print(f'okx funding: {len(f)} rates')
            except Exception as e:
                _print(f'okx funding failed: {e!r}')
        res = ingest(s, workers=a.workers, limit=a.limit, overwrite=a.overwrite)
        bad = [r for r in res if 'error' in r]
        _print({'processed': len(res), 'failed': [b['date'] for b in bad]})
        from .data.store import FeatureStore
        _print(FeatureStore(s).splits(refresh=True)['summary'])
        return 1 if bad else 0
    if a.cmd == 'carry-data':
        from .data.carry_data import build_carry_data
        if a.bybit:
            from .data.carry_data import build_bybit_pairs
            res = build_bybit_pairs(s, workers=a.workers)
            _print({'pairs': len(res), 'errors': [r for r in res if 'error' in r or 'skipped' in r]})
            return 0
        res = build_carry_data(s, n=a.coins, workers=a.workers)
        _print({'coins': len(res), 'errors': [r for r in res if 'error' in r or 'skipped' in r]})
        return 0
    if a.cmd == 'tape':
        from .data.ingest import archive_inventory
        from .data.external import build_tape
        _print(build_tape([r['date'] for r in archive_inventory(s.data_dir)], s.store, workers=a.workers))
        return 0
    if a.cmd == 'splits':
        from .data.store import FeatureStore
        _print(FeatureStore(s).splits()['summary'])
        return 0
    if a.cmd == 'run':
        from .company.runtime import run_company
        return run_company(s, agents=a.agents, hours=a.hours, concurrency=a.concurrency, drill=a.drill,
                           tokens_per_hour=a.tokens_per_hour)
    if a.cmd == 'stop':
        from .company.ledger import Ledger
        Ledger(s.ledger).request_stop()
        _print('stop requested; agents finish their current step')
        return 0
    if a.cmd == 'operator':
        from .company.ledger import Ledger
        L = Ledger(s.ledger)
        if a.text:
            L.memo('operator', 'operator', a.text)
        _print(L.latest_memo('operator') or 'no operator directive')
        return 0
    if a.cmd == 'status':
        from .company.ledger import Ledger
        _print(Ledger(s.ledger).status())
        return 0
    if a.cmd == 'serve':
        from .web.server import serve
        serve(s, a.port, host=a.host)
        return 0
    if a.cmd == 'paper':
        from .live.paper import run_paper
        return run_paper(s, a.target, hours=a.hours, feature_source=a.features)
    if a.cmd == 'portfolio':
        from .pipeline.portfolio import build_portfolio
        _print(build_portfolio(s, max_members=a.max_members))
        return 0
    if a.cmd == 'record':
        if a.coins:
            import asyncio as _a
            from pathlib import Path as _P
            from .live.multi_recorder import record_multi
            from .live.okx_feed import okx_features
            from .live.recorder import record_async

            async def run_all():
                coins = tuple(x.strip().upper() for x in a.coins.split(',') if x.strip())
                return await _a.gather(record_multi(s, coins, a.hours, _P(a.archive) if a.archive else None),
                                       okx_features(s, a.hours), record_async(s, a.hours))
            _print(_a.run(run_all()))
            return 0
        from .live.recorder import record
        return record(s, hours=a.hours)
    if a.cmd == 'rescore-factors':
        import json as _j
        from .company.ledger import Ledger
        from .lab.factors import rescore
        L = Ledger(s.ledger)
        changed = []
        for f in L.q("SELECT id, name, report, status FROM factors WHERE report IS NOT NULL AND status IN ('active','weak')"):
            rep = rescore(_j.loads(f['report']))
            if 'horizons' not in rep or rep.get('best_t') is None:
                continue
            nov = (rep.get('novelty') or {}).get('max_abs_corr', 0.0) or 0.0
            score = rep['economic_value'] * (1 - 0.6 * nov)
            status = 'active' if rep.get('causal') and abs(rep['best_t']) >= 3.0 else 'weak'
            L.factor_report(f['id'], rep, score, status)
            changed.append({'id': f['id'], 'name': f['name'], 'status': f'{f["status"]}->{status}',
                            'best_h': rep['best_horizon_s'], 't': rep['best_t'], 'score': round(score, 2)})
        _print(changed)
        return 0
    if a.cmd == 'forward':
        import asyncio
        from .pipeline.validation import Pipeline
        _print(asyncio.run(Pipeline(s).forward_check()))
        return 0
    if a.cmd == 'validate':
        from .pipeline.validation import Pipeline
        _print(Pipeline(s).process(a.strategy_id))
        return 0
    if a.cmd == 'drill':
        from .company.ledger import Ledger
        from .lab.panel_drills import build_panel_drill, panel_drill_report, record_drill_score
        main_ledger = Ledger(s.ledger)
        if a.kind == 'scoreboard':
            _print(main_ledger.get('drill_scores', None) or {'note': 'no drill scored yet'})
            return 0
        if a.desk == 'l2':
            from .lab.drills import build_drill, drill_report
            if a.report:
                res = drill_report(s, a.kind)
                res.update({'drill': a.kind, 'desk': 'l2', 'kind': a.kind, 'scored_at': __import__('time').time()})
                record_drill_score(main_ledger, res)
                _print(res)
            else:
                _print(build_drill(s, a.kind, n_days=a.days))
            return 0
        _print(panel_drill_report(s, a.desk, a.kind, record_in=main_ledger) if a.report
               else build_panel_drill(s, a.desk, a.kind))
        return 0
    if a.cmd == 'kernel':
        from .kernel import seal, verify
        if a.action == 'seal':
            _print(seal())
            return 0
        if a.action == 'guard':                   # which OS-level protection the strategy jail gets on this machine
            from .lab import jail
            res, _ = jail.call('meta', "DESK='perps'\nPARAMS={}\ndef signal(panel, p):\n    return 0\n")
            g = res.get('kernel_guard') or 'none'
            _print({'kernel_guard': g, 'ok_for_unattended_runs': 'landlock' in g and 'seccomp' in g,
                    'note': 'landlock+seccomp = enforced by the OS. Anything else: Python-level guard only; '
                            'run under WSL2 or Linux for unattended runs.'})
            return 0 if 'landlock' in g else 1
        res = verify()
        _print(res)
        return 0 if res['ok'] else 1
    if a.cmd == 'process':
        from .company.ledger import Ledger
        from .lab import process_trials as PT
        L = Ledger(s.ledger)
        if a.action == 'list':
            _print({'live': [{k: r[k] for k in ('id', 'role', 'pod', 'patch_text')} for r in PT.live_variants(L)],
                    'history': PT.history(L)})
            return 0
        if a.action == 'propose':
            res = PT.propose(L, a.role, a.patch, a.rationale, author='human-cli', pod=a.pod)
            _print(res)
            return 1 if 'error' in res else 0
        if a.action == 'retire':
            ok = PT.retire(L, a.variant_id, a.rationale or 'retired by human')
            print(f'{a.variant_id} retired' if ok else f'{a.variant_id} is not active or proposed')
            return 0 if ok else 1
        if a.action == 'purge-errors':               # drop ERROR trials that scored nothing (back up the ledger first)
            _print({'deleted': PT.purge_empty_errors(L)})
            return 0
        if a.action == 'prune':                      # delete market copies of finished trials (ledgers are kept)
            _print(PT.prune_finished(s, L))
            return 0
        if a.action == 'loop':                       # supervisor service: trial every queued proposal, forever
            from .ops.supervisor import code_hash
            h0 = code_hash()
            PT.trial_loop(s, desk=a.desk, pairs=a.pairs, hours=a.hours, agents=a.agents,
                          stop=lambda: code_hash() != h0, log=lambda m: print(m, flush=True))
            return 0
        vid = a.variant_id or PT.next_proposed(L)
        if not vid:
            print('no proposed process variant to trial')
            return 1
        _print(PT.run_trial(s, vid, desk=a.desk, pairs=a.pairs, hours=a.hours, agents=a.agents, parallel=not a.serial))
        return 0
    if a.cmd == 'costs':
        from .company.ledger import Ledger
        import json as _json
        import time as _time
        L = Ledger(s.ledger)
        if a.action == 'list':
            cur = L.q('SELECT params, author, created FROM cost_models ORDER BY id DESC LIMIT 1')
            _print({'current': _json.loads(cur[0]['params']) if cur else 'default CostModel',
                    'pending': L.q("SELECT id, author, params, rationale, created FROM cost_model_proposals "
                                   "WHERE status='pending' ORDER BY id")})
            return 0
        row = L.q('SELECT * FROM cost_model_proposals WHERE id=?', (a.proposal_id,))
        if not row or row[0]['status'] != 'pending':
            print(f'no pending proposal #{a.proposal_id}')
            return 1
        status = 'approved' if a.action == 'approve' else 'rejected'
        L.x('UPDATE cost_model_proposals SET status=?, decided=?, decided_by=? WHERE id=?',
            (status, _time.time(), 'human-cli', a.proposal_id))
        if status == 'approved':
            L.x('INSERT INTO cost_models(author,params,rationale,created) VALUES(?,?,?,?)',
                (f"{row[0]['author']} (approved by human)", row[0]['params'], row[0]['rationale'], _time.time()))
        print(f'proposal #{a.proposal_id} {status}')
        return 0
    if a.cmd == 'rsi':
        if a.action == 'quarantine':
            from .company.quarantine import quarantine
            _print(quarantine(s.ledger, apply=a.apply))
            return 0
        if a.action == 'build-store':
            from .data.store_builder import ensure_store_panels
            _print(ensure_store_panels(s, force=True, try_remote=True))
            return 0
        from .company.ledger import Ledger
        from .company.rsi import rsi_snapshot, compile_error_hints, compute_generation_scoreboard
        L = Ledger(s.ledger)
        if a.action == 'compile-hints':
            _print(compile_error_hints(L))
        elif a.action == 'scoreboard':
            _print(compute_generation_scoreboard(L))
        else:
            _print(rsi_snapshot(L, settings=s))
        return 0
    return 2
