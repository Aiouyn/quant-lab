"""python -m desk <command>"""
from __future__ import annotations

import argparse
import json
import sys


def _print(x):
    print(json.dumps(x, indent=1, default=str) if not isinstance(x, str) else x, flush=True)


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
    r.add_argument('--drill', choices=['planted', 'null'], help='run on a synthetic drill market instead of real data')
    r.add_argument('--tokens-per-hour', type=int, help='optional hard cap on LLM tokens per rolling hour')
    sub.add_parser('stop', help='ask the running company to stop gracefully')
    stt = sub.add_parser('status', help='company status, KPIs and leaderboard'); stt.add_argument('--drill', choices=['planted', 'null'])
    s = sub.add_parser('serve', help='control-room dashboard'); s.add_argument('--port', type=int, default=8900)
    s.add_argument('--drill', choices=['planted', 'null'])
    sub.choices['stop'].add_argument('--drill', choices=['planted', 'null'])
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
    rs.add_argument('--drill', choices=['planted', 'null'])
    sub.add_parser('forward', help='ingest new archive days and score approved strategies on the rolling forward holdout')
    v = sub.add_parser('validate', help='run validation pipeline on a submitted strategy'); v.add_argument('strategy_id')
    dr = sub.add_parser('drill', help='build or score drill markets (measure the firm itself)')
    dr.add_argument('kind', choices=['planted', 'null']); dr.add_argument('--report', action='store_true')
    dr.add_argument('--days', type=int, default=90)
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
    if a.cmd == 'status':
        from .company.ledger import Ledger
        _print(Ledger(s.ledger).status())
        return 0
    if a.cmd == 'serve':
        from .web.server import serve
        serve(s, a.port)
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
        from .lab.drills import build_drill, drill_report
        _print(drill_report(s, a.kind) if a.report else build_drill(s, a.kind, n_days=a.days))
        return 0
    return 2
