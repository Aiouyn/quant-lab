"""The truth machine: deterministic promotion pipeline.

submitted -> dev_eval -> (gate A) -> review -> [officer verdict] -> validating -> (gate B) -> holdout -> (gate C)
-> approved (bundle exported) -> paper -> (portfolio)
Researchers only learn which gates failed on sealed data, never the sealed metrics themselves.
"""
from __future__ import annotations

import asyncio
import json
import time

from ..lab.sandbox import run_job

# Desk-aware Sharpe bars (dev, val). The default was calibrated for intraday crypto; daily cross-asset macro premia run
# at Sharpe 0.5-1.0, so 1.5 made the macro desk unpassable by design. Over 15 dev years Sharpe 0.7 is t ~ 2.7; DSR,
# bootstrap, stress, holdout and forward gates still apply unchanged.
SHARPE_BARS = {'macro': (0.7, 0.6)}


def _bar(r, i, default):
    return SHARPE_BARS.get(r.get('desk'), (1.5, default))[i] if r.get('desk') in SHARPE_BARS else default


DEV_GATES = {
    'dev_sharpe': lambda r: r['summary']['sharpe'] >= _bar(r, 0, 1.5),
    'dev_trades': lambda r: r['summary']['trades'] >= (20 if r.get('desk') == 'carry' else 60),
    'causal': lambda r: bool(r.get('causality') and r['causality'].get('ok')),
    'engine_certified': lambda r: bool(r.get('certification', {}).get('ok')),
    'fees_x1.5_profitable': lambda r: r['robustness']['fees_x1.5']['pnl'] > 0,
    'latency_+2s_profitable': lambda r: r['robustness']['latency_+2s']['pnl'] > 0,
    'both_halves_positive': lambda r: r['robustness']['first_half_sharpe'] > 0 and r['robustness']['second_half_sharpe'] > 0,
    'positive_months>=50%': lambda r: r['robustness']['positive_months'] >= 0.5,
    'missed_entries_25%_profitable': lambda r: 'missed_entries_25%' not in r['robustness']
                                             or r['robustness']['missed_entries_25%']['pnl'] > 0,
    'queue_x2_profitable': lambda r: 'queue_x2' not in r['robustness'] or r['robustness']['queue_x2']['pnl'] > 0,
    # tail-aware (2026-10-03 lesson: crash-rebound / lottery payoffs look great on the mean and die out of sample)
    'dev_mean_ex_top5_days>0': lambda r: r['summary'].get('mean_ex_top5_days_bps', 1.0) > 0,
    'param_neighborhood>=60%': lambda r: (r['robustness'].get('param_neighborhood') or {}).get('fraction_profitable') is None
                                         or r['robustness']['param_neighborhood']['fraction_profitable'] >= 0.6,
}


def val_gates(s):
    return {
        'val_sharpe>=%.1f' % s.min_val_sharpe: lambda r: r['summary']['sharpe'] >= _bar(r, 1, s.min_val_sharpe),
        'deflated_sharpe>=%.2f' % s.min_dsr: lambda r: (r['stats'].get('dsr') or 0) >= s.min_dsr,
        'bootstrap_lower>0': lambda r: (r['stats'].get('bootstrap_lower_mean_bps') or -1) > 0,
        'positive_weeks>=%.0f%%' % (100 * s.min_positive_weeks): lambda r: r['stats']['positive_weeks'] >= s.min_positive_weeks,
        'trades>=%d' % s.min_val_trades: lambda r: r['summary']['trades'] >= (12 if r.get('desk') == 'carry' else s.min_val_trades),
        'top5_days<%.0f%%' % (100 * s.max_top5_day_share): lambda r: r['stats']['top5_day_share'] < s.max_top5_day_share,
        'max_drawdown<=%.2f' % s.max_drawdown: lambda r: r['summary']['max_drawdown'] <= s.max_drawdown * 1.0,
        'stress_profitable': lambda r: r['stress']['pnl'] > 0,
        'mean_ex_top5_days>0': lambda r: r['stats'].get('mean_ex_top5_days_bps', 1.0) > 0,
        'causal': lambda r: bool(r.get('causality') and r['causality'].get('ok')),
    }


def holdout_gates(s):
    return {
        'holdout_profitable': lambda r: r['summary']['pnl'] > 0,
        'holdout_sharpe>=%.1f' % s.min_holdout_sharpe: lambda r: r['summary']['sharpe'] >= s.min_holdout_sharpe,
        'holdout_drawdown': lambda r: r['summary']['max_drawdown'] <= s.max_drawdown,
    }


def check(gates, report):
    failed = []
    for name, g in gates.items():
        try:
            if not g(report):
                failed.append(name)
        except Exception:
            failed.append(name)
    return failed


class Pipeline:
    def __init__(self, settings, ledger=None, org=None):
        from ..company.ledger import Ledger
        self.s = settings
        self.L = ledger or Ledger(settings.ledger)
        self.org = org
        self.queue: asyncio.Queue | None = None
        self.pending: set = set()

    def cost(self):
        return self.org.cost_params() if self.org else {}

    def enqueue(self, sid):
        if self.queue is not None and sid not in self.pending:
            self.pending.add(sid)
            self.queue.put_nowait(sid)

    async def run(self, workers=3):
        self.queue = asyncio.Queue()
        self.pending = set()
        for r in self.L.q("SELECT id FROM strategies WHERE status IN ('submitted','dev_eval','validating','holdout') ORDER BY created"):
            self.enqueue(r['id'])
        await asyncio.gather(self._rescan(), *[self._worker() for _ in range(workers)])

    async def _rescan(self, every_s=600):
        """Pick up submissions that reached 'submitted' outside the normal path (operator re-queue, a crash mid-submit)."""
        while True:
            await asyncio.sleep(every_s)
            for r in self.L.q("SELECT id FROM strategies WHERE status='submitted' ORDER BY created"):
                self.enqueue(r['id'])

    async def _worker(self):
        while True:
            sid = await self.queue.get()
            self.pending.discard(sid)
            try:
                await self.process_async(sid)
            except Exception as e:
                self.L.event('pipeline', 'PIPELINE_ERROR', {'id': sid, 'error': repr(e)[:800]})
                self.L.update_strategy(sid, status='error', gate_reasons=[repr(e)[:500]])

    def process(self, sid):
        return asyncio.run(self.process_async(sid))

    async def _job(self, job, timeout=5400):
        slot = self.org.job_slots if self.org else None
        r = await run_job(job, self.s, timeout_s=timeout, slot=slot)
        if str(r.get('error', '')).startswith(('harness failed (memory', 'harness failed (system_memory')):
            # our infrastructure ran out of room, not the strategy: retry once with a larger cap (the memory gate waits
            # until the machine can actually provide it)
            self.L.event('pipeline', 'EVAL_RETRY_MEMORY', {'type': job.get('type'), 'split': job.get('split')})
            r = await run_job(job, self.s, timeout_s=timeout, slot=slot, mem_mb=8000)
        return r

    def _notify(self, s, text, also=()):
        for who in {s['author'], *also}:
            self.L.send(who, 'pipeline', text)

    def _split_complete(self, split):
        from ..data.store import FeatureStore
        st = FeatureStore(self.s)
        sp = st.splits()
        return len(st.days(split)) >= len(sp[split])

    async def _wait_for_data(self, sid, split):
        self.L.event('pipeline', 'WAITING_FOR_DATA', {'id': sid, 'split': split})
        await asyncio.sleep(900)
        self.enqueue(sid)

    async def process_async(self, sid):
        s = self.L.strategy(sid)
        try:
            from ..lab.strategy import load_strategy
            okx_desk = load_strategy(s['code']).DESK == 'l2'
        except Exception:
            okx_desk = True
        # integrity: never validate or seal on a partially ingested split (OKX store; other desks have fixed panels)
        for status, split in (('validating', 'val'), ('holdout', 'holdout')):
            if okx_desk and s['status'] == status and not self._split_complete(split):
                if self.queue is not None:
                    asyncio.create_task(self._wait_for_data(sid, split))
                return
        if okx_desk and s['status'] == 'validating' and not self._split_complete('holdout'):
            if self.queue is not None:
                asyncio.create_task(self._wait_for_data(sid, 'holdout'))
            return
        if s['status'] in ('submitted', 'dev_eval'):
            self.L.update_strategy(sid, status='dev_eval')
            trials = self.L.trial_stats()
            r = await self._job({'type': 'strategy_eval', 'code': s['code'], 'split': 'dev', 'allowed': ['dev'],
                                 'robustness': True, 'certify': True, 'cost': self.cost(),
                                 'n_trials': max(trials['distinct_strategies'], 1), 'trial_sr_std': trials['sr_std_daily']})
            if 'error' in r:
                if str(r['error']).startswith('harness failed'):
                    # infrastructure failure (memory/timeout/crash): NOT a verdict on the strategy and not a trial
                    self.L.update_strategy(sid, status='eval_error', dev_report=r, gate_reasons=['infrastructure_error'])
                    self._notify(s, f'[{sid}] {s["name"]}: the lab could not evaluate it ({r["error"][:200]}). This is an '
                                    'infrastructure problem, not a rejection; operators were alerted. Do not resubmit as is.')
                    self.L.event('pipeline', 'EVAL_INFRA_ERROR', {'id': sid, 'error': str(r['error'])[:200]})
                    return
                self.L.update_strategy(sid, status='rejected_dev', dev_report=r, gate_reasons=['evaluation_error'])
                self._notify(s, f'[{sid}] {s["name"]}: evaluation failed: {r["error"][:600]}')
                return
            self.L.trial(s['author'], 'submission', r['code_hash'], r['summary']['sharpe'], r['summary']['days'])
            failed = check(DEV_GATES, r)
            if failed:
                self.L.update_strategy(sid, status='rejected_dev', dev_report=r, gate_reasons=failed)
                self._notify(s, f'[{sid}] {s["name"]} did not pass the full-dev gate: {failed}. Dev summary: '
                                f'{json.dumps(r["summary"])[:700]}')
                self.L.event('pipeline', 'DEV_REJECT', {'id': sid, 'failed': failed})
                return
            self.L.update_strategy(sid, status='review', dev_report=r, gate_reasons=[])
            bid = self.L.post('review', f'Review {sid}: {s["name"]}', f'Author {s["author"]} ({s["pod"]}). Thesis: {s["thesis"][:1500]}',
                              'pipeline', pod='risk', tags=['review', sid])
            for o in self.L.q("SELECT name FROM agents WHERE role='officer'"):
                self.L.send(o['name'], 'pipeline', f'[{bid}] New submission {sid} awaits review (passed full-dev gate).')
            self._notify(s, f'[{sid}] passed the full-dev gate (dev Sharpe {r["summary"]["sharpe"]:.2f}); now with risk officers.')
            self.L.event('pipeline', 'DEV_PASS', {'id': sid, 'sharpe': r['summary']['sharpe']})
            return
        if s['status'] == 'validating':
            # Selection on VALIDATION data happens only among strategies evaluated on it, so the deflation count is
            # the number of validation evaluations (dev-side search is what validation itself tests). Null SR noise.
            n_val = self.L.q("SELECT COUNT(*) n FROM strategies WHERE val_report IS NOT NULL")[0]['n'] + 1
            job = {'type': 'strategy_eval', 'code': s['code'], 'split': 'val', 'allowed': ['dev', 'val'], 'cost': self.cost(),
                   'return_daily': True, 'stress': True, 'n_trials': n_val}
            r = await self._job(job)
            if 'error' in r:
                self.L.update_strategy(sid, status='error', val_report=r, gate_reasons=['validation_error'])
                self._notify(s, f'[{sid}] validation run failed with an error (see officer).')
                return
            failed = check(val_gates(self.s), r)
            if failed:
                self.L.update_strategy(sid, status='rejected_val', val_report=r, gate_reasons=failed)
                self._notify(s, f'[{sid}] {s["name"]} FAILED sealed validation gates: {failed}. (Sealed metrics are not '
                                'disclosed; resubmitting tuned variants counts as more trials.)')
                self.L.event('pipeline', 'VAL_REJECT', {'id': sid, 'failed': failed})
                return
            self.L.update_strategy(sid, status='holdout', val_report=r, gate_reasons=[])
            self.L.event('pipeline', 'VAL_PASS', {'id': sid, 'sharpe': r['summary']['sharpe'], 'dsr': r['stats'].get('dsr')})
            s = self.L.strategy(sid)
        if s['status'] == 'holdout':
            used = self.L.q('SELECT strategy FROM holdout_uses WHERE code_hash=?', (s['code_hash'],))
            if used and used[0]['strategy'] != sid:
                self.L.update_strategy(sid, status='rejected_holdout', gate_reasons=['holdout_already_used_by_identical_code'])
                return
            self.L.x('INSERT OR REPLACE INTO holdout_uses VALUES(?,?,?,?)', (s['code_hash'], sid, time.time(), None))
            r = await self._job({'type': 'strategy_eval', 'code': s['code'], 'split': 'holdout', 'allowed': ['dev', 'val', 'holdout'],
                                 'cost': self.cost(), 'return_daily': True, 'stress': True})
            if 'error' in r:
                self.L.update_strategy(sid, status='error', holdout_report=r, gate_reasons=['holdout_error'])
                return
            failed = check(holdout_gates(self.s), r)
            self.L.x('UPDATE holdout_uses SET passed=? WHERE code_hash=?', (0 if failed else 1, s['code_hash']))
            if failed:
                self.L.update_strategy(sid, status='rejected_holdout', holdout_report=r, gate_reasons=failed)
                self._notify(s, f'[{sid}] {s["name"]} passed validation but FAILED the one-shot holdout. It is retired.')
                self.L.event('pipeline', 'HOLDOUT_REJECT', {'id': sid})
                return
            self.L.update_strategy(sid, status='approved', holdout_report=r, gate_reasons=[])
            from .bundle import export_bundle
            path = export_bundle(self.s, self.L.strategy(sid), self.cost())
            pms = [x['name'] for x in self.L.q("SELECT name FROM agents WHERE role IN ('pm','cio')")]
            self._notify(s, f'[{sid}] {s["name"]} is APPROVED (validation + holdout passed). Bundle: {path}', also=pms)
            self.L.event('pipeline', 'APPROVED', {'id': sid, 'bundle': str(path)})

    async def verdict(self, sid, officer, decision, rationale):
        s = self.L.strategy(sid)
        if not s:
            return {'error': 'not found'}
        if s['status'] != 'review':
            return {'error': f'strategy is in status {s["status"]}, not review'}
        if officer == s['author']:
            return {'error': 'authors cannot review their own work'}
        review = (s.get('review') or []) if isinstance(s.get('review'), list) else []
        review.append({'officer': officer, 'decision': decision, 'rationale': rationale, 'time': time.time()})
        if decision == 'approve':
            self.L.update_strategy(sid, status='validating', review=review)
            self.enqueue(sid)
            self._notify(s, f'[{sid}] approved for sealed validation by {officer}: {rationale[:500]}')
        elif decision == 'revise':
            self.L.update_strategy(sid, status='revise', review=review)
            self._notify(s, f'[{sid}] revision requested by {officer}: {rationale[:1500]}. Resubmit as a new strategy.')
        else:
            self.L.update_strategy(sid, status='rejected_review', review=review)
            self._notify(s, f'[{sid}] rejected by {officer}: {rationale[:1500]}')
        self.L.event(officer, 'VERDICT', {'id': sid, 'decision': decision})
        for b in self.L.q("SELECT id FROM board WHERE kind='review' AND title LIKE ? AND status<>'done'", (f'Review {sid}%',)):
            self.L.update_board(b['id'], officer, 'done', f'{decision}: {rationale[:800]}')
        return {'ok': True, 'status': self.L.strategy(sid)['status']}

    async def forward_check(self):
        """Rolling forward holdout, per desk. l2: ingest OKX archive days that appeared after the split was frozen;
        perps: forward days are appended daily by desk.data.refresh from verified live recordings. Every approved
        strategy is scored on ALL its desk's forward days (never seen by anyone) -> forward_report with a verdict:
        FORWARD_CONFIRMED after >= 60 forward days with Sharpe >= 0.5, positive pnl and positive mean ex top-5 days;
        FORWARD_FAILED after >= 60 days otherwise; RUNNING before."""
        from ..data.store import FeatureStore
        from ..lab.strategy import load_strategy
        import asyncio as _a
        fwd_days = {}
        try:
            from ..data import external
            from ..data.ingest import ingest, day_start_ms
            store = FeatureStore(self.s)
            sp = store.splits(refresh=True)
            missing = [d for d in sp['forward'] if not (self.s.store / '1s' / f'{d}.parquet').exists()]
            if missing:
                await _a.to_thread(external.build_trades, missing, self.s.store)
                await _a.to_thread(external.fetch_hl_funding, self.s.store, day_start_ms(sp['dev'][0]) - 86_400_000)
                await _a.to_thread(ingest, self.s, 4)
            fwd_days['l2'] = len(store.days('forward'))
        except Exception as e:
            self.L.event('pipeline', 'FORWARD_L2_ERROR', {'error': repr(e)[:400]})
        try:
            import polars as _pl
            from ..lab.perps import PerpsStore
            ps = PerpsStore(self.s, allowed=('dev', 'val', 'holdout', 'forward'))
            a0, _ = ps.range_ms('forward')
            fwd_days['perps'] = int(ps.panel().filter(_pl.col('ts') >= a0).select((_pl.col('ts') // 86_400_000).n_unique()).item())
        except Exception as e:
            self.L.event('pipeline', 'FORWARD_PERPS_ERROR', {'error': repr(e)[:400]})
        try:
            from ..lab.macro import MacroStore
            ms = MacroStore(self.s, allowed=('dev', 'val', 'holdout', 'forward'))
            a0, _ = ms.range_dates('forward')
            fwd_days['macro'] = int(ms.panel().filter((_pl.col('date') >= a0) & (_pl.col('tradable') == 1))['date'].n_unique())
        except Exception as e:
            self.L.event('pipeline', 'FORWARD_MACRO_ERROR', {'error': repr(e)[:400]})
        try:
            from ..lab.pred import PredStore
            qs = PredStore(self.s, allowed=('dev', 'val', 'holdout', 'forward'))
            a0, _ = qs.range_ms('forward')
            fwd_days['pred'] = int(qs.panel().filter(_pl.col('ts') >= a0).select((_pl.col('ts') // 86_400_000).n_unique()).item())
        except Exception as e:
            self.L.event('pipeline', 'FORWARD_PRED_ERROR', {'error': repr(e)[:400]})
        out = {}
        for r in self.L.q("SELECT id FROM strategies WHERE status IN ('approved','paper')"):
            st = self.L.strategy(r['id'])
            try:
                desk = load_strategy(st['code']).DESK
            except Exception:
                continue
            n = fwd_days.get(desk, 0)
            prev = st.get('forward_report') or {}
            if not n or prev.get('days') == n:
                continue
            res = await self._job({'type': 'strategy_eval', 'code': st['code'], 'split': 'forward',
                                   'allowed': ['dev', 'val', 'holdout', 'forward'], 'cost': self.cost(), 'return_daily': True})
            if 'error' in res:
                continue
            sm = res['summary']
            ex = (res.get('stats') or {}).get('mean_ex_top5_days_bps', 0)
            verdict = 'RUNNING' if n < 60 else ('FORWARD_CONFIRMED' if sm['sharpe'] >= 0.5 and sm['pnl'] > 0 and ex > 0
                                                else 'FORWARD_FAILED')
            rep = {'days': n, 'desk': desk, 'summary': sm, 'stats': res.get('stats'), 'daily': res.get('daily'),
                   'verdict': verdict, 'updated': time.time()}
            self.L.update_strategy(r['id'], forward_report=rep)
            out[r['id']] = {'days': n, 'sharpe': sm['sharpe'], 'verdict': verdict}
            if verdict != prev.get('verdict') or (sm['pnl'] < 0 and n >= 10):
                for who in [x['name'] for x in self.L.q("SELECT name FROM agents WHERE role IN ('pm','cio')")] + [st['author']]:
                    self.L.send(who, 'pipeline', f'[{r["id"]}] rolling forward holdout ({n} unseen days): pnl {sm["pnl"]:.0f}, '
                                                 f'Sharpe {sm["sharpe"]:.2f}, verdict {verdict}.')
            self.L.event('pipeline', 'FORWARD_SCORED', {'id': r['id'], 'days': n, 'verdict': verdict})
        return {'forward_days': fwd_days, 'scored': out}

    def approved_summary(self):
        out = []
        for r in self.L.q("SELECT id FROM strategies WHERE status IN ('approved','paper') ORDER BY updated DESC"):
            s = self.L.strategy(r['id'])
            v, h = s.get('val_report') or {}, s.get('holdout_report') or {}
            out.append({'id': s['id'], 'name': s['name'], 'author': s['author'], 'pod': s['pod'], 'resolution': s['resolution'],
                        'val': v.get('summary'), 'val_stats': v.get('stats'), 'holdout': h.get('summary'), 'status': s['status']})
        return out
