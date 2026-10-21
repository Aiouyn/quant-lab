"""Agent tools: schemas per role and async implementations. Results are compact JSON (full artifacts on disk)."""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from pathlib import Path

from ..lab.identity import strategy_desk
from ..pipeline.validation import _host_identity
from ..lab.sandbox import run_python, run_job
from ..lab.strategy import code_hash, load_strategy, StrategyError

S, I, N, B = {'type': 'string'}, {'type': 'integer'}, {'type': 'number'}, {'type': 'boolean'}


def fn(name, desc, props=None, req=None):
    return {'type': 'function', 'function': {'name': name, 'description': desc,
            'parameters': {'type': 'object', 'properties': props or {}, 'required': req or []}}}


SPECS = {
    # ------------------------------------------------------------ common
    'board_post': fn('board_post', 'Create a board item: thesis (research line with test plan), task, question, request. '
                     'Assign to a colleague by name to delegate.',
                     {'kind': {'type': 'string', 'enum': ['thesis', 'task', 'question', 'request']}, 'title': S, 'body': S,
                      'assignee': S, 'pod': S, 'tags': {'type': 'array', 'items': S}}, ['kind', 'title', 'body']),
    'board_list': fn('board_list', 'List board items (default: active items of your pod).',
                     {'status': {'type': 'string', 'enum': ['active', 'open', 'claimed', 'in_progress', 'done', 'dropped']},
                      'kind': S, 'pod': S, 'assignee': S, 'limit': I}),
    'board_read': fn('board_read', 'Read a board item with its comment thread.', {'id': S}, ['id']),
    'board_update': fn('board_update', 'Claim/advance/close a board item and/or comment on it.',
                       {'id': S, 'status': {'type': 'string', 'enum': ['open', 'claimed', 'in_progress', 'done', 'dropped']},
                        'comment': S, 'assignee': S}, ['id']),
    'message': fn('message', 'Send a message to a colleague (agent name), a pod ("pod:<name>") or everyone ("all").',
                  {'to': S, 'body': S}, ['to', 'body']),
    'search': fn('search', 'Full-text search across board, playbook lessons, factors and strategies.',
                 {'query': S, 'kind': {'type': 'string', 'enum': ['board', 'lesson', 'factor', 'strategy']}, 'limit': I}, ['query']),
    'read': fn('read', 'Read any artifact by id: factor F-..., strategy S-..., board item, lesson L-..., or "colleagues".', {'id': S}, ['id']),
    'leaderboard': fn('leaderboard', 'Top factors (by score) or strategies (by pipeline stage) of the firm.',
                      {'kind': {'type': 'string', 'enum': ['factors', 'strategies']}, 'limit': I, 'pod': S}, ['kind']),
    'notebook': fn('notebook', 'Your persistent notebook (survives context resets). mode=append|replace|read.',
                   {'mode': {'type': 'string', 'enum': ['append', 'replace', 'read']}, 'text': S}, ['mode']),
    'record_lesson': fn('record_lesson', 'Add a playbook lesson (also negative results). Be specific and quantitative.',
                        {'claim': S, 'evidence': S, 'confidence': N, 'scope': S, 'tags': {'type': 'array', 'items': S}},
                        ['claim', 'evidence', 'confidence']),
    'end_session': fn('end_session', 'End this work session with a concise summary of what you did and what is next.',
                      {'summary': S}, ['summary']),
    # ------------------------------------------------------------ research
    'run_python': fn('run_python', 'Run Python in your sandbox. `from desk.research import *` is ALREADY imported: it is the '
                     'whole API - never import desk.lab / desk.data / desk.settings internals. DEV data only. Print compact results. '
                     'DESKS: macro_panel() [date, symbol, adjclose, adj_open, open/high/low/close, volume, cls, tradable, cost_bps], '
                     'perps_panel() [ts, coin, px, px_fill, f, oi, vlm, isp, premium, bn_*, dvol_btc, vix, dxy, ...], '
                     'pred_panel() [ts, market, p, end_ms, category, hours_to_end]; backtests macro_backtest / perps_backtest / '
                     'pred_backtest(panel, weights); event_study, xs_factor_report, walk_forward; example_strategy(desk), '
                     'example_factor(desk), source(module), research_agenda(). There is NO "ret" column: compute returns. '
                     'POLARS (not pandas): n_unique(), group_by().agg(), sort(), with_columns(), filter(), join(), cum_max(), '
                     'pl.corr(a, b) inside agg, per-asset ops pl.col(x).shift(1).over("symbol"), rolling_mean(n).over(...), '
                     'to_list()/to_numpy(); no .items/.iloc/.loc/.apply/.groupby/.nunique/.cummax. '
                     'Your cwd is your workspace: save files there with relative names.',
                     {'code': S, 'timeout_s': I}, ['code']),
    'register_factor': fn('register_factor', 'Register a causal factor module in the shared Factor Library. OKX l2: '
                          'factor(df, params) with RESOLUTION; PERPS/MACRO desks: set DESK = "perps" | "macro" and '
                          'factor(panel, params) -> value per panel row (cross-sectional IC by horizon, mean quintile '
                          'spread incl. without the 5 best periods, cost hurdle, causality, novelty vs the library). '
                          'Start from example_factor(desk): factor(panel, params) must return ONE value per panel row.',
                          {'name': S, 'code': S, 'description': S}, ['name', 'code', 'description']),
    'evaluate_strategy': fn('evaluate_strategy', 'Backtest a strategy module on its desk DEV period: DESK="perps" '
                            '(hourly, all HL perps, 2023-06..2025-05), DESK="carry", or OKX l2 (sample of 60 days, or '
                            'all=true). Optional robustness battery. Counts as a trial.',
                            {'code': S, 'all': B, 'robustness': B}, ['code']),
    'combine_factors': fn('combine_factors', 'Platform alpha engine: combine causal base features (multi-scale imbalance, '
                          'flow, depth asymmetry, returns, vol, spread) and chosen 1m library factors in a walk-forward '
                          'model (lgbm|ridge, fit only on earlier days) predicting the h-minute forward return; trades only '
                          'when |prediction| > edge_bps. Evaluated on DEV (counts as a trial). Returns metrics + strategy '
                          'code you can submit_strategy as-is or refine. Cheapest way to test a combination idea.',
                          {'factor_ids': {'type': 'array', 'items': S}, 'h_minutes': I, 'maker': B, 'edge_bps': N,
                           'hold_minutes': I, 'model': {'type': 'string', 'enum': ['lgbm', 'ridge']}, 'use_base': B,
                           'exit': {'type': 'string', 'enum': ['maker', 'taker']}}, ['h_minutes']),
    'submit_strategy': fn('submit_strategy', 'Submit a strategy to the truth machine (full-dev + robustness + review + '
                          'sealed validation + one-shot holdout). Only submit robust, mechanism-backed work.',
                          {'name': S, 'code': S, 'thesis': S}, ['name', 'code', 'thesis']),
    # ------------------------------------------------------------ officers
    'review_queue': fn('review_queue', 'Strategies awaiting independent review.'),
    'dossier': fn('dossier', 'Full review dossier of a strategy: code, dev report, robustness, causality, certification.',
                  {'id': S}, ['id']),
    'verdict': fn('verdict', 'Review decision on a strategy: approve (to sealed validation), revise, or reject.',
                  {'id': S, 'decision': {'type': 'string', 'enum': ['approve', 'revise', 'reject']}, 'rationale': S},
                  ['id', 'decision', 'rationale']),
    # ------------------------------------------------------------ cio
    'org_stats': fn('org_stats', 'Organization dashboard: per-pod output, tokens, yield, pipeline funnel, bandit suggestion.'),
    'set_pod_weights': fn('set_pod_weights', 'Set budget weights per pod (priority for compute and LLM slots).',
                          {'weights': {'type': 'object', 'additionalProperties': N}, 'rationale': S}, ['weights', 'rationale']),
    'post_memo': fn('post_memo', 'Publish the Desk Memo shown to everyone at session start.', {'body': S}, ['body']),
    'reassign_agent': fn('reassign_agent', 'Move a researcher to another pod (takes effect next session).',
                         {'agent': S, 'pod': S, 'reason': S}, ['agent', 'pod', 'reason']),
    # ------------------------------------------------------------ pm
    'approved_strategies': fn('approved_strategies', 'Strategies that passed validation (and holdout), with return-stream stats.'),
    'build_portfolio': fn('build_portfolio', 'Build a risk-balanced paper portfolio from approved strategy ids.',
                          {'ids': {'type': 'array', 'items': S}, 'rationale': S}, ['ids', 'rationale']),
    'paper_status': fn('paper_status', 'Status of Hyperliquid paper sessions and live-vs-backtest drift.'),
    'paper_start': fn('paper_start', 'Start a Hyperliquid paper session for a strategy bundle or portfolio id.',
                      {'target': S}, ['target']),
    # ------------------------------------------------------------ execution
    'venue_stats': fn('venue_stats', 'Summary of Hyperliquid captures vs OKX: spreads, depth, sweep impact.'),
    'set_cost_model': fn('set_cost_model', 'Set the firm cost model used by validation (impact_mult, slippage_bps, latency_s).',
                         {'impact_mult': N, 'slippage_bps': N, 'latency_s': I, 'rationale': S}, ['rationale']),
    # ------------------------------------------------------------ librarian / toolsmith
    'retire_lesson': fn('retire_lesson', 'Retire a lesson that is wrong, superseded or duplicated.', {'id': S, 'reason': S}, ['id', 'reason']),
    'write_digest': fn('write_digest', 'Publish the knowledge digest injected into every session brief (<= 600 words).',
                       {'body': S}, ['body']),
    'contribute_primitive': fn('contribute_primitive', 'Add a tested reusable function to desk.lib.contrib (importable in '
                               'sandboxes). Provide a test that prints OK.',
                               {'name': S, 'code': S, 'test': S, 'doc': S}, ['name', 'code', 'test', 'doc']),
    # ------------------------------------------------------------ RSI (Recursive Self-Improvement)
    'evolve_population': fn('evolve_population', 'RSI evolutionary engine: breed top active factors into multi-factor rank '
                            'strategies and apply targeted gate-repair mutations (vol_scale_tail_clip, hysteresis_buffer, '
                            'regime_breadth_gate, maker_execution_switch) across desks.',
                            {'desk': {'type': 'string', 'enum': ['macro', 'perps', 'pred', 'carry', 'l2']}, 'max_children': I}),
    'compile_hints': fn('compile_hints', 'RSI error-to-hint compiler: cluster recent RUN_PYTHON and TOOL_ERROR tracebacks '
                        'into verified dynamic regex fix hints.', {}),
    'primitive_lift': fn('primitive_lift', 'Empirical Bayesian lift leaderboard of desk.lib.contrib primitives '
                         '(ΔSharpe, Δfactor score, sandbox pass rate, total uses).', {}),
    'propose_charter_patch': fn('propose_charter_patch', 'Propose a role/pod charter overlay from gate failure bottlenecks '
                                'and evaluate it on the planted/null drill gate before promotion.',
                                {'role': S, 'pod': S}),
    'evolve_primitives': fn('evolve_primitives', 'Autonomous Toolsmith RSI loop: refactor low-lift/failing desk.lib.contrib '
                            'primitives and synthesize new verified causal primitives from lift telemetry.', {}),
    'run_rsi_campaign': fn('run_rsi_campaign', 'Run a multi-generation RSI campaign (hints -> Toolsmith primitives -> '
                           'LLM+AST strategy breeding -> Truth Machine gate -> ERC Master Portfolio) across desks.',
                           {'generations': I, 'desks': {'type': 'array', 'items': S}, 'use_llm': B}),
    'validate_and_promote': fn('validate_and_promote', 'Run the strict 3-stage Truth Machine gate (DEV_GATES -> sealed val_gates with DSR -> '
                               'one-shot holdout_gates -> bundle export) and auto-build the ERC Master Portfolio.',
                               {'desk': S, 'max_candidates': I}),
    'operator_bandit': fn('operator_bandit', 'Inspect the Meta-Bandit operator leaderboard weighted by out-of-sample val survival and Sharpe lift.',
                          {'desk': S}),
}

COMMON = ['board_post', 'board_list', 'board_read', 'board_update', 'message', 'search', 'read', 'leaderboard',
          'notebook', 'record_lesson', 'end_session']
ROLE_TOOLS = {
    'cio': COMMON + ['org_stats', 'set_pod_weights', 'post_memo', 'reassign_agent', 'approved_strategies', 'paper_status',
                     'evolve_population', 'compile_hints', 'primitive_lift', 'propose_charter_patch', 'run_rsi_campaign',
                     'validate_and_promote', 'operator_bandit'],
    'lead': COMMON + ['run_python', 'register_factor', 'evaluate_strategy', 'combine_factors', 'submit_strategy',
                      'evolve_population', 'primitive_lift', 'run_rsi_campaign', 'validate_and_promote', 'operator_bandit'],
    'researcher': COMMON + ['run_python', 'register_factor', 'evaluate_strategy', 'combine_factors', 'submit_strategy',
                            'evolve_population', 'primitive_lift', 'operator_bandit'],
    'officer': COMMON + ['review_queue', 'dossier', 'verdict', 'run_python', 'evaluate_strategy', 'validate_and_promote'],
    'pm': COMMON + ['approved_strategies', 'build_portfolio', 'paper_status', 'paper_start', 'run_python', 'validate_and_promote'],
    'execution': COMMON + ['venue_stats', 'set_cost_model', 'run_python', 'approved_strategies'],
    'librarian': COMMON + ['retire_lesson', 'write_digest', 'compile_hints', 'propose_charter_patch'],
    'toolsmith': COMMON + ['run_python', 'contribute_primitive', 'compile_hints', 'primitive_lift',
                           'evolve_primitives', 'evolve_population', 'propose_charter_patch', 'operator_bandit'],
}


def specs_for(role):
    return [SPECS[n] for n in ROLE_TOOLS[role]]


def compact(obj, limit=7000):
    s = json.dumps(obj, default=str, ensure_ascii=False)
    if len(s) <= limit:
        return s
    return s[:limit - 200] + f'... [truncated {len(s) - limit + 200} chars; ask for narrower output]'


class Toolbox:
    """Executes tool calls for one agent. `org` is the runtime (ledger, settings, pipeline, slots)."""

    def __init__(self, org, agent):
        self.org, self.agent = org, agent
        self.L = org.ledger
        self.s = org.settings

    async def call(self, name, args):
        if name not in ROLE_TOOLS[self.agent.role]:
            return {'error': f'{name} is not available to role {self.agent.role}'}
        handler = getattr(self, 't_' + name, None)
        if handler is None:
            return {'error': f'unknown tool {name}'}
        try:
            res = await handler(**args)
            # artifact pressure (measured 2026-10-06: 170 sandbox analyses for 1 evaluation): after a streak of
            # analyses without a measured artifact, every further result carries a nudge to test the best idea
            if name in ('evaluate_strategy', 'register_factor', 'submit_strategy'):
                self._streak = 0
            elif name == 'run_python' and self.agent.role in ('researcher', 'lead') and isinstance(res, dict):
                self._streak = getattr(self, '_streak', 0) + 1
                if self._streak >= 8:
                    res['ARTIFACT_CHECK'] = (f'{self._streak} analyses in a row without a measured artifact. Turn your best '
                                             'finding into a strategy NOW: start from example_strategy(desk), run '
                                             'evaluate_strategy (or register_factor from example_factor(desk)). A tested '
                                             'negative result is worth more than another exploration.')
            return res
        except TypeError as e:
            return {'error': f'bad arguments for {name}: {e}'}
        except Exception as e:
            self.L.event(self.agent.name, 'TOOL_ERROR', {'tool': name, 'error': repr(e)[:500]})
            return {'error': f'{type(e).__name__}: {e}'}

    # ------------------------------------------------------------ common
    async def t_board_post(self, kind, title, body, assignee=None, pod=None, tags=None):
        bid = self.L.post(kind, title, body, self.agent.name, pod or self.agent.pod, assignee, tags=tags,
                          status='claimed' if assignee else 'open')
        if assignee and assignee != self.agent.name:
            self.L.send(assignee, self.agent.name, f'[{bid}] assigned to you: {title}')
        return {'id': bid}

    async def t_board_list(self, status='active', kind=None, pod=None, assignee=None, limit=15):
        return self.L.board_list(status, kind, pod if pod != 'all' else None, assignee, min(limit or 15, 40))

    async def t_board_read(self, id):
        return self.L.board_item(id) or {'error': 'not found'}

    async def t_board_update(self, id, status=None, comment=None, assignee=None):
        item = self.L.board_item(id)
        if not item:
            return {'error': 'not found'}
        self.L.update_board(id, self.agent.name, status, comment, assignee)
        notify = {item['author'], item.get('assignee'), assignee} - {self.agent.name, None}
        for who in notify:
            self.L.send(who, self.agent.name, f'[{id}] {status or "comment"}: {(comment or "")[:300]}')
        return {'ok': True}

    async def t_message(self, to, body):
        self.L.send(to, self.agent.name, body)
        return {'sent': True}

    async def t_search(self, query, kind=None, limit=8):
        return self.L.search(query, kind, min(limit or 8, 20))

    async def t_read(self, id):
        if id == 'colleagues':
            return self.L.q('SELECT name, role, pod, status, last_summary FROM agents ORDER BY pod')
        if id.startswith('F-'):
            r = self.L.q('SELECT * FROM factors WHERE id=?', (id,))
            if r:
                r[0]['report'] = json.loads(r[0]['report']) if r[0]['report'] else None
            return r[0] if r else {'error': 'not found'}
        if id.startswith('S-'):
            s = self.L.strategy(id)
            if not s:
                return {'error': 'not found'}
            # researchers never see validation/holdout metrics, only stage + gate reasons
            if self.agent.role not in ('officer', 'pm', 'cio', 'execution'):
                s.pop('val_report', None); s.pop('holdout_report', None)
            return s
        if id.startswith('L-'):
            r = self.L.q('SELECT * FROM lessons WHERE id=?', (id,))
            return r[0] if r else {'error': 'not found'}
        return self.L.board_item(id) or {'error': 'not found'}

    async def t_leaderboard(self, kind, limit=12, pod=None):
        if kind == 'factors':
            rows = self.L.factors(min(limit or 12, 30))
            out = []
            for r in rows:
                if pod and r['pod'] != pod:
                    continue
                rep = r['report'] or {}
                h = rep.get('horizons', {}).get(str(rep.get('best_horizon_s')), {}) if rep else {}
                out.append({'id': r['id'], 'name': r['name'], 'author': r['author'], 'res': r['resolution'],
                            'score': round(r['score'] or 0, 2), 'best_h_s': rep.get('best_horizon_s'), 'best_t': rep.get('best_t'),
                            'rank_ic': h.get('daily_rank_ic_mean'), 'q5_q1_bps': h.get('q5_minus_q1_bps'),
                            'novel_corr': (rep.get('novelty') or {}).get('max_abs_corr')})
            return out
        rows = self.L.strategies(limit=min(limit or 12, 40))
        return [r for r in rows if not pod or r['pod'] == pod]

    async def t_notebook(self, mode, text=None):
        p = self.agent.workspace / 'notebook.md'
        if mode == 'read':
            return {'notebook': p.read_text(encoding='utf-8')[-12000:] if p.exists() else ''}
        if not text:
            return {'error': 'text required'}
        if mode == 'replace':
            p.write_text(text, encoding='utf-8')
        else:
            with open(p, 'a', encoding='utf-8') as f:
                f.write(f'\n\n## {time.strftime("%Y-%m-%d %H:%M")}\n{text}')
        return {'ok': True, 'chars': p.stat().st_size}

    async def t_record_lesson(self, claim, evidence, confidence, scope=None, tags=None):
        lid = self.L.add_lesson(self.agent.name, claim, evidence, float(confidence), scope or self.agent.pod, tags)
        return {'id': lid}

    async def t_end_session(self, summary=None, **kw):
        summary = summary or next((v for v in kw.values() if isinstance(v, str)), '') or '(no summary)'
        self.agent.end_requested = summary
        return {'ok': True}

    # ------------------------------------------------------------ research
    def _ticketed(self, job):
        """Charter R7b: every harness job carries a host-ledger ticket bound to it."""
        return dict(job, _ticket=self.L.issue_ticket(self.agent.name, job, desk=job.get('desk') or 'unknown'))

    async def t_run_python(self, code, timeout_s=300):
        # charter R7c: trials are written host-side from what the host launched. One trial per sandbox run, keyed on
        # the submitted script; nothing the sandbox writes (the old DESK_TRIALS file) is read back as a trial.
        res = await run_python(code, self.agent.workspace, self.s, timeout_s=timeout_s, slot=self.org.sandbox_slots)
        from ..lab.identity import evaluation_identity
        try:
            ident = evaluation_identity(code, {'kind': 'sandbox_run'}, self.s.store, desk='l2')[0]
        except SyntaxError:                                  # unparsable script: still one attested trial
            ident = hashlib.sha256(b'unparsable:' + code.encode()).hexdigest()
        self.L.trial(self.agent.name, 'sandbox_run', hashlib.sha256(code.encode()).hexdigest(), None, None,
                     desk='unknown', split='dev', family='unassigned', identity=ident)
        n = 1
        from .hints import hints, record_hint_resolution
        from .rsi import record_primitive_outcome
        ok = (res['status'] == 'ok')
        record_hint_resolution(self.L, self.agent.name, ok)
        record_primitive_outcome(self.L, code, self.agent.name, 'run_python', success=ok, metric_val=1.0 if ok else 0.0)
        err_line = ''
        if not ok:
            h = hints(res.get('stderr', ''), res['status'], ledger=self.L, agent=self.agent.name)
            if h:
                res['fix_hints'] = h
            lines = [l for l in (res.get('stderr') or '').strip().splitlines() if l.strip()]
            named = [l for l in lines if re.match(r'^[\w.]*(Error|Exception|Exit)\b', l.strip())]
            err_line = (named or lines or [res['status']])[-1].strip()[:200]
        self.L.event(self.agent.name, 'RUN_PYTHON', {'status': res['status'], 'seconds': res['seconds'], 'queued_s': res.get('queued_s', 0), 'trials': n,
                                                     'error': err_line})
        return res

    async def t_register_factor(self, name, code, description):
        dm = re.search(r"^DESK\s*=\s*['\"](perps|macro|pred)['\"]", code, re.M)
        if dm:
            return await self._register_panel_factor(name, code, description, dm[1])
        m = re.search(r"RESOLUTION\s*=\s*['\"](1s|10s|1m)['\"]", code)
        if not m:
            guess = next((d for d, k in (('macro', 'adjclose'), ('macro', "'symbol'"), ('pred', "'market'"), ('perps', "'coin'"))
                          if k in code), None)
            return {'error': 'set the desk at module top: DESK = "macro" | "perps" | "pred" (panel factors), or '
                             'RESOLUTION = "1s" | "10s" | "1m" for OKX L2 book factors.'
                             + (f' Your code looks like DESK = "{guess}".' if guess else '')}
        res = m[1]
        if 'def factor' not in code:
            return {'error': 'factor module must define factor(df, params)'}
        fid = self.L.add_factor(name, self.agent.name, self.agent.pod, code, res, description)
        out = await run_job(self._ticketed({'type': 'factor_eval', 'code': code, 'resolution': res, 'factor_id': fid,
                             'values_dir': str(self.s.artifacts / 'factor_values')}), self.s, timeout_s=1200,
                            slot=self.org.job_slots)
        if 'error' in out:
            self.L.factor_report(fid, out, 0.0, 'failed')
            return {'id': fid, 'status': 'failed', 'error': out['error'], 'trace': out.get('trace', '')[-1500:]}
        t = abs(out.get('best_t') or 0.0)
        nov = out.get('novelty', {}).get('max_abs_corr', 0.0) or 0.0
        score = (out.get('economic_value') or 0.0) * (1 - 0.6 * nov)
        # |t| >= 3 at a tradable horizon: noise factors in the drills peaked at |t| ~ 2 per horizon
        status = 'active' if out.get('causal') and t >= 3.0 else ('rejected_noncausal' if not out.get('causal') else 'weak')
        self.L.factor_report(fid, out, score, status)
        from .rsi import record_primitive_outcome
        record_primitive_outcome(self.L, code, self.agent.name, 'factor', ref_id=fid, success=(status == 'active'), metric_val=score)
        self.org.export_library()
        self.L.event(self.agent.name, 'FACTOR', {'id': fid, 'name': name, 'score': round(score, 2), 'status': status})
        brief = {k: out.get(k) for k in ('best_horizon_s', 'best_t', 'best_cost_coverage', 'best_maker_cost_coverage',
                                         'economic_value', 'causal', 'causality_issue',
                                         'novelty', 'autocorr_1', 'coverage', 'benchmark_days')}
        brief['horizons'] = {h: {k: v.get(k) for k in ('rank_ic', 'daily_rank_ic_t', 'pct_days_positive_ic', 'q5_minus_q1_bps',
                                                       'half_spread_vs_cost', 'half_spread_vs_maker_cost', 'by_vol_regime')}
                             for h, v in out.get('horizons', {}).items()}
        return {'id': fid, 'status': status, 'score': round(score, 2), 'report': brief}

    async def _register_panel_factor(self, name, code, description, desk):
        if 'def factor' not in code:
            return {'error': 'factor module must define factor(panel, params) -> array aligned to panel rows'}
        fid = self.L.add_factor(name, self.agent.name, self.agent.pod, code, desk, description)
        lib = [r['id'] for r in self.L.q("SELECT id FROM factors WHERE resolution=? AND status='active'", (desk,))]
        out = await run_job(self._ticketed({'type': 'factor_eval', 'desk': desk, 'code': code, 'factor_id': fid, 'library_ids': lib,
                             'values_dir': str(self.s.artifacts / 'factor_values')}), self.s, timeout_s=1800,
                            slot=self.org.job_slots)
        if 'error' in out:
            self.L.factor_report(fid, out, 0.0, 'failed')
            return {'id': fid, 'status': 'failed', 'error': out['error'], 'trace': out.get('trace', '')[-1500:]}
        nov = out.get('novelty', {}).get('max_abs_corr', 0.0) or 0.0
        score = (out.get('economic_value') or 0.0) * (1 - 0.6 * nov)
        status = ('rejected_noncausal' if not out.get('causal') else
                  'active' if abs(out.get('best_t') or 0) >= 3.0 else 'weak')
        self.L.factor_report(fid, out, score, status)
        from .rsi import record_primitive_outcome
        record_primitive_outcome(self.L, code, self.agent.name, 'factor', ref_id=fid, success=(status == 'active'), metric_val=score)
        self.L.event(self.agent.name, 'FACTOR', {'id': fid, 'name': name, 'desk': desk, 'score': round(score, 2), 'status': status})
        return {'id': fid, 'status': status, 'score': round(score, 2), 'report': out,
                'note': 'active needs |IC t| >= 3 at a horizon where the MEAN quintile spread (also without the 5 best '
                        'periods) agrees with the IC sign. spread_vs_cost >= 1 means the spread clears a round trip.'}

    async def t_evaluate_strategy(self, code, all=False, robustness=False):
        try:
            load_strategy(code)
        except Exception as e:
            return {'error': f'strategy does not load: {type(e).__name__}: {e}'}
        job = {'type': 'strategy_eval', 'code': code, 'split': 'dev', 'allowed': ['dev'],
               'max_days': None if all else 60, 'robustness': bool(robustness), 'cost': self.org.cost_params()}
        if 'RESOLUTION' in code and re.search(r"RESOLUTION\s*=\s*['\"]1s", code) and not all:
            job['max_days'] = 15
        out = await run_job(self._ticketed(job), self.s, timeout_s=3600, slot=self.org.job_slots)
        if 'error' in out:
            return out
        sr = out['summary'].get('sharpe')
        from .rsi import record_primitive_outcome
        record_primitive_outcome(self.L, code, self.agent.name, 'strategy', ref_id=out['code_hash'], success=True, metric_val=sr)
        self.L.trial(self.agent.name, 'dev_backtest', out['code_hash'], sr, out['summary'].get('days'),
                     desk=out.get('desk') or strategy_desk(code), split='dev', family='unassigned',
                     identity=_host_identity(code, job, self.s.store))
        self.L.event(self.agent.name, 'DEV_BACKTEST', {'hash': out['code_hash'], 'sharpe': sr,
                                                       'days': out['summary'].get('days'), 'pnl': out['summary'].get('pnl')})
        return out

    async def t_combine_factors(self, h_minutes, factor_ids=None, maker=True, edge_bps=None, hold_minutes=None,
                                model='lgbm', use_base=True, exit='maker'):
        from ..lib.combo import make_strategy
        codes, used = [], []
        for fid in (factor_ids or [])[:12]:
            r = self.L.q("SELECT id, code, resolution, status FROM factors WHERE id=?", (fid,))
            if not r:
                return {'error': f'unknown factor {fid}'}
            if r[0]['resolution'] != '1m':
                return {'error': f'{fid} is {r[0]["resolution"]}; combine_factors uses 1m factors'}
            codes.append(r[0]['code']); used.append(fid)
        code = make_strategy(codes, h=int(h_minutes), edge_bps=edge_bps, hold=hold_minutes, model=model,
                             use_base=use_base, maker=bool(maker), exit=exit)
        out = await self.t_evaluate_strategy(code, all=True, robustness=False)
        if 'error' in out:
            return out
        return {'factors': used, 'summary': out.get('summary'), 'stats': out.get('stats'),
                'maker_stats': out.get('maker_stats'), 'causality': out.get('causality'), 'code': code,
                'note': 'walk-forward on dev (models fit on strictly earlier days). Submit the code unchanged with '
                        'submit_strategy if it is strong and robust; tweak PARAMS only with a reason (every eval is a trial).'}

    async def t_submit_strategy(self, name, code, thesis):
        try:
            mod = load_strategy(code)
        except Exception as e:
            return {'error': f'strategy does not load: {type(e).__name__}: {e}'}
        h = code_hash(code)
        dup = self.L.q('SELECT id, status FROM strategies WHERE code_hash=?', (h,))
        if dup:
            return {'error': f'identical code already submitted as {dup[0]["id"]} ({dup[0]["status"]})'}
        sid = self.L.add_strategy(name, self.agent.name, self.agent.pod, code, h, mod.RESOLUTION, thesis)
        self.org.pipeline.enqueue(sid)
        self.L.event(self.agent.name, 'SUBMITTED', {'id': sid, 'name': name})
        return {'id': sid, 'status': 'submitted', 'next': 'full-dev evaluation + robustness, then officer review. '
                'You will be messaged with the outcome.'}

    # ------------------------------------------------------------ officers
    async def t_review_queue(self):
        return self.L.q("SELECT id, name, author, pod, resolution, updated FROM strategies WHERE status='review' ORDER BY updated")

    async def t_dossier(self, id):
        s = self.L.strategy(id)
        if not s:
            return {'error': 'not found'}
        return {k: s.get(k) for k in ('id', 'name', 'author', 'pod', 'thesis', 'code', 'resolution', 'status', 'dev_report', 'review', 'gate_reasons')}

    async def t_verdict(self, id, decision, rationale):
        return await self.org.pipeline.verdict(id, self.agent.name, decision, rationale)

    # ------------------------------------------------------------ cio
    async def t_org_stats(self):
        return self.org.org_stats()

    async def t_set_pod_weights(self, weights, rationale):
        for pod, w in weights.items():
            self.L.x('UPDATE pods SET weight=?, updated=? WHERE name=?', (max(0.1, min(5.0, float(w))), time.time(), pod))
        self.L.event(self.agent.name, 'POD_WEIGHTS', {'weights': weights, 'rationale': rationale})
        return {'ok': True, 'pods': self.L.q('SELECT name, weight FROM pods')}

    async def t_post_memo(self, body):
        self.L.memo(self.agent.name, 'desk', body)
        return {'ok': True}

    async def t_reassign_agent(self, agent, pod, reason):
        from .roles import PODS
        if pod not in PODS:
            return {'error': f'unknown pod; choose from {list(PODS)}'}
        r = self.L.q('SELECT role FROM agents WHERE name=?', (agent,))
        if not r or r[0]['role'] not in ('researcher',):
            return {'error': 'only researchers can be reassigned'}
        self.L.x('UPDATE agents SET pod=? WHERE name=?', (pod, agent))
        self.L.send(agent, self.agent.name, f'You are reassigned to pod {pod}: {reason}')
        self.L.event(self.agent.name, 'REASSIGN', {'agent': agent, 'pod': pod, 'reason': reason})
        return {'ok': True}

    # ------------------------------------------------------------ pm
    async def t_approved_strategies(self):
        return self.org.pipeline.approved_summary()

    async def t_build_portfolio(self, ids, rationale):
        from ..pipeline.portfolio import build_portfolio
        return build_portfolio(self.s, ids=ids, author=self.agent.name, rationale=rationale, ledger=self.L)

    async def t_paper_status(self):
        from ..live.monitor import paper_status
        return paper_status(self.s)

    async def t_paper_start(self, target):
        from ..live.monitor import start_paper
        sid = str(target).split('/')[-1]
        row = self.L.q('SELECT code FROM strategies WHERE id=?', (sid,))
        if row:
            try:
                desk = load_strategy(row[0]['code']).DESK
            except Exception:
                desk = None
            if desk in ('perps', 'macro'):
                rep = (self.L.strategy(sid) or {}).get('forward_report') or {}
                return {'status': 'forward_holdout_is_the_paper_test',
                        'note': f'{desk} strategies are paper-traded by the rolling FORWARD holdout: every 6 h they are '
                                'rescored on all new verified live days nobody has seen (verdict after 60 days). Live '
                                'order execution for this desk is a later, human-approved step.',
                        'forward_report': {k: rep.get(k) for k in ('days', 'verdict', 'summary')}}
        return start_paper(self.s, target)

    # ------------------------------------------------------------ execution
    async def t_venue_stats(self):
        from ..live.recorder import venue_stats
        return venue_stats(self.s)

    async def t_set_cost_model(self, rationale, impact_mult=None, slippage_bps=None, latency_s=None):
        cur = self.org.cost_params()
        if impact_mult is not None:
            cur['impact_mult'] = max(0.5, min(5.0, float(impact_mult)))
        if slippage_bps is not None:
            cur['slippage_bps'] = max(0.0, min(5.0, float(slippage_bps)))
        if latency_s is not None:
            cur['latency_s'] = max(1, min(10, int(latency_s)))
        self.L.x('INSERT INTO cost_models(author,params,rationale,created) VALUES(?,?,?,?)',
                 (self.agent.name, json.dumps(cur), rationale, time.time()))
        self.L.send('all', self.agent.name, f'Cost model updated: {cur}. Rationale: {rationale[:400]}')
        return {'cost_model': cur}

    # ------------------------------------------------------------ librarian / toolsmith
    async def t_retire_lesson(self, id, reason):
        self.L.x("UPDATE lessons SET status='retired', evidence=evidence || ? WHERE id=?", (f' [retired: {reason}]', id))
        return {'ok': True}

    async def t_write_digest(self, body):
        self.L.memo(self.agent.name, 'digest', body)
        return {'ok': True}

    async def t_contribute_primitive(self, name, code, test, doc):
        if not re.fullmatch(r'[a-z][a-z0-9_]{2,40}', name):
            return {'error': 'name must be a lowercase identifier'}
        contrib = Path(__file__).resolve().parent.parent / 'lib' / 'contrib'
        contrib.mkdir(parents=True, exist_ok=True)
        (contrib / '__init__.py').touch()
        staging = self.agent.workspace / f'contrib_{name}.py'
        staging.write_text(code, encoding='utf-8')
        res = await run_python(f'import sys; sys.path.insert(0, r"{self.agent.workspace}")\n'
                               f'from contrib_{name} import *\n' + test, self.agent.workspace, self.s, timeout_s=300,
                               slot=self.org.sandbox_slots)
        if res['status'] != 'ok' or 'OK' not in res['stdout']:
            return {'error': 'test failed (must exit cleanly and print OK)', 'result': res}
        fut = ''.join(l + '\n' for l in code.splitlines() if l.startswith('from __future__ import'))
        body = '\n'.join(l for l in code.splitlines() if not l.startswith('from __future__ import'))
        target = contrib / f'{name}.py'
        target.write_text(f'"""{doc}"""\n{fut}import numpy as np\nimport polars as pl\n' + body, encoding='utf-8')
        try:                                     # the INSTALLED file must compile, not just the staged one
            compile(target.read_text(encoding='utf-8'), str(target), 'exec')
        except SyntaxError as e:
            target.unlink(missing_ok=True)
            return {'error': f'installed module does not compile: {e}'}
        idx = contrib / 'INDEX.md'
        with open(idx, 'a', encoding='utf-8') as f:
            f.write(f'- `from desk.lib.contrib.{name} import ...` - {doc[:300]} (by {self.agent.name})\n')
        self.L.send('all', self.agent.name, f'New primitive desk.lib.contrib.{name}: {doc[:400]}')
        return {'ok': True, 'module': f'desk.lib.contrib.{name}'}

    # ------------------------------------------------------------ RSI handlers
    async def t_evolve_population(self, desk=None, max_children=3):
        from .rsi import evolve_population
        res = evolve_population(self.L, settings=self.s, desk=desk, max_children=min(int(max_children or 3), 6), author=self.agent.name)
        if res.get('evolved_count'):
            self._streak = 0
        return res

    async def t_compile_hints(self):
        from .rsi import compile_error_hints
        return compile_error_hints(self.L, author=self.agent.name)

    async def t_primitive_lift(self):
        from .rsi import primitive_lift_leaderboard
        return primitive_lift_leaderboard(self.L)

    async def t_propose_charter_patch(self, role='researcher', pod=None):
        from .rsi import propose_charter_patch, evaluate_rsi_drill
        cand = propose_charter_patch(self.L, role=role or 'researcher', pod=pod)
        drill = evaluate_rsi_drill(self.L, settings=self.s, charter_id=cand['id'])
        return {'charter': cand, 'drill_gate': drill}

    async def t_evolve_primitives(self):
        from .rsi import evolve_primitives
        return evolve_primitives(self.L, settings=self.s, author=self.agent.name)

    async def t_run_rsi_campaign(self, generations=2, desks=None, use_llm=True):
        from .rsi import run_rsi_campaign
        return run_rsi_campaign(self.L, settings=self.s, generations=int(generations or 2), desks=desks, use_llm=bool(use_llm))

    async def t_validate_and_promote(self, desk=None, max_candidates=6):
        from .rsi import validate_and_promote_population
        return validate_and_promote_population(self.L, settings=self.s, desk=desk, max_candidates=int(max_candidates or 6), author=self.agent.name)

    async def t_operator_bandit(self, desk=None):
        from .rsi import operator_bandit_leaderboard
        return operator_bandit_leaderboard(self.L, desk=desk)

