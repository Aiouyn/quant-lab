"""The organization runtime: roster, pods, compute slots, the promotion pipeline and the yield bandit."""
from __future__ import annotations

import asyncio
import dataclasses
import json
import random
import signal
import time
from pathlib import Path

import numpy as np

from .agent import Agent
from .ledger import Ledger
from .llm import LLM
from .roles import PODS

CORE = [('cio', 'cio', 'exec'), ('officer-1', 'officer', 'risk'), ('officer-2', 'officer', 'risk'),
        ('pm', 'pm', 'portfolio'), ('librarian', 'librarian', 'knowledge'), ('execution', 'execution', 'venue'),
        ('toolsmith', 'toolsmith', 'platform')]


# measured yield first: carry (structural funding edge) > order flow / meta (combination engine) > the rest
POD_PRIORITY = ['macro', 'xsec', 'events', 'regime', 'carry', 'positioning', 'meta', 'pred']   # pred last: Polymarket is blocked for the owner (France)


def roster(n_agents):
    """Core functions first, then pod leads, then researchers round-robin across pods.
    Lean by default: one risk officer (a second joins at >= 24 agents), staff roles only when the firm is big
    enough to feed them."""
    n_agents = max(n_agents, 4)
    if n_agents < 10:
        core = [CORE[0], CORE[1]]
    elif n_agents < 24:
        core = [CORE[0], CORE[1], CORE[3], CORE[4], CORE[6]] + ([CORE[5]] if n_agents >= 18 else [])
    else:
        core = CORE
    out = list(core)
    pods = [p for p in POD_PRIORITY if p in PODS] + [p for p in PODS if p not in POD_PRIORITY]
    remaining = n_agents - len(out)
    leads = pods[:max(1, min(len(pods), remaining // 2 if remaining < 14 else len(pods)))]
    for p in leads:
        if remaining <= 0:
            break
        out.append((f'{p}-lead', 'lead', p)); remaining -= 1
    # researchers: the highest-yield desk gets an extra seat first, then round-robin by priority
    order = ([leads[0]] if leads else []) + leads
    i, counters = 0, {p: 0 for p in leads}
    while remaining > 0:
        p = order[i % len(order)] if i < len(order) else leads[(i - len(order)) % len(leads)]
        counters[p] += 1
        out.append((f'{p}-r{counters[p]}', 'researcher', p)); remaining -= 1; i += 1
    return out


class Org:
    def __init__(self, settings, n_agents=28, hours=8.0, concurrency=None, tokens_per_hour=None):
        self.settings = settings.ensure()
        self.tokens_per_hour = tokens_per_hour
        self._budget_checked, self._budget_ok = 0.0, True
        self.ledger = Ledger(settings.ledger)
        self.deadline = time.time() + hours * 3600
        self.started = time.time()
        self.llm = LLM(settings, self.ledger, cap=concurrency or settings.max_concurrency,
                       start=min(settings.start_concurrency, concurrency or settings.max_concurrency))
        self.sandbox_slots = asyncio.Semaphore(max(1, settings.compute_workers))
        self.job_slots = asyncio.Semaphore(max(1, settings.heavy_slots))
        from ..pipeline.validation import Pipeline
        self.pipeline = Pipeline(settings, self.ledger, self)
        self.agents = []
        for name, role, pod in roster(n_agents):
            self.ledger.upsert_agent(name, role, pod)
            self.ledger.x('UPDATE agents SET model=? WHERE name=?', (self.model_for_role(role), name))
            self.agents.append(Agent(name, role, pod, settings.workspaces / name))
        for p, mandate in PODS.items():
            self.ledger.x('INSERT OR IGNORE INTO pods VALUES(?,?,?,?)', (p, mandate, 1.0, time.time()))
        self._stop = asyncio.Event()

    # ------------------------------------------------------------------ helpers
    def model_for_role(self, role):
        return self.settings.flash_model if role in self.settings.flash_roles else self.settings.model

    def should_stop(self):
        return self._stop.is_set() or time.time() > self.deadline or self.ledger.stop_requested()

    async def sleep(self, seconds, wake=None):
        end = time.time() + seconds
        while time.time() < end and not self.should_stop():
            await asyncio.sleep(min(5, max(0.1, end - time.time())))
            if wake is not None:
                try:
                    if wake():
                        return
                except Exception:
                    pass

    async def budget_gate(self):
        """Optional hard cap on tokens per rolling hour: callers wait while the firm is over budget."""
        if not self.tokens_per_hour:
            return
        while not self.should_stop():
            if time.time() - self._budget_checked > 15:
                used = self.ledger.q('SELECT SUM(prompt_tokens+completion_tokens) t FROM llm_requests WHERE created>?',
                                     (time.time() - 3600,))[0]['t'] or 0
                self._budget_ok = used < self.tokens_per_hour
                self._budget_checked = time.time()
            if self._budget_ok:
                return
            await asyncio.sleep(20)

    def stagger(self, agent):
        order = {'cio': 0, 'lead': 5, 'researcher': 20, 'toolsmith': 40, 'execution': 60, 'librarian': 900, 'officer': 30, 'pm': 60}
        return order.get(agent.role, 30) + random.random() * 30

    def priority(self, agent):
        if agent.role == 'officer':
            return 3.0
        if agent.role in ('cio', 'pm'):
            return 2.0
        w = self.ledger.q('SELECT weight FROM pods WHERE name=?', (agent.pod,))
        return float(w[0]['weight']) if w else 1.0

    def cost_params(self):
        r = self.ledger.q('SELECT params FROM cost_models ORDER BY id DESC LIMIT 1')
        if r:
            return json.loads(r[0]['params'])
        from ..lab.backtest import CostModel
        d = dataclasses.asdict(CostModel())
        d['taker_bps'] = self.settings.venue.taker_bps
        return d

    def export_library(self):
        rows = self.ledger.q("SELECT id, name, author, pod, code, resolution, description, score FROM factors WHERE status='active' ORDER BY score DESC")
        p = self.settings.artifacts / 'library.json'
        p.write_text(json.dumps(rows, default=str))

    # ------------------------------------------------------------------ brief
    def brief(self, agent):
        L = self.ledger
        now = time.time()
        parts = [f'SESSION BRIEF for {agent.name} (role={agent.role}, pod={agent.pod}) - '
                 f'{time.strftime("%Y-%m-%d %H:%M")} - run elapsed {(now - self.started) / 3600:.1f} h, '
                 f'remaining {(self.deadline - now) / 3600:.1f} h.']
        memo = L.latest_memo('desk')
        if memo:
            parts.append(f'== DESK MEMO (from {memo["author"]}, {(now - memo["created"]) / 60:.0f} min ago) ==\n{memo["body"][:2500]}')
        digest = L.latest_memo('digest')
        if digest:
            parts.append(f'== KNOWLEDGE DIGEST ==\n{digest["body"][:2500]}')
        else:
            lessons = L.q("SELECT id, author, claim, confidence FROM lessons WHERE status='active' ORDER BY created DESC LIMIT 8")
            if lessons:
                parts.append('== RECENT PLAYBOOK LESSONS ==\n' + '\n'.join(
                    f'- {x["id"]} ({x["author"]}, conf {x["confidence"]}): {x["claim"][:300]}' for x in lessons))
        if agent.pod in PODS:
            w = L.q('SELECT weight FROM pods WHERE name=?', (agent.pod,))
            items = L.board_list('active', pod=agent.pod, limit=12)
            parts.append(f'== POD {agent.pod} (budget weight {w[0]["weight"] if w else 1}) ==\nMandate: {PODS[agent.pod]}\n'
                         + ('Active board items:\n' + '\n'.join(f'- {b["id"]} [{b["kind"]}/{b["status"]}] {b["title"][:120]} '
                                                               f'(by {b["author"]}, assignee {b["assignee"]})' for b in items)
                            if items else 'No active board items in this pod yet - create the first theses.'))
        facs = L.factors(8)
        if facs:
            lines = []
            for f in facs:
                rep = f['report'] or {}
                lines.append(f'- {f["id"]} {f["name"][:60]} ({f["author"]}, {f["resolution"]}) score {f["score"]:.1f}, '
                             f'best h={rep.get("best_horizon_s")}s t={rep.get("best_t")}, cost_cov={rep.get("best_cost_coverage")}')
            parts.append('== FACTOR LIBRARY (top by score) ==\n' + '\n'.join(lines))
        funnel = L.q('SELECT status, COUNT(*) n FROM strategies GROUP BY status')
        if funnel:
            parts.append('== STRATEGY PIPELINE == ' + ', '.join(f'{r["status"]}: {r["n"]}' for r in funnel))
        if agent.role in ('researcher', 'lead'):
            nf = L.q('SELECT COUNT(*) n FROM factors WHERE author=?', (agent.name,))[0]['n']
            ne = L.q("SELECT COUNT(*) n FROM trials WHERE author=? AND kind='dev_backtest'", (agent.name,))[0]['n']
            ns = L.q('SELECT COUNT(*) n FROM strategies WHERE author=?', (agent.name,))[0]['n']
            sess = L.q('SELECT sessions FROM agents WHERE name=?', (agent.name,))
            sess = sess[0]['sessions'] if sess else 0
            line = f'== YOUR MEASURED OUTPUT == factors registered: {nf}, dev evaluations: {ne}, submissions: {ns} (sessions so far: {sess}).'
            if nf + ne + ns == 0 and sess >= 2:
                line += (' You have produced NO measured artifact yet. THIS SESSION: turn your best finding into a '
                         'register_factor or evaluate_strategy/combine_factors call before anything else.')
            parts.append(line)
        mine = L.q('SELECT id, name, status, gate_reasons FROM strategies WHERE author=? ORDER BY updated DESC LIMIT 5', (agent.name,))
        if mine:
            parts.append('== YOUR SUBMISSIONS ==\n' + '\n'.join(f'- {s["id"]} {s["name"][:60]}: {s["status"]} {s["gate_reasons"] or ""}' for s in mine))
        if agent.role == 'officer':
            q = L.q("SELECT id, name, author FROM strategies WHERE status='review' ORDER BY updated")
            parts.append('== REVIEW QUEUE ==\n' + ('\n'.join(f'- {x["id"]} {x["name"]} by {x["author"]}' for x in q) or 'empty'))
        inbox = L.inbox(agent.name, agent.pod)
        if inbox:
            parts.append('== INBOX ==\n' + '\n'.join(f'- from {m["from_agent"]} (to {m["to_agent"]}): {m["body"][:600]}' for m in inbox))
        nb = agent.workspace / 'notebook.md'
        if nb.exists():
            text = nb.read_text(encoding='utf-8')
            parts.append('== YOUR NOTEBOOK (tail) ==\n' + text[-3000:])
        else:
            parts.append('== YOUR NOTEBOOK == empty - this is your first session. Orient yourself quickly, then act.')
        idx = Path(__file__).resolve().parent.parent / 'lib' / 'contrib' / 'INDEX.md'
        if idx.exists() and agent.role in ('lead', 'researcher', 'toolsmith', 'officer'):
            parts.append('== SHARED PRIMITIVES (desk.lib.contrib) ==\n' + idx.read_text(encoding='utf-8')[-2500:])
        cost = self.cost_params()
        parts.append(f'== COST MODEL == taker {cost["taker_bps"]} bps + slippage {cost["slippage_bps"]} bps per side, '
                     f'impact x{cost["impact_mult"]}, latency {cost["latency_s"]}s, hourly funding on.')
        parts.append('Plan this session: choose the single highest-value objective for the firm given everything above, '
                     'execute it with tools, record what you learned, update your notebook, and end_session.')
        return '\n\n'.join(parts)

    # ------------------------------------------------------------------ CIO analytics
    def org_stats(self):
        L = self.ledger
        out = {'elapsed_h': round((time.time() - self.started) / 3600, 2), 'remaining_h': round((self.deadline - time.time()) / 3600, 2),
               'trials': L.trial_stats(), 'concurrency': L.get('concurrency'), 'pods': {}}
        for p in PODS:
            members = L.q('SELECT name, role, prompt_tokens+completion_tokens t, sessions FROM agents WHERE pod=?', (p,))
            names = [m['name'] for m in members]
            tok = sum(m['t'] or 0 for m in members)
            fac = L.q('SELECT status, COUNT(*) n, MAX(score) best FROM factors WHERE pod=? GROUP BY status', (p,))
            st = L.q('SELECT status, COUNT(*) n FROM strategies WHERE pod=? GROUP BY status', (p,))
            fac_d = {r['status']: r['n'] for r in fac}
            st_d = {r['status']: r['n'] for r in st}
            succ = 6 * st_d.get('approved', 0) + 3 * (st_d.get('holdout', 0) + st_d.get('rejected_holdout', 0) + st_d.get('validating', 0)) \
                + 1.5 * st_d.get('review', 0) + 0.5 * fac_d.get('active', 0)
            fail = st_d.get('rejected_dev', 0) + st_d.get('rejected_val', 0) + 0.3 * fac_d.get('weak', 0) + tok / 2e6
            out['pods'][p] = {'members': names, 'tokens': tok, 'factors': fac_d,
                              'best_factor_score': max([r['best'] or 0 for r in fac] or [0]), 'strategies': st_d,
                              'yield_per_1M_tokens': round(succ / max(tok / 1e6, 0.1), 2),
                              'weight': L.q('SELECT weight FROM pods WHERE name=?', (p,))[0]['weight']}
            out['pods'][p]['_beta'] = (1 + succ, 1 + fail)
        draws = {p: np.random.beta(*v.pop('_beta'), size=4000) for p, v in out['pods'].items()}
        stack = np.vstack(list(draws.values()))
        win = np.bincount(stack.argmax(0), minlength=len(draws)) / stack.shape[1]
        out['bandit_suggested_weights'] = {p: round(0.5 + 2.5 * w, 2) for p, w in zip(draws, win)}
        out['funnel'] = {r['status']: r['n'] for r in L.q('SELECT status, COUNT(*) n FROM strategies GROUP BY status')}
        out['tokens_total'] = L.status()['tokens']
        return out


def run_company(settings, agents=28, hours=8.0, concurrency=None, drill=None, tokens_per_hour=None):
    if drill:
        home = settings.home / 'drills' / drill
        if not (home / 'store' / 'splits.json').exists():
            raise SystemExit(f'drill market not built: python -m desk drill {drill}')
        settings = dataclasses.replace(settings, home=home)
    from ..data.store import FeatureStore
    store = FeatureStore(settings)
    store.build_dev_view()
    return asyncio.run(_main(settings, agents, hours, concurrency, tokens_per_hour))


async def _main(settings, n_agents, hours, concurrency, tokens_per_hour=None):
    org = Org(settings, n_agents, hours, concurrency, tokens_per_hour)
    L = org.ledger
    L.set('stop', False)
    L.set('running', True)
    L.set('run', {'started': org.started, 'deadline': org.deadline, 'agents': len(org.agents), 'home': str(settings.home)})
    org.export_library()
    print(f'Desk OS: {len(org.agents)} agents for {hours} h (concurrency cap {org.llm.throttle.cap}). '
          f'Dashboard: python -m desk serve. Stop: python -m desk stop', flush=True)
    loop = asyncio.get_running_loop()
    try:
        loop.add_signal_handler(signal.SIGINT, org._stop.set)
    except (NotImplementedError, RuntimeError):
        pass
    pipe = asyncio.create_task(org.pipeline.run(workers=3))
    tasks = [asyncio.create_task(a.live(org)) for a in org.agents]

    async def watchdog():
        from ..data.store import FeatureStore
        last_view = time.time()
        last_forward = time.time() - 3 * 3600          # first forward check ~3h into the run, then every 6h
        while not org.should_stop():
            await asyncio.sleep(10)
            if time.time() - last_forward > 6 * 3600:
                last_forward = time.time()
                try:
                    await org.pipeline.forward_check()
                except Exception as e:
                    L.event('runtime', 'FORWARD_ERROR', {'error': repr(e)[:500]})
            if time.time() - last_view > 600:          # newly ingested dev days become visible to sandboxes
                last_view = time.time()
                try:
                    await asyncio.to_thread(FeatureStore(settings).build_dev_view)
                except Exception as e:
                    L.event('runtime', 'DEV_VIEW_ERROR', {'error': repr(e)})
        org._stop.set()
    try:
        await asyncio.gather(watchdog(), *tasks)
    finally:
        # let in-flight pipeline jobs finish their current item briefly, then stop
        pipe.cancel()
        L.set('running', False)
        L.event('runtime', 'RUN_END', L.status()['tokens'])
    print(json.dumps(L.status()['tokens']), flush=True)
    return 0
