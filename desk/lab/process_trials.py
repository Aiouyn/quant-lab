"""Process trials: the only way a change to how the firm works goes live.

A "process variant" is a short note appended to one role's charter (optionally for one pod): a working rule such as
"before submitting, check the signal on each half of the dev period". Agents may propose variants; none of them
changes the firm until a trial shows it helps on markets where the right answer is known.

A variant with role 'primitive' is a research tool instead: a tested desk.lib.contrib module (pod = module name). It is
installed when contributed but announced to agents (session brief) only in arms and firms where it is live, so the
same paired trial decides whether knowing about the tool makes the firm better.

A trial is a paired A/B experiment on drill markets (desk/lab/panel_drills.py):

  for each pair i:
    - one planted market (a hidden, tradable signal) is built with seed s_i and copied for both arms;
    - BASE runs the firm with the charters currently live; CAND runs it with the candidate variant added
      (replacing a live variant for the same role and pod, if any);
    - both runs are scored by how well their approved strategies track the hidden signal; if nothing was approved,
      half the |corr| of the best strategy they built, or a quarter of the best registered factor's (each 0 below
      0.1), so a short drill can still show which arm got closer;
  plus one null market per pair (returns block-shuffled, nothing to find), run by CAND only.

Pre-registered decision rule (decide()):
  ACCEPT  if the candidate approved nothing on any null market, AND its planted score beats BASE on a majority of
          pairs, AND the mean paired improvement is at least `min_lift`.
  REJECT  otherwise. ERROR if any run failed (an incomplete trial never promotes anything).

On ACCEPT the variant becomes 'active' and is appended to that role's charter in live runs; a live variant it
replaces becomes 'superseded'. The decision, every per-run score and the seeds are stored in the main ledger.

This file is part of the sealed kernel: agents cannot change how trials are scored.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import math
import time
import uuid

import numpy as np

MAX_PATCH_CHARS = 1500
MAX_OPEN_PER_AUTHOR = 3
DEFAULT_PAIRS = 3
DEFAULT_HOURS = 2.0
DEFAULT_AGENTS = 8
MIN_LIFT = 0.05            # mean paired gain in |corr| between approved strategies and the hidden signal
TIE = 0.02                 # paired differences smaller than this count as neither a win nor a loss
PARTIAL_CREDIT = 0.5       # weight of a found-but-not-approved signal in a planted run's score
FACTOR_CREDIT = 0.25       # weight of a registered factor (no strategy) that tracks the hidden signal
PARTIAL_MIN_CORR = 0.1     # below this |corr| a strategy is treated as not tracking the hidden signal (noise)
ARM_STATUS = 'arm'         # rows written by the trial runner into a trial run's own (fresh) ledger


def _new_id(prefix, n=6):
    return f'{prefix}-{uuid.uuid4().hex[:n]}'


# ---------------------------------------------------------------------------------------------- variants
def propose(L, role: str, patch: str, rationale: str, author: str, pod: str | None = None) -> dict:
    from ..company.roles import ROLES, PODS
    patch, rationale = (patch or '').strip(), (rationale or '').strip()
    if role not in ROLES:
        return {'error': f'unknown role {role!r}; one of {sorted(ROLES)}'}
    if pod and pod not in PODS:
        return {'error': f'unknown pod {pod!r}; one of {sorted(PODS)}'}
    if not patch or len(patch) > MAX_PATCH_CHARS:
        return {'error': f'patch must be 1..{MAX_PATCH_CHARS} characters'}
    if not rationale:
        return {'error': 'rationale required: which drill or ledger evidence motivates the change?'}
    open_n = L.q("SELECT COUNT(*) n FROM process_variants WHERE author=? AND status IN ('proposed','testing')",
                 (author,))[0]['n']
    if open_n >= MAX_OPEN_PER_AUTHOR:
        return {'error': f'you already have {open_n} untested proposals; wait for their trials'}
    vid = _new_id('PV')
    L.x('INSERT INTO process_variants(id,role,pod,patch_text,rationale,author,status,created) VALUES(?,?,?,?,?,?,?,?)',
        (vid, role, pod or '', patch, rationale, author, 'proposed', time.time()))
    return {'id': vid, 'status': 'proposed',
            'note': 'It changes nothing until a drill trial accepts it (python -m desk rsi trial).'}


PRIMITIVE = 'primitive'      # a variant of this role is a research tool (desk.lib.contrib module), not a charter rule


def propose_primitive(L, name: str, doc: str, rationale: str, author: str) -> dict:
    """A new research tool goes into the firm's shared toolbox only when a drill trial shows firms that know about it
    find planted signals better (and nothing on null markets). Until then it is installed but not announced."""
    doc, rationale = (doc or '').strip(), (rationale or '').strip() or (doc or '').strip()
    if not doc:
        return {'error': 'doc required'}
    open_n = L.q("SELECT COUNT(*) n FROM process_variants WHERE author=? AND status IN ('proposed','testing')",
                 (author,))[0]['n']
    if open_n >= MAX_OPEN_PER_AUTHOR:
        return {'error': f'you already have {open_n} untested proposals; wait for their trials'}
    patch = f'`from desk.lib.contrib.{name} import ...` - {doc[:MAX_PATCH_CHARS - 80]}'
    vid = _new_id('PV')
    L.x('INSERT INTO process_variants(id,role,pod,patch_text,rationale,author,status,created) VALUES(?,?,?,?,?,?,?,?)',
        (vid, PRIMITIVE, name, patch, rationale, author, 'proposed', time.time()))
    return {'id': vid, 'status': 'proposed'}


def primitives_text(L) -> str:
    rows = [r for r in live_variants(L) if r['role'] == PRIMITIVE]
    if not rows:
        return ''
    return ('== TRIAL-PROVEN PRIMITIVES (each adopted only after a drill trial showed it helps) ==\n'
            + '\n'.join(f'- {r["patch_text"]}' for r in rows))


def live_variants(L) -> list[dict]:
    """Variants that shape live charters: accepted by a trial in this ledger, or placed here by a trial runner."""
    return L.q("SELECT * FROM process_variants WHERE status=? OR (status='active' AND id IN "
               "(SELECT variant_id FROM process_trials WHERE verdict='ACCEPT')) ORDER BY created", (ARM_STATUS,))


def overlay_text(L, role: str, pod: str | None = None) -> str:
    rows = [r for r in live_variants(L) if r['role'] == role and (not r['pod'] or r['pod'] == (pod or ''))]
    if not rows:
        return ''
    return 'PROCESS NOTES (each adopted only after a drill trial showed it helps):\n' + \
        '\n'.join(f'- {r["patch_text"]}' for r in rows)


def _arm_rows(live: list[dict], candidate: dict | None) -> list[dict]:
    rows = [r for r in live if r['status'] != ARM_STATUS]
    if candidate:
        rows = [r for r in rows if (r['role'], r['pod']) != (candidate['role'], candidate['pod'])] + [candidate]
    return rows


def _seed_arm(arm_ledger, rows):
    for r in rows:
        arm_ledger.x('INSERT OR REPLACE INTO process_variants(id,role,pod,patch_text,rationale,author,status,created) '
                     'VALUES(?,?,?,?,?,?,?,?)', (r['id'], r['role'], r['pod'], r['patch_text'], r['rationale'],
                                                 r['author'], ARM_STATUS, r['created']))


# ---------------------------------------------------------------------------------------------- decision
def decide(pairs: list[dict], null_false_discoveries: list[int], min_lift: float = MIN_LIFT) -> dict:
    """pairs: [{'base': score|None, 'cand': score|None}], scores in [0, 1]; None means the run failed."""
    if not pairs or any(p.get('base') is None or p.get('cand') is None for p in pairs) \
            or any(n is None for n in null_false_discoveries) or not null_false_discoveries:
        return {'verdict': 'ERROR', 'reason': 'a run failed or was not scored; nothing is promoted'}
    d = np.array([p['cand'] - p['base'] for p in pairs], float)
    wins, losses = int(np.sum(d > TIE)), int(np.sum(d < -TIE))
    out = {'mean_lift': round(float(d.mean()), 4), 'wins': wins, 'losses': losses, 'pairs': len(pairs),
           'null_false_discoveries': int(sum(null_false_discoveries)), 'min_lift': min_lift}
    need = len(pairs) // 2 + 1
    if out['null_false_discoveries'] > 0:
        return {**out, 'verdict': 'REJECT', 'reason': 'the candidate approved strategies on a market with nothing to find'}
    if wins < need or wins <= losses:
        return {**out, 'verdict': 'REJECT', 'reason': f'won {wins} of {len(pairs)} pairs (needs {need})'}
    if d.mean() < min_lift:
        return {**out, 'verdict': 'REJECT', 'reason': f'mean lift {d.mean():.3f} < {min_lift}'}
    return {**out, 'verdict': 'ACCEPT', 'reason': f'won {wins}/{len(pairs)} pairs, mean lift {d.mean():.3f}, '
                                                  'no false discoveries on null markets'}


def futile(pairs_done: list[dict], nulls: list, planned: int, min_lift: float = MIN_LIFT) -> str | None:
    """A reason to stop a trial early, or None. Only returns a reason when NO result of the remaining pairs could
    make decide() ACCEPT (scores are in [0, 1], so a remaining pair adds at most +1 lift), so stopping early never
    changes a verdict, it only saves time. A failed run is not futility: the trial finishes and ends in ERROR."""
    if any(p.get('base') is None or p.get('cand') is None for p in pairs_done) or any(n is None for n in nulls):
        return None
    if sum(nulls) > 0:
        return 'the candidate approved strategies on a market with nothing to find'
    rem = planned - len(pairs_done)
    d = [p['cand'] - p['base'] for p in pairs_done]
    wins, losses = sum(x > TIE for x in d), sum(x < -TIE for x in d)
    need = planned // 2 + 1
    if wins + rem < need or wins + rem <= losses:
        return f'won {wins} of {len(d)} pairs; cannot reach {need} of {planned}'
    if (sum(d) + rem * 1.0) / planned < min_lift:
        return f'mean lift cannot reach {min_lift}'
    return None


# ---------------------------------------------------------------------------------------------- runner
def _seed(trial_id, i, kind):
    return int(hashlib.sha256(f'{trial_id}:{i}:{kind}'.encode()).hexdigest()[:8], 16)


def _default_runner(settings, drill, hours, agents):
    from ..company.runtime import run_company
    return run_company(settings, agents=agents, hours=hours, drill=drill)


def _score(report, kind):
    if kind == 'null':
        fd = report.get('false_discoveries')
        return int(fd) if isinstance(fd, (int, float)) else None
    c = report.get('planted_signal_corr')
    if not (isinstance(c, (int, float)) and math.isfinite(c)):
        return None
    if c > 0:
        return float(c)
    # Nothing approved: half credit for the best strategy the firm built that tracks the hidden signal at all. Without
    # it a firm that never gets a planted signal through validation in a short drill scores 0 in both arms and no
    # change can ever show a lift. Approval still earns twice as much, and the null-market veto still counts approvals.
    def best(key):
        b = max((x[0] for x in report.get(key) or [] if isinstance(x[0], (int, float))), default=0.0)
        return b if b >= PARTIAL_MIN_CORR else 0.0
    return round(max(PARTIAL_CREDIT * best('best_corr_with_hidden'),
                     FACTOR_CREDIT * best('best_factor_corr_with_hidden')), 4)


def run_trial(settings, variant_id: str, desk: str = 'perps', pairs: int = DEFAULT_PAIRS,
              hours: float = DEFAULT_HOURS, agents: int = DEFAULT_AGENTS, min_lift: float = MIN_LIFT,
              runner=None, parallel: bool = True) -> dict:
    """Run a full paired trial for one proposed variant and record the decision in the main ledger.

    parallel: the three runs of a pair (BASE planted, CAND planted, CAND null) run at the same time, so both arms face
    the same conditions (plan limits, machine load) and a trial takes `pairs * hours` instead of three times that.
    The trial stops early only when no outcome of the remaining pairs could still lead to ACCEPT."""
    from ..company.ledger import Ledger
    from .panel_drills import SPECS, build_panel_drill, drill_name, panel_drill_report
    if desk not in SPECS:
        raise SystemExit(f'process trials run on panel drills: {sorted(SPECS)}')
    runner = runner or _default_runner
    L = Ledger(settings.ledger)
    row = L.q('SELECT * FROM process_variants WHERE id=?', (variant_id,))
    if not row:
        raise SystemExit(f'no process variant {variant_id}')
    cand = row[0]
    if cand['status'] not in ('proposed', 'testing'):
        raise SystemExit(f'{variant_id} is {cand["status"]}; only proposed variants are trialled')
    live = live_variants(L)
    tid = _new_id('PT', 10)                                  # 6 hex collided after 10k rows (11-06)
    L.x('UPDATE process_variants SET status=? WHERE id=?', ('testing', variant_id))
    L.x('INSERT INTO process_trials(id,variant_id,desk,pairs,hours,agents,results,verdict,created) VALUES(?,?,?,?,?,?,?,?,?)',
        (tid, variant_id, desk, pairs, hours, agents, '{}', 'RUNNING', time.time()))
    root = settings.home / 'trials' / tid
    arms = {'base': _arm_rows(live, None), 'cand': _arm_rows(live, cand)}
    runs, pair_scores, nulls = [], [], []

    def one(arm, i, kind):
        home = root / f'{arm}-{i}'
        s = dataclasses.replace(settings, home=home, store_dir=None)
        name = drill_name(desk, kind)
        rec = {'arm': arm, 'pair': i, 'kind': kind, 'seed': _seed(tid, i, kind)}
        try:
            build_panel_drill(settings, desk, kind, seed=rec['seed'], drills_root=home / 'drills')
            _seed_arm(Ledger(home / 'drills' / name / 'desk.sqlite3'), arms[arm])
            runner(s, name, hours, agents)
            rep = panel_drill_report(s, desk, kind)
            prune_run(home / 'drills' / name)               # scored: drop the bulky market copy, keep ledger + secret
            rec.update({'score': _score(rep, kind), 'verdict': rep.get('verdict'), 'approved': rep.get('approved'),
                        'strategies': rep.get('strategies'), 'best_corr': (rep.get('best_corr_with_hidden') or [])[:3],
                        'best_factor_corr': (rep.get('best_factor_corr_with_hidden') or [])[:3]})
        except (Exception, SystemExit) as e:                 # a failed run scores None: the trial ends in ERROR
            rec.update({'score': None, 'error': repr(e)[:300]})
        return rec

    jobs = [('base', 'planted'), ('cand', 'planted'), ('cand', 'null')]
    stopped = None
    for i in range(pairs):
        if parallel:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=len(jobs)) as ex:
                recs = list(ex.map(lambda j: one(j[0], i, j[1]), jobs))
        else:
            recs = [one(arm, i, kind) for arm, kind in jobs]
        runs.extend(recs)
        L.x('UPDATE process_trials SET results=? WHERE id=?', (json.dumps({'runs': runs}), tid))
        pair_scores.append({'base': recs[0]['score'], 'cand': recs[1]['score']})
        nulls.append(recs[2]['score'])
        stopped = futile(pair_scores, nulls, pairs, min_lift)
        if stopped and i < pairs - 1:
            break
    decision = decide(pair_scores, nulls, min_lift) if not stopped else \
        {**decide(pair_scores, nulls, min_lift), 'verdict': 'REJECT', 'reason': f'stopped early: {stopped}',
         'pairs_planned': pairs}
    L.x('UPDATE process_trials SET results=?, verdict=?, finished=? WHERE id=?',
        (json.dumps({'runs': runs, 'pairs': pair_scores, 'decision': decision}), decision['verdict'], time.time(), tid))
    if decision['verdict'] == 'ACCEPT':
        L.x("UPDATE process_variants SET status='superseded', decided=? WHERE status='active' AND role=? AND pod=?",
            (time.time(), cand['role'], cand['pod']))
        L.x("UPDATE process_variants SET status='active', decided=? WHERE id=?", (time.time(), variant_id))
    elif decision['verdict'] == 'REJECT':
        L.x("UPDATE process_variants SET status='rejected', decided=? WHERE id=?", (time.time(), variant_id))
    else:
        L.x("UPDATE process_variants SET status='proposed' WHERE id=?", (variant_id,))   # retry later
    L.event('process-trials', 'PROCESS_TRIAL', {'trial': tid, 'variant': variant_id, **decision})
    return {'trial': tid, 'variant': variant_id, 'role': cand['role'], 'pod': cand['pod'], **decision,
            'pair_scores': pair_scores, 'null_false_discoveries_per_pair': nulls}


BULKY = ('store', 'artifacts', 'workspaces')


def prune_run(drill_home) -> int:
    """Delete a scored drill run's market copy, cached factor values and agent workspaces (several GB per trial).
    The drill ledger and drill_secret.json stay, so every score can still be audited."""
    import shutil
    from pathlib import Path
    freed, home = 0, Path(drill_home).resolve()
    for sub in BULKY:
        d = Path(drill_home) / sub
        if d.is_symlink() or getattr(d, 'is_junction', lambda: False)():
            continue                                # a link to the real store: never follow it
        if d.is_dir() and d.resolve().parent == home:
            freed += sum(f.stat().st_size for f in d.rglob('*') if f.is_file())
            shutil.rmtree(d, ignore_errors=True)
    return freed


def prune_finished(settings, L) -> dict:
    """Prune every drill run of trials that are no longer RUNNING."""
    root = settings.home / 'trials'
    done = {r['id'] for r in L.q("SELECT id FROM process_trials WHERE verdict != 'RUNNING'")}
    freed, runs = 0, 0
    for tdir in (root.iterdir() if root.is_dir() else []):
        if tdir.name not in done:
            continue
        for drill_home in tdir.glob('*/drills/*'):
            freed += prune_run(drill_home)
            runs += 1
    return {'runs': runs, 'freed_gb': round(freed / 1e9, 2)}


def recover_stale(L) -> list[str]:
    """Trials still marked RUNNING when the trial loop starts were killed (reboot, crash): mark them ABORTED and put
    their variants back in the queue; variants left in 'testing' with no running trial are re-queued too. Only call
    this from the single trial loop."""
    out = []
    for r in L.q("SELECT id, variant_id FROM process_trials WHERE verdict='RUNNING'"):
        L.x("UPDATE process_trials SET verdict='ABORTED', finished=? WHERE id=?", (time.time(), r['id']))
        L.x("UPDATE process_variants SET status='proposed' WHERE id=? AND status='testing'", (r['variant_id'],))
        out.append(r['id'])
    L.x("UPDATE process_variants SET status='proposed' WHERE status='testing' AND id NOT IN "
        "(SELECT variant_id FROM process_trials WHERE verdict='RUNNING')")
    return out


def store_problem(settings, desk='perps') -> str | None:
    """Read the files every drill copies from the main store. A failing disk (D: on 2026-11-06) made each trial
    ERROR instantly; the loop must wait for the store instead of burning proposals."""
    if os.path.lexists(settings.store) and not settings.store.exists():
        return f'{settings.store} is a link to a missing drive'
    for f in (settings.store / 'splits.json', settings.store / desk / 'splits.json', settings.store / desk / 'panel.parquet'):
        try:
            if f.exists():
                with open(f, 'rb') as fh:
                    fh.read(4096)
        except OSError as e:
            return f'{f}: {e!r}'
    return None


def purge_empty_errors(L) -> int:
    """Delete ERROR trials that scored no run at all (no information; e.g. 10k rows from the 11-06 disk failure)."""
    rows = [r['id'] for r in L.q("SELECT id, results FROM process_trials WHERE verdict='ERROR'")
            if not any(run.get('score') is not None for run in (json.loads(r['results'] or '{}') or {}).get('runs', []))]
    for i in range(0, len(rows), 500):
        chunk = rows[i:i + 500]
        L.x(f"DELETE FROM process_trials WHERE id IN ({','.join('?' * len(chunk))})", tuple(chunk))
    return len(rows)


def trial_loop(settings, desk='perps', pairs=DEFAULT_PAIRS, hours=DEFAULT_HOURS, agents=DEFAULT_AGENTS,
               idle_s=600, stop=None, log=print, runner=None, error_backoff_s=900):
    """Run queued proposals one after another, forever (managed by the supervisor). Returns when stop() is true,
    which the supervisor service uses to restart on new code between trials."""
    from ..company.ledger import Ledger
    L = Ledger(settings.ledger)
    for tid in recover_stale(L):
        log(f'trial {tid} was interrupted: marked ABORTED, its proposal re-queued')
    errors = 0
    while not (stop and stop()):
        vid = next_proposed(L)
        if not vid:
            time.sleep(idle_s)
            continue
        bad = store_problem(settings, desk)
        if bad:
            log(f'store unreadable, not starting a trial: {bad}')
            time.sleep(idle_s)
            continue
        log(f'trial start {vid}')
        res = run_trial(settings, vid, desk=desk, pairs=pairs, hours=hours, agents=agents, runner=runner)
        log(f"trial {res['trial']} {vid}: {res['verdict']} - {res.get('reason')}")
        errors = errors + 1 if res['verdict'] == 'ERROR' else 0
        if errors:                                      # never spin: 15 min, 30, 60 ... capped at 6 h
            wait = min(error_backoff_s * 2 ** (errors - 1), 6 * 3600)
            log(f'{errors} trial error(s) in a row: waiting {wait / 60:.0f} min')
            time.sleep(wait)


def next_proposed(L) -> str | None:
    r = L.q("SELECT id FROM process_variants WHERE status='proposed' ORDER BY created LIMIT 1")
    return r[0]['id'] if r else None


def history(L, limit: int = 20) -> list[dict]:
    """What was tried and what the drills said: agents read this before proposing again."""
    out = []
    for r in L.q('SELECT v.id, v.role, v.pod, v.patch_text, v.author, v.status, t.verdict, t.results, '
                 '(SELECT COUNT(*) FROM process_trials x WHERE x.variant_id=v.id) trials '
                 'FROM process_variants v LEFT JOIN process_trials t ON t.id = (SELECT id FROM process_trials y '
                 '  WHERE y.variant_id=v.id ORDER BY y.created DESC, y.rowid DESC LIMIT 1) '
                 'ORDER BY v.created DESC LIMIT ?', (limit,)):
        dec = (json.loads(r.pop('results') or '{}') or {}).get('decision') or {}
        out.append({**r, 'reason': dec.get('reason'), 'mean_lift': dec.get('mean_lift')})
    return out


def retire(L, variant_id: str, reason: str) -> bool:
    n = L.q("SELECT COUNT(*) n FROM process_variants WHERE id=? AND status IN ('active','proposed')", (variant_id,))[0]['n']
    if n:
        L.x("UPDATE process_variants SET status='retired', decided=?, rationale=rationale||? WHERE id=?",
            (time.time(), f' | retired: {reason}', variant_id))
    return bool(n)
