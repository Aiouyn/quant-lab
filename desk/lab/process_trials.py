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
    - both runs are scored by how well their approved strategies track the hidden signal (0 if nothing approved);
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
ARM_STATUS = 'arm'         # rows written by the trial runner into a trial run's own (fresh) ledger


def _new_id(prefix):
    return f'{prefix}-{uuid.uuid4().hex[:6]}'


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
    return float(c) if isinstance(c, (int, float)) and math.isfinite(c) else None


def run_trial(settings, variant_id: str, desk: str = 'perps', pairs: int = DEFAULT_PAIRS,
              hours: float = DEFAULT_HOURS, agents: int = DEFAULT_AGENTS, min_lift: float = MIN_LIFT,
              runner=None) -> dict:
    """Run a full paired trial for one proposed variant and record the decision in the main ledger."""
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
    tid = _new_id('PT')
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
            rec.update({'score': _score(rep, kind), 'verdict': rep.get('verdict'), 'approved': rep.get('approved')})
        except (Exception, SystemExit) as e:                 # a failed run scores None: the trial ends in ERROR
            rec.update({'score': None, 'error': repr(e)[:300]})
        runs.append(rec)
        L.x('UPDATE process_trials SET results=? WHERE id=?', (json.dumps({'runs': runs}), tid))
        return rec['score']

    for i in range(pairs):
        pair_scores.append({'base': one('base', i, 'planted'), 'cand': one('cand', i, 'planted')})
        nulls.append(one('cand', i, 'null'))
    decision = decide(pair_scores, nulls, min_lift)
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


def next_proposed(L) -> str | None:
    r = L.q("SELECT id FROM process_variants WHERE status='proposed' ORDER BY created LIMIT 1")
    return r[0]['id'] if r else None


def history(L, limit: int = 20) -> list[dict]:
    """What was tried and what the drills said: agents read this before proposing again."""
    out = []
    for r in L.q('SELECT v.id, v.role, v.pod, v.patch_text, v.author, v.status, t.verdict, t.results '
                 'FROM process_variants v LEFT JOIN process_trials t ON t.variant_id=v.id '
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
