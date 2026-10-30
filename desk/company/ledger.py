"""Durable organizational state: board, messages, factor library, strategies, trial ledger, playbook,
memos, LLM accounting, events and control flags. SQLite in WAL mode + FTS5 search."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

SCHEMA = '''
CREATE TABLE IF NOT EXISTS agents(name TEXT PRIMARY KEY, role TEXT, pod TEXT, status TEXT, sessions INT DEFAULT 0,
  steps INT DEFAULT 0, prompt_tokens INT DEFAULT 0, completion_tokens INT DEFAULT 0, last_active REAL, last_summary TEXT);
CREATE TABLE IF NOT EXISTS board(id TEXT PRIMARY KEY, kind TEXT, title TEXT, body TEXT, author TEXT, assignee TEXT,
  pod TEXT, status TEXT, parent TEXT, tags TEXT, created REAL, updated REAL);
CREATE TABLE IF NOT EXISTS comments(id INTEGER PRIMARY KEY AUTOINCREMENT, board_id TEXT, author TEXT, body TEXT, created REAL);
CREATE TABLE IF NOT EXISTS messages(id INTEGER PRIMARY KEY AUTOINCREMENT, to_agent TEXT, from_agent TEXT, body TEXT,
  created REAL, read INT DEFAULT 0);
CREATE TABLE IF NOT EXISTS factors(id TEXT PRIMARY KEY, name TEXT, author TEXT, pod TEXT, code TEXT, resolution TEXT,
  description TEXT, report TEXT, score REAL, status TEXT, created REAL);
CREATE TABLE IF NOT EXISTS strategies(id TEXT PRIMARY KEY, name TEXT, author TEXT, pod TEXT, code TEXT, code_hash TEXT,
  resolution TEXT, thesis TEXT, status TEXT, dev_report TEXT, review TEXT, val_report TEXT, holdout_report TEXT,
  forward_report TEXT, paper_report TEXT, gate_reasons TEXT, created REAL, updated REAL);
CREATE TABLE IF NOT EXISTS trials(id INTEGER PRIMARY KEY AUTOINCREMENT, author TEXT, kind TEXT, code_hash TEXT,
  sharpe REAL, days INT, created REAL);
CREATE TABLE IF NOT EXISTS holdout_uses(code_hash TEXT PRIMARY KEY, strategy TEXT, created REAL, passed INT);
CREATE TABLE IF NOT EXISTS lessons(id TEXT PRIMARY KEY, author TEXT, claim TEXT, evidence TEXT, confidence REAL,
  scope TEXT, tags TEXT, status TEXT DEFAULT 'active', created REAL);
CREATE TABLE IF NOT EXISTS memos(id INTEGER PRIMARY KEY AUTOINCREMENT, author TEXT, kind TEXT, body TEXT, created REAL);
CREATE TABLE IF NOT EXISTS llm_requests(id INTEGER PRIMARY KEY AUTOINCREMENT, agent TEXT, created REAL, seconds REAL,
  prompt_tokens INT, completion_tokens INT, status TEXT);
CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY AUTOINCREMENT, created REAL, agent TEXT, kind TEXT, payload TEXT);
CREATE TABLE IF NOT EXISTS control(key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS pods(name TEXT PRIMARY KEY, mandate TEXT, weight REAL, updated REAL);
CREATE TABLE IF NOT EXISTS cost_models(id INTEGER PRIMARY KEY AUTOINCREMENT, author TEXT, params TEXT, rationale TEXT, created REAL);
CREATE TABLE IF NOT EXISTS portfolios(id TEXT PRIMARY KEY, author TEXT, members TEXT, weights TEXT, report TEXT, status TEXT, created REAL);
CREATE TABLE IF NOT EXISTS dynamic_hints(id TEXT PRIMARY KEY, pattern TEXT, hint TEXT, source_error TEXT, author TEXT,
  hits INT DEFAULT 0, resolved_count INT DEFAULT 0, status TEXT DEFAULT 'active', created REAL);
CREATE TABLE IF NOT EXISTS primitive_usages(id INTEGER PRIMARY KEY AUTOINCREMENT, primitive TEXT, agent TEXT, kind TEXT,
  ref_id TEXT, success INT DEFAULT 1, metric_val REAL, created REAL);
CREATE TABLE IF NOT EXISTS lineages(child_id TEXT PRIMARY KEY, child_kind TEXT, parent_ids TEXT, operator TEXT, desk TEXT,
  delta_score REAL, created REAL);
CREATE TABLE IF NOT EXISTS charter_versions(id TEXT PRIMARY KEY, role TEXT, pod TEXT, patch_text TEXT, rationale TEXT,
  status TEXT DEFAULT 'candidate', drill_planted_score REAL, drill_null_fdr INT DEFAULT 0, error_rate_delta REAL, created REAL);
CREATE TABLE IF NOT EXISTS rsi_runs(id TEXT PRIMARY KEY, kind TEXT, summary TEXT, metrics TEXT, created REAL);
CREATE TABLE IF NOT EXISTS rsi_reasoning(id TEXT PRIMARY KEY, created REAL, agent TEXT, desk TEXT, category TEXT,
  title TEXT, observation TEXT, diagnosis TEXT, hypothesis TEXT, action_taken TEXT, code_snippet TEXT,
  proof_json TEXT, verdict TEXT);
CREATE VIRTUAL TABLE IF NOT EXISTS search USING fts5(kind, ref, title, body, author, tokenize='porter');
CREATE INDEX IF NOT EXISTS ix_events ON events(created);
CREATE INDEX IF NOT EXISTS ix_msg ON messages(to_agent, read);
CREATE INDEX IF NOT EXISTS ix_board ON board(status, pod);
CREATE INDEX IF NOT EXISTS ix_prim ON primitive_usages(primitive, kind);
CREATE INDEX IF NOT EXISTS ix_reas ON rsi_reasoning(created, desk);
'''

# One-shot holdout lock (cycle 1b). Shots are keyed by strategy FAMILY (desk/lab/holdout.py). Both tables, and the
# legacy code_hash-keyed holdout_uses, are insert-only: UPDATE and DELETE abort, and so does any INSERT that would
# collide with an existing family (so INSERT OR REPLACE cannot delete-and-reinsert a used shot).
HOLDOUT_LOCK_SQL = '''
CREATE TABLE IF NOT EXISTS holdout_shots(family TEXT PRIMARY KEY, desk TEXT, strategy TEXT, code_hash TEXT,
  window TEXT, skeleton TEXT, reason TEXT, created REAL);
CREATE TABLE IF NOT EXISTS holdout_results(id INTEGER PRIMARY KEY AUTOINCREMENT, family TEXT, strategy TEXT,
  passed INT, created REAL);
CREATE TRIGGER IF NOT EXISTS holdout_shots_no_dup BEFORE INSERT ON holdout_shots
  WHEN EXISTS (SELECT 1 FROM holdout_shots WHERE family = NEW.family)
  BEGIN SELECT RAISE(ABORT, 'holdout_shots is insert-only: family already used its holdout shot'); END;
'''
# Trial ledger (cycle 1b): every backtest run (dev/val, any variant) appends one row; the deflated Sharpe counts them.
HOLDOUT_LOCK_SQL += '''
CREATE TABLE IF NOT EXISTS eval_trials(id INTEGER PRIMARY KEY AUTOINCREMENT, family TEXT, params_hash TEXT,
  code_hash TEXT, desk TEXT, split TEXT, variant TEXT, sharpe REAL, n_obs INT, periods INT, sr_pp REAL, source TEXT,
  created REAL);
CREATE INDEX IF NOT EXISTS ix_eval_trials_family ON eval_trials(family);
'''
INSERT_ONLY_TABLES = ('holdout_shots', 'holdout_results', 'holdout_uses', 'eval_trials')
for _t in INSERT_ONLY_TABLES:
    HOLDOUT_LOCK_SQL += (f"CREATE TRIGGER IF NOT EXISTS {_t}_no_update BEFORE UPDATE ON {_t} "
                         f"BEGIN SELECT RAISE(ABORT, '{_t} is insert-only'); END;\n"
                         f"CREATE TRIGGER IF NOT EXISTS {_t}_no_delete BEFORE DELETE ON {_t} "
                         f"BEGIN SELECT RAISE(ABORT, '{_t} is insert-only'); END;\n")
HOLDOUT_LOCK_SQL += '''CREATE TRIGGER IF NOT EXISTS holdout_uses_no_dup BEFORE INSERT ON holdout_uses
  WHEN EXISTS (SELECT 1 FROM holdout_uses WHERE code_hash = NEW.code_hash)
  BEGIN SELECT RAISE(ABORT, 'holdout_uses is insert-only'); END;
'''


def _id(prefix):
    return f'{prefix}-{uuid.uuid4().hex[:6]}'


class Ledger:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        with self.db() as db:
            db.executescript(SCHEMA)
            db.executescript(HOLDOUT_LOCK_SQL)
            self._burn_legacy_holdouts(db)
            for table, col in (('llm_requests', 'model TEXT'), ('agents', 'model TEXT')):
                try:
                    db.execute(f'ALTER TABLE {table} ADD COLUMN {col}')
                except sqlite3.OperationalError:
                    pass                                   # column already exists

    @contextmanager
    def db(self):
        con = sqlite3.connect(self.path, timeout=60)
        con.row_factory = sqlite3.Row
        con.execute('PRAGMA journal_mode=WAL')
        con.execute('PRAGMA synchronous=NORMAL')
        con.execute('PRAGMA recursive_triggers=ON')        # REPLACE-deletes fire the insert-only DELETE triggers
        try:
            with self._lock:
                yield con
                con.commit()
        finally:
            con.close()

    # ----------------------------------------------------------------- holdout lock
    @staticmethod
    def _family_rows(code, code_hash_, parent_codes=()):
        from ..lab.holdout import code_desk, family_keys, skeleton_tokens
        try:
            keys = family_keys(code, parent_codes)
            return code_desk(code), keys, ' '.join(skeleton_tokens(code))
        except (SyntaxError, ValueError, TypeError):
            return None, [f'legacy:{code_hash_}'], None

    def _burn_legacy_holdouts(self, db):
        """Every holdout shot taken before the lock (legacy holdout_uses rows, any strategy with a holdout report or a
        holdout verdict) burns its whole family. Idempotent; runs on every open."""
        rows = db.execute(
            "SELECT s.id, s.code, s.code_hash FROM strategies s WHERE s.holdout_report IS NOT NULL "
            "OR s.status IN ('approved','rejected_holdout') "
            "OR s.code_hash IN (SELECT code_hash FROM holdout_uses) OR s.id IN (SELECT strategy FROM holdout_uses)").fetchall()
        seen = set()
        for r in rows:
            desk, keys, sk = self._family_rows(r['code'] or '', r['code_hash'])
            for i, k in enumerate(keys):
                if k in seen or db.execute('SELECT 1 FROM holdout_shots WHERE family=?', (k,)).fetchone():
                    continue
                seen.add(k)
                db.execute('INSERT INTO holdout_shots VALUES(?,?,?,?,?,?,?,?)',
                           (k, desk, r['id'], r['code_hash'], None, sk if i == 0 else None, 'burned_legacy', time.time()))
        for r in db.execute('SELECT code_hash, strategy FROM holdout_uses').fetchall():
            k = f'legacy:{r["code_hash"]}'
            if not db.execute('SELECT 1 FROM holdout_shots WHERE family=?', (k,)).fetchone():
                db.execute('INSERT INTO holdout_shots VALUES(?,?,?,?,?,?,?,?)',
                           (k, None, r['strategy'], r['code_hash'], None, None, 'burned_legacy', time.time()))

    def parent_codes(self, sid, depth=5):
        out, frontier = [], [sid]
        for _ in range(depth):
            nxt = []
            for c in frontier:
                row = self.q('SELECT parent_ids FROM lineages WHERE child_id=?', (c,))
                if not row:
                    continue
                try:
                    pids = json.loads(row[0]['parent_ids'] or '[]')
                except (TypeError, ValueError):
                    pids = []
                for pid in pids if isinstance(pids, list) else [pids]:
                    got = self.q('SELECT code FROM strategies WHERE id=?', (str(pid),))
                    if got and got[0]['code']:
                        out.append(got[0]['code'])
                    nxt.append(str(pid))
            frontier = nxt
        return out

    def claim_holdout_shot(self, sid, code, code_hash_, window=None):
        """Atomically take this family's one holdout shot or raise HoldoutAlreadyUsed. Returns the family key."""
        from ..lab.holdout import FAMILY_SIMILARITY, HoldoutAlreadyUsed, similar
        from ..lab.holdout import assert_holdout_allowed
        assert_holdout_allowed(sid)                     # e.g. ensure_seeded: never consumes a shot
        desk, keys, sk = self._family_rows(code, code_hash_, self.parent_codes(sid))
        toks = sk.split() if sk else []
        with self.db() as db:
            db.execute('BEGIN IMMEDIATE')
            for k in keys + [f'legacy:{code_hash_}']:
                hit = db.execute('SELECT strategy FROM holdout_shots WHERE family=?', (k,)).fetchone()
                if hit:
                    raise HoldoutAlreadyUsed(f'family {k} already used its holdout shot (strategy {hit["strategy"]})')
            for r in db.execute('SELECT family, strategy, skeleton FROM holdout_shots WHERE desk=? AND skeleton IS NOT NULL',
                                (desk,)).fetchall():
                if toks and similar(toks, r['skeleton'].split()) >= FAMILY_SIMILARITY:
                    raise HoldoutAlreadyUsed(f'near copy of family {r["family"]} (strategy {r["strategy"]}) which '
                                             'already used its holdout shot')
            now = time.time()
            for i, k in enumerate(keys):
                db.execute('INSERT INTO holdout_shots VALUES(?,?,?,?,?,?,?,?)',
                           (k, desk, sid, code_hash_, json.dumps(window, default=str) if window is not None else None,
                            sk if i == 0 else None, 'shot', now))
            db.execute('INSERT INTO holdout_uses VALUES(?,?,?,?)', (code_hash_, sid, now, None))
        return keys[0]

    def record_holdout_result(self, family, sid, passed):
        self.x('INSERT INTO holdout_results(family,strategy,passed,created) VALUES(?,?,?,?)',
               (family, sid, 1 if passed else 0, time.time()))

    def q(self, sql, args=()):
        with self.db() as db:
            return [dict(r) for r in db.execute(sql, args).fetchall()]

    def x(self, sql, args=()):
        with self.db() as db:
            cur = db.execute(sql, args)
            return cur.lastrowid

    # ----------------------------------------------------------------- control
    def set(self, key, value):
        self.x('INSERT OR REPLACE INTO control VALUES(?,?)', (key, json.dumps(value)))

    def get(self, key, default=None):
        r = self.q('SELECT value FROM control WHERE key=?', (key,))
        return json.loads(r[0]['value']) if r else default

    def request_stop(self):
        self.set('stop', True)

    def stop_requested(self):
        return bool(self.get('stop', False))

    def event(self, agent, kind, payload=None):
        self.x('INSERT INTO events(created,agent,kind,payload) VALUES(?,?,?,?)',
               (time.time(), agent, kind, json.dumps(payload or {}, default=str)[:20000]))

    def index(self, kind, ref, title, body, author):
        with self.db() as db:
            db.execute('DELETE FROM search WHERE kind=? AND ref=?', (kind, ref))
            db.execute('INSERT INTO search VALUES(?,?,?,?,?)', (kind, ref, title or '', body or '', author or ''))

    def search(self, query, kind=None, limit=8):
        terms = ' OR '.join(f'"{t}"' for t in query.replace('"', ' ').split() if len(t) > 1)
        if not terms:
            return []
        sql = 'SELECT kind, ref, title, snippet(search, 3, "[", "]", "...", 24) AS snippet, author FROM search WHERE search MATCH ?'
        args = [terms]
        if kind:
            sql += ' AND kind=?'; args.append(kind)
        sql += ' ORDER BY rank LIMIT ?'; args.append(limit)
        try:
            return self.q(sql, args)
        except sqlite3.OperationalError:
            return []

    # ----------------------------------------------------------------- agents
    def upsert_agent(self, name, role, pod):
        self.x('INSERT INTO agents(name,role,pod,status,last_active) VALUES(?,?,?,?,?) '
               'ON CONFLICT(name) DO UPDATE SET role=excluded.role, pod=excluded.pod', (name, role, pod, 'idle', time.time()))

    def agent_status(self, name, status, summary=None):
        if summary is None:
            self.x('UPDATE agents SET status=?, last_active=? WHERE name=?', (status, time.time(), name))
        else:
            self.x('UPDATE agents SET status=?, last_active=?, last_summary=? WHERE name=?', (status, time.time(), summary[:4000], name))

    def usage(self, agent, prompt, completion, seconds, status='ok', model=None, cached=0):
        if not getattr(self, '_cached_col', False):
            cols = [c['name'] for c in self.q('PRAGMA table_info(llm_requests)')]
            if 'cached_tokens' not in cols:
                self.x('ALTER TABLE llm_requests ADD COLUMN cached_tokens INTEGER DEFAULT 0')
            self._cached_col = True
        self.x('INSERT INTO llm_requests(agent,created,seconds,prompt_tokens,completion_tokens,status,model,cached_tokens) VALUES(?,?,?,?,?,?,?,?)',
               (agent, time.time(), seconds, prompt, completion, status, model, cached))
        self.x('UPDATE agents SET prompt_tokens=prompt_tokens+?, completion_tokens=completion_tokens+?, steps=steps+1 WHERE name=?',
               (prompt, completion, agent))

    # ----------------------------------------------------------------- board
    def post(self, kind, title, body, author, pod=None, assignee=None, parent=None, tags=None, status='open'):
        bid = _id({'thesis': 'T', 'task': 'K', 'review': 'R', 'question': 'Q', 'memo': 'M', 'request': 'Q'}.get(kind, 'B'))
        now = time.time()
        self.x('INSERT INTO board VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',
               (bid, kind, title, body, author, assignee, pod, status, parent, json.dumps(tags or []), now, now))
        self.index('board', bid, title, body, author)
        return bid

    def update_board(self, bid, author, status=None, comment=None, assignee=None):
        now = time.time()
        if status:
            self.x('UPDATE board SET status=?, updated=? WHERE id=?', (status, now, bid))
        if assignee:
            self.x('UPDATE board SET assignee=?, updated=? WHERE id=?', (assignee, now, bid))
        if comment:
            self.x('INSERT INTO comments(board_id,author,body,created) VALUES(?,?,?,?)', (bid, author, comment, now))
            self.x('UPDATE board SET updated=? WHERE id=?', (now, bid))
            b = self.q('SELECT title, body FROM board WHERE id=?', (bid,))
            if b:
                cs = ' '.join(c['body'] for c in self.q('SELECT body FROM comments WHERE board_id=?', (bid,)))
                self.index('board', bid, b[0]['title'], b[0]['body'] + ' ' + cs, author)

    def board_item(self, bid):
        r = self.q('SELECT * FROM board WHERE id=?', (bid,))
        if not r:
            return None
        item = r[0]
        item['comments'] = self.q('SELECT author, body, created FROM comments WHERE board_id=? ORDER BY id', (bid,))
        return item

    def board_list(self, status=None, kind=None, pod=None, assignee=None, limit=20):
        sql, args = 'SELECT id, kind, title, author, assignee, pod, status, updated FROM board WHERE 1=1', []
        for col, v in (('status', status), ('kind', kind), ('pod', pod), ('assignee', assignee)):
            if v:
                if col == 'status' and v == 'active':
                    sql += " AND status IN ('open','claimed','in_progress')"
                else:
                    sql += f' AND {col}=?'; args.append(v)
        sql += ' ORDER BY updated DESC LIMIT ?'; args.append(limit)
        return self.q(sql, args)

    # ----------------------------------------------------------------- messages
    def send(self, to, frm, body):
        self.x('INSERT INTO messages(to_agent,from_agent,body,created) VALUES(?,?,?,?)', (to, frm, body[:6000], time.time()))

    def inbox(self, agent, pod, mark=True, limit=12):
        rows = self.q("SELECT id, from_agent, to_agent, body, created FROM messages WHERE read=0 AND (to_agent=? OR to_agent=? OR to_agent='all') "
                      "AND from_agent<>? ORDER BY id LIMIT ?", (agent, f'pod:{pod}', agent, limit))
        if mark and rows:
            # broadcast messages stay unread for others: track per-agent reads for pod/all
            direct = [r['id'] for r in rows if r['to_agent'] == agent]
            if direct:
                self.x(f"UPDATE messages SET read=1 WHERE id IN ({','.join('?' * len(direct))})", direct)
            seen = set(self.get(f'seen:{agent}', []))
            rows = [r for r in rows if r['to_agent'] == agent or r['id'] not in seen]
            self.set(f'seen:{agent}', sorted(seen | {r['id'] for r in rows if r['to_agent'] != agent})[-500:])
        return rows

    def unread_count(self, agent, pod):
        seen = set(self.get(f'seen:{agent}', []))
        rows = self.q("SELECT id, to_agent FROM messages WHERE read=0 AND (to_agent=? OR to_agent=? OR to_agent='all') AND from_agent<>?",
                      (agent, f'pod:{pod}', agent))
        return sum(1 for r in rows if r['to_agent'] == agent or r['id'] not in seen)

    # ----------------------------------------------------------------- library
    def add_factor(self, name, author, pod, code, resolution, description):
        fid = _id('F')
        self.x('INSERT INTO factors VALUES(?,?,?,?,?,?,?,?,?,?,?)',
               (fid, name, author, pod, code, resolution, description, None, None, 'evaluating', time.time()))
        return fid

    def factor_report(self, fid, report, score, status):
        self.x('UPDATE factors SET report=?, score=?, status=? WHERE id=?', (json.dumps(report, default=str), score, status, fid))
        f = self.q('SELECT name, description, author FROM factors WHERE id=?', (fid,))[0]
        self.index('factor', fid, f['name'], f['description'], f['author'])

    def factors(self, limit=20, status='active', order='score'):
        rows = self.q(f'SELECT id, name, author, pod, resolution, description, score, report, status FROM factors '
                      f'WHERE status=? ORDER BY {"score DESC" if order == "score" else "created DESC"} LIMIT ?', (status, limit))
        for r in rows:
            r['report'] = json.loads(r['report']) if r['report'] else None
        return rows

    # ----------------------------------------------------------------- strategies
    def add_strategy(self, name, author, pod, code, code_hash, resolution, thesis):
        sid = _id('S')
        now = time.time()
        self.x('INSERT INTO strategies(id,name,author,pod,code,code_hash,resolution,thesis,status,created,updated) VALUES(?,?,?,?,?,?,?,?,?,?,?)',
               (sid, name, author, pod, code, code_hash, resolution, thesis, 'submitted', now, now))
        self.index('strategy', sid, name, thesis, author)
        return sid

    def update_strategy(self, sid, **fields):
        fields['updated'] = time.time()
        sets = ', '.join(f'{k}=?' for k in fields)
        vals = [json.dumps(v, default=str) if isinstance(v, (dict, list)) else v for v in fields.values()]
        self.x(f'UPDATE strategies SET {sets} WHERE id=?', vals + [sid])

    def strategy(self, sid):
        r = self.q('SELECT * FROM strategies WHERE id=?', (sid,))
        if not r:
            return None
        s = r[0]
        for k in ('dev_report', 'review', 'val_report', 'holdout_report', 'forward_report', 'paper_report', 'gate_reasons'):
            if s.get(k):
                try:
                    s[k] = json.loads(s[k])
                except Exception:
                    pass
        return s

    def strategies(self, status=None, limit=50):
        sql = 'SELECT id, name, author, pod, resolution, status, code_hash, created, updated FROM strategies'
        args = []
        if status:
            sql += ' WHERE status IN (%s)' % ','.join('?' * len(status)); args += list(status)
        sql += ' ORDER BY updated DESC LIMIT ?'
        return self.q(sql, args + [limit])

    # ----------------------------------------------------------------- trials (multiple-testing ledger)
    def trial(self, author, kind, code_hash, sharpe, days):
        self.x('INSERT INTO trials(author,kind,code_hash,sharpe,days,created) VALUES(?,?,?,?,?,?)',
               (author, kind, code_hash, sharpe, days, time.time()))

    @staticmethod
    def family_of(code, code_hash_=None):
        from ..lab.holdout import structural_family
        try:
            return structural_family(code)
        except (SyntaxError, ValueError, TypeError):
            return f'unknown:{code_hash_}'

    def record_eval_trial(self, family, desk, split, variant, sharpe, n_obs, periods, source, code_hash_=None,
                          params_hash=None):
        import math
        if sharpe is None or not math.isfinite(float(sharpe)):
            return
        periods = int(periods or 365)
        self.x('INSERT INTO eval_trials(family,params_hash,code_hash,desk,split,variant,sharpe,n_obs,periods,sr_pp,'
               'source,created) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',
               (family, params_hash, code_hash_, desk, split, variant, float(sharpe), int(n_obs or 0), periods,
                float(sharpe) / math.sqrt(periods), source, time.time()))

    def record_eval_trials(self, code, out, source):
        """Append every row of a harness `trial_log` (base + variants + parameter neighbours)."""
        fam = self.family_of(code, out.get('code_hash'))
        desk = out.get('desk') or 'l2'
        for r in out.get('trial_log') or []:
            self.record_eval_trial(fam, desk, out.get('split'), r.get('variant'), r.get('sharpe'), r.get('n_obs'),
                                   r.get('periods'), source, out.get('code_hash'), r.get('params_hash'))

    def trial_moments(self, code=None):
        """Cumulative per-period Sharpe moments {n, sum, sumsq} for the code's family and across all families."""
        def m(where='', args=()):
            r = self.q(f'SELECT COUNT(sr_pp) n, TOTAL(sr_pp) s, TOTAL(sr_pp*sr_pp) ss FROM eval_trials {where}', args)[0]
            return {'n': int(r['n'] or 0), 'sum': float(r['s'] or 0.0), 'sumsq': float(r['ss'] or 0.0)}
        out = {'global': m()}
        if code is not None:
            out['family'] = m('WHERE family=?', (self.family_of(code),))
        return out

    def trial_stats(self):
        # exploratory backtests on fewer than 10 days are too noisy to count as distinct strategy trials
        r = self.q("SELECT COUNT(DISTINCT code_hash) AS n, COUNT(*) AS evals FROM trials WHERE kind IN ('dev_backtest','submission') AND days>=10")[0]
        srs = [x['sharpe'] for x in self.q("SELECT sharpe FROM trials WHERE kind IN ('dev_backtest','submission') AND sharpe IS NOT NULL AND days>=10")]
        import numpy as np
        std = float(np.std(np.array(srs) / np.sqrt(365))) if len(srs) > 5 else None   # per-day Sharpe dispersion
        return {'distinct_strategies': r['n'] or 0, 'evaluations': r['evals'] or 0, 'sr_std_daily': std}

    # ----------------------------------------------------------------- playbook
    def add_lesson(self, author, claim, evidence, confidence, scope, tags):
        lid = _id('L')
        self.x('INSERT INTO lessons VALUES(?,?,?,?,?,?,?,?,?)',
               (lid, author, claim, evidence, confidence, scope, json.dumps(tags or []), 'active', time.time()))
        self.index('lesson', lid, claim, evidence + ' ' + (scope or '') + ' ' + ' '.join(tags or []), author)
        return lid

    def memo(self, author, kind, body):
        self.x('INSERT INTO memos(author,kind,body,created) VALUES(?,?,?,?)', (author, kind, body[:12000], time.time()))

    def latest_memo(self, kind):
        r = self.q('SELECT author, body, created FROM memos WHERE kind=? ORDER BY id DESC LIMIT 1', (kind,))
        return r[0] if r else None

    # ----------------------------------------------------------------- status
    def status(self):
        now = time.time()
        tok = self.q('SELECT SUM(prompt_tokens) p, SUM(completion_tokens) c, COUNT(*) n FROM llm_requests')[0]
        hour = self.q('SELECT SUM(prompt_tokens+completion_tokens) t, COUNT(*) n FROM llm_requests WHERE created>?', (now - 3600,))[0]
        return {
            'running': self.get('running', False), 'stop_requested': self.stop_requested(),
            'agents': self.q('SELECT name, role, pod, model, status, sessions, steps, prompt_tokens+completion_tokens AS tokens, last_active FROM agents ORDER BY pod, name'),
            'tokens_by_model': self.q('SELECT model, COUNT(*) AS requests, SUM(prompt_tokens+completion_tokens) AS tokens FROM llm_requests GROUP BY model'),
            'tokens': {'prompt': tok['p'] or 0, 'completion': tok['c'] or 0, 'requests': tok['n'] or 0,
                       'last_hour_tokens': hour['t'] or 0, 'last_hour_requests': hour['n'] or 0},
            'factors': self.q("SELECT status, COUNT(*) n FROM factors GROUP BY status"),
            'strategies': self.q('SELECT status, COUNT(*) n FROM strategies GROUP BY status'),
            'trials': self.trial_stats(),
            'board': self.q('SELECT status, COUNT(*) n FROM board GROUP BY status'),
            'lessons': self.q("SELECT COUNT(*) n FROM lessons WHERE status='active'")[0]['n'],
            'pods': self.q('SELECT name, weight, mandate FROM pods'),
            'concurrency': self.get('concurrency'),
        }

    # ----------------------------------------------------------------- RSI (Recursive Self-Improvement)
    def add_dynamic_hint(self, pattern, hint, source_error='', author='rsi-compiler', hits=0, resolved=0, status='active'):
        existing = self.q('SELECT id FROM dynamic_hints WHERE pattern=?', (pattern,))
        if existing:
            hid = existing[0]['id']
            self.x('UPDATE dynamic_hints SET hint=?, status=? WHERE id=?', (hint, status, hid))
            return hid
        hid = _id('H')
        self.x('INSERT INTO dynamic_hints(id,pattern,hint,source_error,author,hits,resolved_count,status,created) VALUES(?,?,?,?,?,?,?,?,?)',
               (hid, pattern, hint, (source_error or '')[:500], author, int(hits), int(resolved), status, time.time()))
        return hid

    def log_primitive_usage(self, primitive, agent, kind, ref_id='', success=True, metric_val=None):
        self.x('INSERT INTO primitive_usages(primitive,agent,kind,ref_id,success,metric_val,created) VALUES(?,?,?,?,?,?,?)',
               (primitive, agent, kind, str(ref_id or ''), 1 if success else 0,
                float(metric_val) if metric_val is not None else None, time.time()))

    def add_lineage(self, child_id, child_kind, parent_ids, operator, desk, delta_score=0.0):
        pids = json.dumps(list(parent_ids) if isinstance(parent_ids, (list, tuple, set)) else [str(parent_ids)])
        self.x('INSERT OR REPLACE INTO lineages(child_id,child_kind,parent_ids,operator,desk,delta_score,created) VALUES(?,?,?,?,?,?,?)',
               (str(child_id), str(child_kind), pids, str(operator), str(desk), float(delta_score or 0.0), time.time()))

    def add_charter_version(self, role, patch_text, rationale, pod=None, status='candidate',
                            drill_planted_score=None, drill_null_fdr=0, error_rate_delta=None):
        cid = _id('C')
        self.x('INSERT INTO charter_versions(id,role,pod,patch_text,rationale,status,drill_planted_score,drill_null_fdr,error_rate_delta,created) '
               'VALUES(?,?,?,?,?,?,?,?,?,?)',
               (cid, role, pod or '', patch_text, rationale, status,
                float(drill_planted_score) if drill_planted_score is not None else None,
                int(drill_null_fdr or 0),
                float(error_rate_delta) if error_rate_delta is not None else None,
                time.time()))
        return cid

    def log_rsi_run(self, kind, summary, metrics=None):
        rid = _id('RSI')
        self.x('INSERT INTO rsi_runs(id,kind,summary,metrics,created) VALUES(?,?,?,?,?)',
               (rid, kind, str(summary)[:2000], json.dumps(metrics or {}, default=str), time.time()))
        return rid

    def log_reasoning(self, agent: str, desk: str, category: str, title: str,
                      observation: str, diagnosis: str, hypothesis: str,
                      action_taken: str, code_snippet: str = '', proof: dict | None = None,
                      verdict: str = 'VERIFIED_LIFT') -> str:
        rid = _id('R')
        now = time.time()
        self.x(
            'INSERT INTO rsi_reasoning(id,created,agent,desk,category,title,observation,diagnosis,hypothesis,action_taken,code_snippet,proof_json,verdict) '
            'VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)',
            (
                rid, now, str(agent or 'rsi-engine'), str(desk or 'all'), str(category or 'tool_synthesis'),
                str(title or '')[:300], str(observation or '')[:2000], str(diagnosis or '')[:2000],
                str(hypothesis or '')[:2000], str(action_taken or '')[:2000], str(code_snippet or '')[:8000],
                json.dumps(proof or {}, default=str), str(verdict or 'VERIFIED_LIFT'),
            )
        )
        self.index('reasoning', rid, title, f"{observation} {diagnosis} {hypothesis} {action_taken}", agent)
        return rid

    def reasoning_episodes(self, limit: int = 35, desk: str | None = None) -> list[dict]:
        sql = 'SELECT * FROM rsi_reasoning'
        args: list = []
        if desk and desk != 'all':
            sql += " WHERE desk IN (?, 'all')"
            args.append(desk)
        sql += ' ORDER BY created DESC LIMIT ?'
        args.append(limit)
        rows = self.q(sql, tuple(args))
        for r in rows:
            if r.get('proof_json'):
                try:
                    r['proof'] = json.loads(r['proof_json'])
                except Exception:
                    r['proof'] = {}
            else:
                r['proof'] = {}
        return rows

    def reasoning_episode(self, rid: str) -> dict | None:
        rows = self.q('SELECT * FROM rsi_reasoning WHERE id=?', (rid,))
        if not rows:
            return None
        r = rows[0]
        if r.get('proof_json'):
            try:
                r['proof'] = json.loads(r['proof_json'])
            except Exception:
                r['proof'] = {}
        else:
            r['proof'] = {}
        return r


