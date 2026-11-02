"""RSI telemetry for Desk OS: the small, honest part of the self-improvement layer.

What lives here (Phase 0, see the RSI audit of 2026-11-02):
- primitive usage attribution: which `desk.lib.contrib` modules agents import, and what happened when they did;
- the error-to-hint compiler: recurring sandbox failures clustered into fix hints, with hit / resolution counters
  that are only ever incremented by real events (desk.company.hints);
- the generation scoreboard (RSI_SCOREBOARD.md): every field is read from the ledger; null means unknown.

What deliberately does NOT live here any more: strategy breeding, primitive synthesis, charter patches, the
auto-promotion loop and the background daemon. The previous engine evaluated candidates on synthetic panels,
had fallbacks that invented Sharpe ratios, and pushed candidates through validation and holdout in one call
without officer review. Self-modification comes back only after the evaluator is sealed and the firm is scored on
planted / null drills for every live desk.

Contract for everything below: never invent a number. No priors dressed as observations, no default Sharpe,
no default pass rate. When there is no evidence, return None (or 0 for counts).
"""
from __future__ import annotations

import ast
import json
import re
import time
from pathlib import Path

from .hints import RULES as STATIC_HINT_RULES

CONTRIB_DIR = Path(__file__).resolve().parent.parent / 'lib' / 'contrib'
MIN_USES_FOR_STATS = 5           # below this, per-primitive means are reported as None (too few to mean anything)


# ------------------------------------------------------------------ primitive usage attribution
def extract_contrib_imports(code: str) -> list[str]:
    """Sorted unique `desk.lib.contrib.<module>` names imported by `code` (only modules that exist on disk)."""
    if not code or 'contrib' not in code:
        return []
    found: set[str] = set()
    try:
        for node in ast.walk(ast.parse(code)):
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module == 'desk.lib.contrib':
                    found.update(a.name for a in node.names if a.name != '*')
                elif node.module.startswith('desk.lib.contrib.'):
                    found.add(node.module.split('desk.lib.contrib.', 1)[1].split('.')[0])
            elif isinstance(node, ast.Import):
                for a in node.names:
                    if a.name.startswith('desk.lib.contrib.'):
                        found.add(a.name.split('desk.lib.contrib.', 1)[1].split('.')[0])
    except SyntaxError:
        found.update(re.findall(r'desk\.lib\.contrib\.([A-Za-z_]\w*)', code))
    known = {p.stem for p in CONTRIB_DIR.glob('*.py') if not p.stem.startswith('_')}
    return sorted(found & known)


def discover_contrib_modules() -> list[dict]:
    """Installed `desk/lib/contrib` modules with their public functions and first docstring line."""
    out = []
    for p in sorted(CONTRIB_DIR.glob('*.py')) if CONTRIB_DIR.exists() else []:
        if p.stem.startswith('_'):
            continue
        try:
            tree = ast.parse(p.read_text(encoding='utf-8'))
            doc = (ast.get_docstring(tree) or '').strip().splitlines()
            funcs = [n.name for n in tree.body if isinstance(n, ast.FunctionDef) and not n.name.startswith('_')]
        except (SyntaxError, UnicodeDecodeError):
            doc, funcs = [], []
        out.append({'name': p.stem, 'functions': funcs, 'doc': (doc[0] if doc else '')[:220]})
    return out


def record_primitive_outcome(ledger, code: str, agent: str, kind: str, ref_id: str = '',
                             success: bool = True, metric_val: float | None = None) -> list[str]:
    """Log one usage row per contrib module imported by `code`. Returns the modules found."""
    prims = extract_contrib_imports(code)
    for p in prims:
        ledger.log_primitive_usage(p, agent, kind, ref_id=ref_id, success=success, metric_val=metric_val)
    return prims


def primitive_lift_leaderboard(ledger) -> list[dict]:
    """Per-primitive usage statistics from real ledger rows.

    These are CORRELATIONAL dev-side statistics (agents choose when to use a primitive), not causal lift. The field
    names keep the old API, but no prior, default or shrinkage target is injected: a mean over fewer than
    MIN_USES_FOR_STATS rows is None.
    """
    rows = ledger.q('SELECT primitive, kind, COUNT(*) AS n, SUM(success) AS ok_n, AVG(metric_val) AS mean_val '
                    'FROM primitive_usages GROUP BY primitive, kind')
    by: dict[str, dict] = {m['name']: {'primitive': m['name'], 'functions': m['functions'], 'doc': m['doc'],
                                       'sandbox_runs': 0, 'sandbox_ok': 0, 'factor_uses': 0, 'mean_factor_score': None,
                                       'strategy_uses': 0, 'mean_strategy_dev_sharpe': None}
                           for m in discover_contrib_modules()}
    for r in rows:
        if r['primitive'] not in by:          # module was removed from disk: its history is not a current tool
            continue
        d, n = by[r['primitive']], int(r['n'] or 0)
        mean = round(float(r['mean_val']), 3) if r['mean_val'] is not None and n >= MIN_USES_FOR_STATS else None
        if r['kind'] == 'run_python':
            d['sandbox_runs'] += n
            d['sandbox_ok'] += int(r['ok_n'] or 0)
        elif r['kind'] == 'factor':
            d['factor_uses'] += n
            d['mean_factor_score'] = mean
        elif r['kind'] == 'strategy':
            d['strategy_uses'] += n
            d['mean_strategy_dev_sharpe'] = mean
    out = []
    for d in by.values():
        runs = d['sandbox_runs']
        d['total_uses'] = runs + d['factor_uses'] + d['strategy_uses']
        d['pass_rate'] = round(d['sandbox_ok'] / runs, 3) if runs >= MIN_USES_FOR_STATS else None
        d['status'] = ('UNUSED' if d['total_uses'] == 0 else
                       'INSUFFICIENT_DATA' if d['pass_rate'] is None else
                       'LOW_PASS_RATE' if d['pass_rate'] < 0.6 else 'OK')
        d['note'] = 'correlational usage statistics on dev data; not a causal lift estimate'
        out.append(d)
    out.sort(key=lambda x: (x['total_uses'], x['pass_rate'] or 0), reverse=True)
    return out


def format_primitive_brief(ledger, limit: int = 8) -> str:
    """Session-brief block listing the most used primitives with their observed (not estimated) stats."""
    used = [r for r in primitive_lift_leaderboard(ledger) if r['total_uses'] > 0][:limit]
    if not used:
        return ''
    lines = ['== PRIMITIVE USAGE (desk.lib.contrib; observed, correlational, dev only) ==']
    for r in used:
        fn = ', '.join(r['functions'][:3]) or r['primitive']
        pr = f"{r['pass_rate'] * 100:.0f}%" if r['pass_rate'] is not None else 'n/a'
        lines.append(f"- `from desk.lib.contrib.{r['primitive']} import {fn}` · uses={r['total_uses']} · "
                     f"sandbox pass {pr} · {r['status']}")
    return '\n'.join(lines)


# ------------------------------------------------------------------ error-to-hint compiler
HINT_TEMPLATES = [
    (r"InvalidOperationError:.*rolling_|rolling_\w+.*min_periods",
     "Polars rolling functions take integer windows and `min_samples=` (not `min_periods=`): "
     "pl.col('x').rolling_mean(20, min_samples=5).over('symbol')."),
    (r"ComputeError:.*(division by zero|log|sqrt|domain)",
     "Guard denominators and log inputs: .clip(lower_bound=1e-9) or + 1e-9 before dividing or taking logs."),
    (r"ShapeError:.*(broadcast|length|shape)",
     "Keep the row count: compute inside panel.with_columns(<expr>.over('symbol')) instead of mixing filtered Series."),
    (r"TypeError:.*'Expr'",
     "A pl.Expr cannot be mixed with NumPy directly: evaluate with df.select(<expr>).to_series().to_numpy() first."),
    (r"(ValueError:.*Input X contains|LightGBMError)",
     "Model fit got NaN/inf: np.nan_to_num(X) and drop rows where the label is not finite before fitting."),
    (r"ZeroDivisionError",
     "Add + 1e-9 to rolling std, volume sums and depth denominators."),
    (r"IndexError:.*(out of bounds|empty list|out of range)",
     "Check len(sub) == 0 before indexing [0], or aggregate with group_by(...).agg(...)."),
    (r"(PanicException|join_asof.*sorted|not sorted)",
     "join_asof needs both frames sorted on the key: call .sort('ts') (or 'date') on both."),
]


def _failures(ledger, limit=500) -> list[str]:
    out = []
    for ev in ledger.q("SELECT payload FROM events WHERE kind IN ('RUN_PYTHON','TOOL_ERROR') ORDER BY id DESC LIMIT ?", (limit,)):
        try:
            p = json.loads(ev['payload'] or '{}')
        except (TypeError, ValueError):
            continue
        err = (p.get('error') or '').strip()
        if not err and p.get('status') not in (None, 'ok'):
            err = str(p.get('status'))
        if err:
            out.append(err)
    return out


def compile_error_hints(ledger, author: str = 'rsi-compiler', min_cluster: int = 2) -> dict:
    """Turn recurring, not-yet-covered sandbox failures into dynamic hints.

    A hint is created only for a failure signature seen at least `min_cluster` times in the ledger. New hints start
    at hits=0, resolved=0: their counters move only when desk.company.hints serves them and the agent's next run
    succeeds or fails.
    """
    failures = _failures(ledger)
    covered = {p for p, _ in STATIC_HINT_RULES} | {r['pattern'] for r in ledger.q('SELECT pattern FROM dynamic_hints')}

    def is_covered(err):
        return any(re.search(p, err, re.S) for p in covered)

    compiled = []
    for pat, fix in HINT_TEMPLATES:
        n = sum(1 for f in failures if re.search(pat, f, re.S))
        if n >= min_cluster and pat not in covered:
            hid = ledger.add_dynamic_hint(pattern=pat, hint=fix, source_error=next(f for f in failures if re.search(pat, f, re.S)),
                                          author=author, hits=0, resolved=0, status='active')
            compiled.append({'id': hid, 'pattern': pat, 'cluster_size': n, 'source': 'template'})
            covered.add(pat)

    clusters: dict[str, list[str]] = {}
    for err in failures:
        if is_covered(err):
            continue
        sig = re.sub(r"'[^']{1,40}'|\"[^\"]{1,40}\"", "'<id>'", err)
        sig = re.sub(r'\b\d+(\.\d+)?\b', '<N>', sig).strip()[:140]
        if len(sig) >= 12:
            clusters.setdefault(sig, []).append(err)
    for sig, examples in sorted(clusters.items(), key=lambda kv: len(kv[1]), reverse=True)[:5]:
        if len(examples) < min_cluster:
            continue
        m = re.match(r'^([A-Za-z_][\w.]*(?:Error|Exception))\s*:\s*(.+)$', examples[0])
        if not m:
            continue
        words = [re.escape(w) for w in re.findall(r'[A-Za-z_]{3,}', m.group(2))[:3]]
        if not words:
            continue
        pat = rf"{re.escape(m.group(1))}:.*{'.*'.join(words)}"
        if pat in covered or not re.search(pat, examples[0], re.S):
            continue
        fix = (f"Recurring {m.group(1)} ({len(examples)}x in recent runs): print(panel.columns) to check names, keep "
               f"expressions per asset with .over(...), and guard NaN/zero before returning.")
        hid = ledger.add_dynamic_hint(pattern=pat, hint=fix, source_error=examples[0][:300], author=author,
                                      hits=0, resolved=0, status='active')
        compiled.append({'id': hid, 'pattern': pat, 'cluster_size': len(examples), 'source': 'cluster'})
        covered.add(pat)

    active = ledger.q("SELECT COUNT(*) AS n FROM dynamic_hints WHERE status='active'")[0]['n']
    summary = {'scanned_failures': len(failures), 'new_hints_compiled': len(compiled),
               'active_dynamic_hints': active, 'compiled': compiled}
    ledger.log_rsi_run('compile_hints', f'Compiled {len(compiled)} new dynamic hints ({active} active)', summary)
    return summary


# ------------------------------------------------------------------ charter overlays (drill-gated)
def active_charter_overlay(ledger, role: str, pod: str | None = None) -> str:
    """Process notes appended to a role's charter. Only variants a paired drill trial accepted are live
    (desk/lab/process_trials.py); the retired engine's charter_versions are never read."""
    from ..lab.process_trials import overlay_text
    try:
        return overlay_text(ledger, role, pod)
    except Exception:
        return ''


# ------------------------------------------------------------------ scoreboard (RSI_SCOREBOARD.md)
def _json(v):
    if isinstance(v, (dict, list)) or v is None:
        return v
    try:
        return json.loads(v)
    except (TypeError, ValueError):
        return None


def _sharpe(rep):
    s = ((_json(rep) or {}).get('summary') or {}).get('sharpe')
    return float(s) if isinstance(s, (int, float)) else None


def _count(ledger, sql):
    try:
        return ledger.q(sql)[0]['n']
    except Exception:
        return None


def _trial_counts(ledger):
    try:
        rows = ledger.q('SELECT verdict, COUNT(*) n FROM process_trials GROUP BY verdict')
    except Exception:
        return {}
    return {r['verdict']: r['n'] for r in rows}


def compute_generation_scoreboard(ledger) -> dict:
    """The RSI scoreboard. Every value comes from ledger rows; null means unknown, never a default."""
    rows = ledger.q('SELECT id, status, dev_report, val_report, holdout_report, forward_report FROM strategies')
    dev = [x for x in (_sharpe(r['dev_report']) for r in rows) if x is not None]
    val = [x for x in (_sharpe(r['val_report']) for r in rows) if x is not None]
    hold = [x for x in (_sharpe(r['holdout_report']) for r in rows) if x is not None]
    approved = [r for r in rows if r['status'] in ('approved', 'paper')]
    fwd = [_json(r['forward_report']) for r in approved if _json(r['forward_report'])]
    fwd_sharpes = [x for x in (((f.get('summary') or {}).get('sharpe')) for f in fwd) if isinstance(x, (int, float))]
    verdicts = [f.get('verdict') for f in fwd if f.get('verdict')]
    confirmed = sum(v == 'FORWARD_CONFIRMED' for v in verdicts)
    decided = sum(v in ('FORWARD_CONFIRMED', 'FORWARD_FAILED') for v in verdicts)

    port = ledger.q('SELECT report FROM portfolios ORDER BY created DESC LIMIT 1')
    port_rep = _json(port[0]['report']) if port else None
    port_sharpe = None
    if isinstance(port_rep, dict):
        ps = port_rep.get('sharpe', (port_rep.get('summary') or {}).get('sharpe'))
        port_sharpe = float(ps) if isinstance(ps, (int, float)) else None

    h = ledger.q('SELECT SUM(hits) AS hits, SUM(resolved_count) AS res FROM dynamic_hints WHERE hits > 0')[0]
    hint_rate = round(h['res'] / h['hits'], 3) if h['hits'] else None

    drill = ledger.get('drill_scores', None) if hasattr(ledger, 'get') else None   # written only by a real drill report
    drill = drill if isinstance(drill, dict) else {}

    if decided:
        forward_verdict = 'CONFIRMED' if confirmed else 'FAILED'
    elif fwd:
        forward_verdict = 'RUNNING'
    else:
        forward_verdict = 'NOT_RUN'
    return {
        'computed_at': time.time(),
        'strategies_total': len(rows),
        'approved': len(approved),
        'best_sharpe': round(max(dev), 3) if dev else None,
        'mean_sharpe': round(sum(dev) / len(dev), 3) if dev else None,
        'best_val_sharpe': round(max(val), 3) if val else None,
        'best_holdout_sharpe': round(max(hold), 3) if hold else None,
        'portfolio_sharpe': port_sharpe,
        'planted_signal_corr': drill.get('planted_signal_corr'),
        'drill_verdict': drill.get('verdict'),
        'null_false_discoveries': drill.get('null_false_discoveries'),
        'drill_desks_covered': drill.get('desks_covered') or [],
        'hint_resolution_rate': hint_rate,
        'process_variants_live': _count(ledger, "SELECT COUNT(*) n FROM process_variants WHERE status='active'"),
        'process_trials': _trial_counts(ledger),
        'mean_primitive_lift': None,     # no causal lift estimator exists yet (needs drill ablations)
        'forward_sharpe': round(max(fwd_sharpes), 3) if fwd_sharpes else None,
        'forward_verdict': forward_verdict,
        'forward_confirmed': confirmed,
        'forward_decided': decided,
        # Phase 0 fitness: share of forward-decided approvals that were confirmed on unseen data. Proxies the agents
        # can move themselves (hint resolution, primitive usage, breeding volume) are reported above but excluded.
        'generation_fitness': round(confirmed / decided, 3) if decided else 0.0,
    }


def rsi_snapshot(ledger, settings=None) -> dict:
    """KPIs for the brief, the CIO and the dashboard."""
    sb = compute_generation_scoreboard(ledger)
    kpis = {k: sb[k] for k in ('approved', 'best_val_sharpe', 'forward_verdict', 'forward_confirmed',
                               'hint_resolution_rate', 'generation_fitness', 'drill_verdict', 'null_false_discoveries',
                               'drill_desks_covered', 'process_variants_live', 'process_trials')}
    kpis['active_dynamic_hints'] = ledger.q("SELECT COUNT(*) AS n FROM dynamic_hints WHERE status='active'")[0]['n']
    kpis['contrib_modules'] = len(discover_contrib_modules())
    return {'kpis': kpis, 'latest_scoreboard': sb}
