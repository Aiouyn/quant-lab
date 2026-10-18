"""An agent = a long-lived colleague who works in sessions. Each session starts from a lean brief
(memo, digest, pod state, inbox, own notebook) instead of a replayed transcript, runs a tool loop, and ends
with a notebook entry so the next session (or a context reset) continues seamlessly."""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from .llm import assistant_message, estimate_tokens
from .roles import charter, PODS
from .tools import Toolbox, specs_for, compact

CADENCE_S = {'cio': 2700, 'librarian': 3600, 'pm': 1800, 'execution': 3600, 'toolsmith': 2400,
             'officer': 0, 'lead': 30, 'researcher': 10}


class Agent:
    def __init__(self, name, role, pod, workspace: Path):
        self.name, self.role, self.pod = name, role, pod
        self.workspace = workspace
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.end_requested = None
        self.sessions = 0

    # ------------------------------------------------------------------ lifecycle
    async def live(self, org):
        await asyncio.sleep(org.stagger(self))
        while not org.should_stop():
            if not await self._has_work(org):
                await org.sleep(60)
                continue
            try:
                await self.session(org)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                org.ledger.event(self.name, 'SESSION_ERROR', {'error': repr(e)[:1000]})
                org.ledger.agent_status(self.name, 'error')
                await org.sleep(90)
            gap = CADENCE_S.get(self.role, 10)
            if self.role in ('officer', 'pm'):
                await org.sleep(gap or 30, wake=lambda: self._direct_unread(org) > 0)
            else:
                await org.sleep(gap, wake=lambda: org.ledger.unread_count(self.name, self.pod) > 0 and self.role not in ('cio',))

    async def _has_work(self, org):
        if self.role == 'officer':
            q = org.ledger.q("SELECT COUNT(*) n FROM strategies WHERE status='review'")[0]['n']
            return q > 0 or self._direct_unread(org) > 0
        if self.role == 'pm':
            n = org.ledger.q("SELECT COUNT(*) n FROM strategies WHERE status IN ('approved','paper')")[0]['n']
            return n > 0 or self._direct_unread(org) > 0
        return True

    def _direct_unread(self, org):
        return org.ledger.q('SELECT COUNT(*) n FROM messages WHERE to_agent=? AND read=0', (self.name,))[0]['n']

    # ------------------------------------------------------------------ session
    async def session(self, org):
        L = org.ledger
        row = L.q('SELECT pod FROM agents WHERE name=?', (self.name,))
        if row and row[0]['pod']:
            self.pod = row[0]['pod']            # CIO may have reassigned us
        self.end_requested = None
        self.sessions += 1
        L.x('UPDATE agents SET sessions=sessions+1 WHERE name=?', (self.name,))
        L.agent_status(self.name, 'working')
        tools = Toolbox(org, self)
        specs = specs_for(self.role)
        messages = [{'role': 'system', 'content': charter(self.role, self.pod, ledger=L)},
                    {'role': 'user', 'content': org.brief(self)}]
        max_steps = org.settings.max_steps_per_session
        nudged = False
        compacting = False
        step = 0
        for step in range(max_steps):
            if org.should_stop():
                break
            if not compacting and estimate_tokens(messages) > org.settings.context_budget_tokens:
                compacting = True
                messages.append({'role': 'user', 'content': 'Context budget reached. Now: append a dense notebook entry '
                                 '(state, key numbers/ids, exact next steps), then call end_session.'})
            if compacting and step >= max_steps - 1:
                break
            await org.budget_gate()
            res = await org.llm.chat(self.name, _rolling_context(_trim_reasoning(messages)), specs, priority=org.priority(self),
                                     model=org.model_for_role(self.role))
            messages.append(assistant_message(res))
            calls = res['tool_calls']
            if not calls:
                if nudged or res['finish'] == 'stop' and res['content'] and step > 2:
                    self.end_requested = self.end_requested or (res['content'] or '')[:2000]
                    break
                nudged = True
                messages.append({'role': 'user', 'content': 'Act through tools (run_python, board, register/submit, ...). '
                                 'If you are done, call end_session with a summary.'})
                continue
            results = await asyncio.gather(*[self._exec(tools, c) for c in calls])
            for c, r in zip(calls, results):
                messages.append({'role': 'tool', 'tool_call_id': c['id'] or f'call_{calls.index(c)}', 'content': r})
            if self.end_requested:
                break
        summary = self.end_requested or f'(session ended after {step + 1} steps without summary)'
        nb = self.workspace / 'notebook.md'
        with open(nb, 'a', encoding='utf-8') as f:
            f.write(f'\n\n### Session {self.sessions} summary ({time.strftime("%Y-%m-%d %H:%M")})\n{summary}\n')
        L.agent_status(self.name, 'idle', summary)
        L.event(self.name, 'SESSION_END', {'steps': step + 1, 'summary': summary[:1500]})

    async def _exec(self, tools, call):
        try:
            args = json.loads(call['arguments'] or '{}')
            if not isinstance(args, dict):
                raise ValueError('arguments must be a JSON object')
        except Exception as e:
            return json.dumps({'error': f'could not parse arguments as JSON: {e}'})
        t0 = time.time()
        out = await tools.call(call['name'], args)
        tools.L.event(self.name, 'TOOL', {'tool': call['name'], 'seconds': round(time.time() - t0, 1),
                                          'error': out.get('error')[:300] if isinstance(out, dict) and out.get('error') else None})
        return compact(out, 6000 if call['name'] in ('run_python', 'dossier', 'evaluate_strategy', 'combine_factors') else 4000)


CLIP_BLOCK = 6


def _frozen(n, keep, block=CLIP_BLOCK):
    """How many of the n oldest items to clip. The clip frontier advances in BLOCKS (not one item per step) so the
    rendered prefix stays byte-identical for `block` steps: MiMo bills cached prefix tokens at 2 vs 100 credits, and a
    frontier that moves every step re-bills the whole tail as cache misses (measured 2026-10-07: ~6.4k missed tokens
    per request vs ~2.5k of real growth)."""
    return max(0, (n - keep) // block * block)


def _rolling_context(messages, keep_tools=3, keep_assistant=4, tool_stub=600, text_stub=1200):
    """Input tokens dominate cost (each step re-sends the session). Keep the system prompt, brief and the most
    recent tool results / assistant turns verbatim; older tool outputs and long assistant texts become short stubs
    (the agent has already digested them; key numbers belong in its reasoning, notebook and lessons)."""
    tool_idx = [i for i, m in enumerate(messages) if m.get('role') == 'tool']
    asst_idx = [i for i, m in enumerate(messages) if m.get('role') == 'assistant']
    old_tools = set(tool_idx[:_frozen(len(tool_idx), keep_tools)])
    old_asst = set(asst_idx[:_frozen(len(asst_idx), keep_assistant)])
    out = []
    for i, m in enumerate(messages):
        if i in old_tools and len(m.get('content') or '') > tool_stub:
            m = dict(m, content=m['content'][:tool_stub] + ' ...[older tool output clipped to save context]')
        elif i in old_asst and len(m.get('content') or '') > text_stub:
            m = dict(m, content=m['content'][:text_stub] + ' ...[clipped]')
        out.append(m)
    return out


def _trim_reasoning(messages, keep=2):
    """Pass opaque reasoning back only for the most recent assistant turns (MiMo needs the active chain);
    older reasoning is dropped to keep the context lean."""
    idx = [i for i, m in enumerate(messages) if m.get('role') == 'assistant' and m.get('reasoning_content')]
    drop = set(idx[:_frozen(len(idx), keep)])
    if not drop:
        return messages
    out = []
    for i, m in enumerate(messages):
        if i in drop:
            m = {k: v for k, v in m.items() if k != 'reasoning_content'}
            if not m.get('content') and not m.get('tool_calls'):
                m['content'] = '(reasoning omitted)'        # MiMo rejects empty assistant turns
        out.append(m)
    return out
