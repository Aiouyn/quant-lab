"""Agent sessions through Claude Code (LLM_PROVIDER=claude_code): the firm runs on the owner's Claude plan.

A Claude plan's usage belongs to Anthropic's own apps, so the firm does not call a model API itself. Each agent session
is one non-interactive Claude Code run (`claude -p`) that:
- gets the agent's charter as its system prompt and the session brief as its prompt;
- has every built-in tool switched off (no shell, no file access): it can act only through the desk's own tools, served
  by this process over a local MCP endpoint, so the same sandbox, jail and evaluator gates apply as before;
- is billed to the plan the `claude` CLI is signed in to. API keys are removed from its environment so it can never
  fall back to pay-as-you-go API billing.

When the plan's usage limit is reached the run reports it; the whole firm then pauses (same gate as a gateway quota) and
probes again later. Plan limits are shared with the owner's own Claude use, so this mode is meant for a small firm
(DESK_CLAUDE_SESSIONS concurrent sessions, default 2).
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import secrets
import shutil
import sys
import time

from .tools import SPECS, ROLE_TOOLS

SERVER = 'desk'
PROTOCOL = '2025-06-18'
STRIP_ENV = ('ANTHROPIC_API_KEY', 'ANTHROPIC_AUTH_TOKEN', 'ANTHROPIC_BASE_URL', 'CLAUDE_CODE_USE_BEDROCK',
             'CLAUDE_CODE_USE_VERTEX', 'CLAUDE_CODE_USE_FOUNDRY', 'OPENAI_API_KEY', 'OPENCODE_API_KEY', 'MIMO_API_KEY')
LIMIT_RE = re.compile(r'(usage limit|rate limit|limit reached|hit your limit|out of extra usage|weekly limit)', re.I)
SESSION_NOTE = ('\n\n== HOW THIS SESSION WORKS ==\nYou act only through the desk tools (named mcp__desk__<tool>, e.g. '
                'mcp__desk__run_python). Work until your goal for this session is met, then call end_session with a '
                'dense summary (state, key numbers and ids, exact next steps) and stop.')


def mcp_tools(role):
    out = []
    for n in ROLE_TOOLS[role]:
        f = SPECS[n]['function']
        out.append({'name': f['name'], 'description': f.get('description') or '',
                    'inputSchema': f.get('parameters') or {'type': 'object', 'properties': {}}})
    return out


class Hub:
    """A minimal MCP server (streamable HTTP, JSON responses) on 127.0.0.1. Each session gets its own secret path."""

    def __init__(self):
        self.sessions = {}
        self.server = None
        self.port = None

    async def start(self):
        self.server = await asyncio.start_server(self._conn, '127.0.0.1', 0)
        self.port = self.server.sockets[0].getsockname()[1]
        return self

    def register(self, role, call):
        """call(name, args) -> awaitable str. Returns the session's URL."""
        token = secrets.token_urlsafe(24)
        self.sessions[token] = (role, call)
        return token, f'http://127.0.0.1:{self.port}/mcp/{token}'

    def unregister(self, token):
        self.sessions.pop(token, None)

    async def _conn(self, reader, writer):
        try:
            while True:
                line = await reader.readline()
                if not line:
                    break
                method, path, _ = (line.decode('latin-1').split(' ', 2) + ['', ''])[:3]
                headers = {}
                while True:
                    h = await reader.readline()
                    if h in (b'\r\n', b'\n', b''):
                        break
                    k, _, v = h.decode('latin-1').partition(':')
                    headers[k.strip().lower()] = v.strip()
                body = await reader.readexactly(int(headers.get('content-length') or 0))
                status, payload = await self._handle(method, path, body)
                data = b'' if payload is None else json.dumps(payload).encode()
                head = (f'HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {len(data)}\r\n'
                        f'Mcp-Session-Id: desk\r\n\r\n')
                writer.write(head.encode() + data)
                await writer.drain()
                if headers.get('connection', '').lower() == 'close':
                    break
        except (asyncio.IncompleteReadError, ConnectionError):
            pass
        finally:
            try:
                writer.close()
            except Exception:
                pass

    async def _handle(self, method, path, body):
        token = path.rstrip('/').rsplit('/', 1)[-1]
        if token not in self.sessions:
            return '404 Not Found', {'error': 'unknown session'}
        if method == 'DELETE':
            return '200 OK', None
        if method != 'POST':
            return '405 Method Not Allowed', None
        role, call = self.sessions[token]
        try:
            msg = json.loads(body or b'{}')
        except ValueError:
            return '400 Bad Request', {'jsonrpc': '2.0', 'id': None, 'error': {'code': -32700, 'message': 'parse error'}}
        msgs = msg if isinstance(msg, list) else [msg]
        out = [r for r in [await self._rpc(m, role, call) for m in msgs] if r is not None]
        if not out:
            return '202 Accepted', None
        return '200 OK', out if isinstance(msg, list) else out[0]

    async def _rpc(self, m, role, call):
        mid, meth, params = m.get('id'), m.get('method'), m.get('params') or {}
        if mid is None:                                   # notification
            return None
        if meth == 'initialize':
            res = {'protocolVersion': params.get('protocolVersion') or PROTOCOL, 'capabilities': {'tools': {}},
                   'serverInfo': {'name': SERVER, 'version': '1'}}
        elif meth == 'tools/list':
            res = {'tools': mcp_tools(role)}
        elif meth == 'tools/call':
            name, args = params.get('name', ''), params.get('arguments') or {}
            text = await call(name, args)
            err = text.startswith('{"error"')
            res = {'content': [{'type': 'text', 'text': text}], 'isError': err}
        elif meth == 'ping':
            res = {}
        else:
            return {'jsonrpc': '2.0', 'id': mid, 'error': {'code': -32601, 'message': f'unknown method {meth}'}}
        return {'jsonrpc': '2.0', 'id': mid, 'result': res}


def claude_bin():
    return os.getenv('DESK_CLAUDE_BIN', '').strip() or shutil.which('claude') or 'claude'


def child_env():
    env = {k: v for k, v in os.environ.items() if k not in STRIP_ENV}
    env['CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC'] = '1'
    return env


def build_cmd(settings, model, sys_file, mcp_file):
    cmd = [claude_bin(), '-p', '--output-format', 'json', '--model', model,
           '--system-prompt-file', str(sys_file), '--mcp-config', str(mcp_file), '--strict-mcp-config',
           '--tools', '', '--allowedTools', f'mcp__{SERVER}', '--no-session-persistence',
           '--disable-slash-commands']
    effort = getattr(settings, 'effort', '') or ''
    if effort:
        cmd += ['--effort', effort]
    return cmd


def limit_reset(text):
    """Seconds to wait after a plan-limit message (the CLI may append '|<epoch>')."""
    m = re.search(r'\|(\d{10})\b', text or '')
    if m:
        return max(300.0, min(float(m.group(1)) - time.time() + 60, 6 * 3600))
    return 1800.0


class Runner:
    """Runs agent sessions as Claude Code processes, a few at a time."""

    def __init__(self, settings, ledger, llm):
        self.s, self.L, self.llm = settings, ledger, llm
        self.slots = asyncio.Semaphore(max(1, int(os.getenv('DESK_CLAUDE_SESSIONS', '2'))))
        self.hub = Hub()
        self.timeout_s = int(os.getenv('DESK_CLAUDE_SESSION_S', '2700'))
        self._started = False

    async def ensure(self):
        if not self._started:
            await self.hub.start()
            self._started = True

    def pause(self, seconds, reason):
        until = time.time() + seconds
        if until > getattr(self.llm, 'quota_paused_until', 0):
            self.llm.quota_paused_until = until
            self.L.set('quota', {'exhausted_at': time.time(), 'next_probe': until, 'reason': reason[:240]})
            self.L.event('llm', 'QUOTA_EXHAUSTED', {'reason': reason[:240], 'provider': 'claude_code'})

    async def run(self, agent, toolbox, system, brief, model, max_steps, execute):
        """execute(toolbox, call) -> str runs one tool call. Returns (summary, info)."""
        await self.ensure()
        await self.llm.quota_gate(agent.name)
        steps = {'n': 0}

        async def call(name, args):
            if agent.end_requested is not None and name != 'end_session':
                return json.dumps({'error': 'session already ended: stop now'})
            steps['n'] += 1
            if steps['n'] > max_steps and name != 'end_session':
                return json.dumps({'error': f'step limit ({max_steps}) reached: call end_session with your summary now'})
            return await execute(toolbox, {'name': name, 'arguments': json.dumps(args)})

        async with self.slots:
            await self.llm.quota_gate(agent.name)
            token, url = self.hub.register(agent.role, call)
            sys_file = agent.workspace / '.session_system.md'
            mcp_file = agent.workspace / '.session_mcp.json'
            sys_file.write_text(system + SESSION_NOTE, encoding='utf-8')
            mcp_file.write_text(json.dumps({'mcpServers': {SERVER: {'type': 'http', 'url': url}}}), encoding='utf-8')
            t0 = time.time()
            proc = None
            try:
                kw = {}
                if sys.platform == 'win32':
                    kw['creationflags'] = 0x08000000           # CREATE_NO_WINDOW
                proc = await asyncio.create_subprocess_exec(
                    *build_cmd(self.s, model, sys_file, mcp_file), stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, cwd=str(agent.workspace),
                    env=child_env(), **kw)
                out, err = await asyncio.wait_for(proc.communicate(brief.encode('utf-8')), self.timeout_s)
            except asyncio.TimeoutError:
                if proc is not None:
                    proc.kill()
                    await proc.wait()
                out, err = b'', f'session timed out after {self.timeout_s}s'.encode()
            finally:
                self.hub.unregister(token)
            return self._finish(agent, model, out, err, proc, time.time() - t0, steps['n'])

    def _finish(self, agent, model, out, err, proc, seconds, steps):
        text = out.decode('utf-8', 'replace').strip()
        errs = err.decode('utf-8', 'replace').strip()
        try:
            res = json.loads(text.splitlines()[-1]) if text else {}
        except ValueError:
            res = {}
        u = res.get('usage') or {}
        read = u.get('cache_read_input_tokens') or 0
        prompt = (u.get('input_tokens') or 0) + read + (u.get('cache_creation_input_tokens') or 0)
        is_err = bool(res.get('is_error')) or not res or (proc is not None and proc.returncode not in (0, None))
        msg = str(res.get('result') or '') if res else (errs or text)[:2000]
        status = 'ok' if not is_err else f"error:claude_code:{res.get('subtype') or (proc.returncode if proc else 'timeout')}"
        self.L.usage(agent.name, prompt, u.get('output_tokens') or 0, seconds, status, model=f'plan:{model}', cached=read)
        info = {'steps': steps, 'turns': res.get('num_turns'), 'subtype': res.get('subtype'),
                'plan_cost_equiv_usd': res.get('total_cost_usd'), 'error': None if not is_err else (msg or errs)[:600]}
        if is_err and LIMIT_RE.search(msg + ' ' + errs):
            self.pause(limit_reset(msg + ' ' + errs), msg or errs)
        elif is_err:
            self.L.event(agent.name, 'CLAUDE_CODE_ERROR', info)
        return (None if is_err else msg), info
