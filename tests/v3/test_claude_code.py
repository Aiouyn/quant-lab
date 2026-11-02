"""LLM_PROVIDER=claude_code: agent sessions run as Claude Code processes that act only through the desk's tools."""
from __future__ import annotations

import asyncio
import json
import os
import stat
import sys
import types
import urllib.request

import pytest

from desk.company import claude_code as CC

pytestmark = pytest.mark.skipif(sys.platform == 'win32', reason='the fake claude binary is a POSIX script')

FAKE = r'''#!{py}
import json, os, sys, urllib.request
args = sys.argv[1:]
cfg = json.load(open(args[args.index('--mcp-config') + 1]))
url = cfg['mcpServers']['desk']['url']
system = open(args[args.index('--system-prompt-file') + 1]).read()
brief = sys.stdin.read()
def rpc(method, params=None, id=1):
    body = json.dumps({{'jsonrpc': '2.0', 'id': id, 'method': method, 'params': params or {{}}}}).encode()
    req = urllib.request.Request(url, body, {{'Content-Type': 'application/json'}})
    return json.loads(urllib.request.urlopen(req).read())
mode = os.environ.get('FAKE_MODE', 'ok')
if mode == 'limit':
    print(json.dumps({{'type': 'result', 'is_error': True, 'subtype': 'success', 'result': 'Claude AI usage limit reached|9999999999'}}))
    sys.exit(1)
rpc('initialize', {{'protocolVersion': '2025-06-18'}})
names = [t['name'] for t in rpc('tools/list')['result']['tools']]
r1 = rpc('tools/call', {{'name': 'board_list', 'arguments': {{}}}}, 2)['result']
r2 = rpc('tools/call', {{'name': 'set_pod_weights', 'arguments': {{'weights': {{}}, 'rationale': 'x'}}}}, 3)['result']
rpc('tools/call', {{'name': 'end_session', 'arguments': {{'summary': 'done'}}}}, 4)
r4 = rpc('tools/call', {{'name': 'board_list', 'arguments': {{}}}}, 5)['result']
report = {{'names': names, 'r1': r1, 'r2': r2, 'r4': r4, 'args': args, 'system': system, 'brief': brief,
          'leaked': [k for k in ('ANTHROPIC_API_KEY', 'OPENCODE_API_KEY') if k in os.environ]}}
open(os.path.join(os.getcwd(), 'fake_report.json'), 'w').write(json.dumps(report))
print(json.dumps({{'type': 'result', 'subtype': 'success', 'is_error': False, 'num_turns': 5, 'result': 'all done',
                  'usage': {{'input_tokens': 100, 'cache_read_input_tokens': 900, 'cache_creation_input_tokens': 0,
                            'output_tokens': 50}}, 'total_cost_usd': 0.001}}))
'''


class _L:
    def __init__(self):
        self.usage_rows, self.events, self.kv = [], [], {}

    def usage(self, *a, **k):
        self.usage_rows.append((a, k))

    def event(self, *a):
        self.events.append(a)

    def set(self, k, v):
        self.kv[k] = v


class _LLM:
    quota_paused_until = 0

    async def quota_gate(self, agent):
        pass


def _setup(tmp_path, monkeypatch, mode='ok'):
    fake = tmp_path / 'claude'
    fake.write_text(FAKE.format(py=sys.executable))
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv('DESK_CLAUDE_BIN', str(fake))
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'sk-ant-must-not-leak')
    monkeypatch.setenv('FAKE_MODE', mode)
    ws = tmp_path / 'ws'
    ws.mkdir()
    agent = types.SimpleNamespace(name='macro-r1', role='researcher', workspace=ws, end_requested=None)
    calls = []

    async def execute(toolbox, call):
        calls.append(call)
        if call['name'] == 'end_session':
            agent.end_requested = json.loads(call['arguments'])['summary']
            return json.dumps({'ok': True})
        if call['name'] == 'set_pod_weights':                 # the real Toolbox enforces role permissions
            return json.dumps({'error': 'set_pod_weights is not available to role researcher'})
        return json.dumps({'items': []})
    L, llm = _L(), _LLM()
    runner = CC.Runner(types.SimpleNamespace(effort='medium'), L, llm)
    return runner, agent, execute, calls, L, llm


def test_session_runs_through_desk_tools_on_the_plan(tmp_path, monkeypatch):
    runner, agent, execute, calls, L, _ = _setup(tmp_path, monkeypatch)
    summary, info = asyncio.run(runner.run(agent, None, 'CHARTER', 'BRIEF', 'claude-haiku-5-5', 30, execute))
    rep = json.loads((agent.workspace / 'fake_report.json').read_text())
    assert summary == 'all done' and info['steps'] == 3 and info['error'] is None
    assert 'run_python' in rep['names'] and 'set_pod_weights' not in rep['names']      # the role's tools only
    assert rep['r1']['isError'] is False and rep['r2']['isError'] is True
    assert 'already ended' in rep['r4']['content'][0]['text']                         # nothing after end_session
    assert [c['name'] for c in calls] == ['board_list', 'set_pod_weights', 'end_session']
    assert rep['system'].startswith('CHARTER') and rep['brief'] == 'BRIEF'
    a = rep['args']
    assert a[a.index('--tools') + 1] == '' and '--strict-mcp-config' in a and a[a.index('--model') + 1] == 'claude-haiku-5-5'
    assert rep['leaked'] == []                                    # never falls back to pay-as-you-go API billing
    (agent_, prompt, out, secs, status), kw = L.usage_rows[-1]
    assert status == 'ok' and prompt == 1000 and out == 50 and kw['model'] == 'plan:claude-haiku-5-5' and kw['cached'] == 900


def test_plan_limit_pauses_the_firm(tmp_path, monkeypatch):
    runner, agent, execute, calls, L, llm = _setup(tmp_path, monkeypatch, mode='limit')
    summary, info = asyncio.run(runner.run(agent, None, 'C', 'B', 'claude-haiku-5-5', 30, execute))
    assert summary is None and 'usage limit' in info['error']
    assert llm.quota_paused_until > 0 and L.kv['quota']['reason'].startswith('Claude AI usage limit')
    assert any(e[1] == 'QUOTA_EXHAUSTED' for e in L.events)


def test_step_limit_tells_the_model_to_wrap_up(tmp_path, monkeypatch):
    runner, agent, execute, calls, L, _ = _setup(tmp_path, monkeypatch)
    asyncio.run(runner.run(agent, None, 'C', 'B', 'claude-haiku-5-5', 1, execute))
    rep = json.loads((agent.workspace / 'fake_report.json').read_text())
    assert 'step limit' in rep['r2']['content'][0]['text'] and calls[-1]['name'] == 'end_session'


def test_config_claude_code_needs_no_key(monkeypatch):
    from desk import config as C
    monkeypatch.setenv('LLM_PROVIDER', 'claude_code')
    monkeypatch.setenv('MODEL_NAME', 'auto')
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'sk-ant-x')
    assert C._key() == '' and C._default_model() == 'claude-haiku-5-5'


def test_idle_agents_back_off_on_the_plan(tmp_path):
    from desk.company import agent as A
    a = A.Agent('macro-lead', 'lead', 'macro', tmp_path / 'ws')

    class L:
        def __init__(self):
            self.ev = []

        def agent_status(self, *x):
            pass

        def event(self, *x):
            self.ev.append(x)
    led = L()
    for _ in range(3):
        a._productive = 0
        a._close(led, 6)                                  # many reads, nothing produced
    assert a.idle_streak == 3
    a._productive = 1
    a._close(led, 2)
    assert a.idle_streak == 0


def test_plan_gap_is_a_hard_floor_that_messages_cannot_cut(tmp_path):
    from desk.company import agent as A
    a = A.Agent('macro-r1', 'researcher', 'macro', tmp_path / 'ws2')
    sleeps = []

    class Org:
        claude = object()
        n = 0

        def stagger(self, agent):
            return 0

        def should_stop(self):
            self.n += 1
            return self.n > 1

        async def _has(self):
            return True

        async def sleep(self, seconds, wake=None):
            sleeps.append((seconds, wake is not None))

    async def no_session(org):
        a._productive = 0
        a.idle_streak += 1
    a.session = no_session
    a._has_work = lambda org: Org()._has()
    asyncio.run(a.live(Org()))
    assert sleeps[0] == (A.MIN_GAP_PLAN_S * 2, False)             # first idle session: doubled, no early wake
    assert A.PRODUCTIVE.isdisjoint({'message', 'board_post', 'post_memo'})


def test_pod_priority_override_puts_open_desks_first(monkeypatch):
    import importlib
    from desk.company import runtime
    monkeypatch.setenv('DESK_POD_PRIORITY', 'xsec,events')
    rt = importlib.reload(runtime)
    try:
        names = [n for n, _, _ in rt.roster(4)]
        assert names == ['cio', 'officer-1', 'xsec-lead', 'xsec-r1']
    finally:
        monkeypatch.delenv('DESK_POD_PRIORITY')
        importlib.reload(runtime)
