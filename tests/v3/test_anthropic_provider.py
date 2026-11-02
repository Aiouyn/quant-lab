"""LLM_PROVIDER=anthropic: OpenAI-format agent messages go to the Messages API and come back in the same shape."""
from __future__ import annotations

import asyncio
import dataclasses
import json
import types

from desk.company.llm import LLM, assistant_message, estimate_tokens, to_anthropic


def test_conversion_merges_tool_results_and_keeps_roles_alternating():
    msgs = [{'role': 'system', 'content': 'charter'},
            {'role': 'user', 'content': 'brief'},
            {'role': 'assistant', 'content': None, 'tool_calls': [
                {'id': 't1', 'type': 'function', 'function': {'name': 'board', 'arguments': '{"n": 3}'}},
                {'id': 't2', 'type': 'function', 'function': {'name': 'run_python', 'arguments': 'not json'}}]},
            {'role': 'tool', 'tool_call_id': 't1', 'content': 'r1'},
            {'role': 'tool', 'tool_call_id': 't2', 'content': 'r2'},
            {'role': 'user', 'content': 'Context budget reached.'}]
    system, out = to_anthropic(msgs)
    assert system == 'charter'
    assert [m['role'] for m in out] == ['user', 'assistant', 'user']
    a = out[1]['content']
    assert [b['type'] for b in a] == ['tool_use', 'tool_use'] and a[0]['input'] == {'n': 3} and a[1]['input'] == {}
    last = out[2]['content']
    assert [b['type'] for b in last] == ['tool_result', 'tool_result', 'text']
    assert [b['tool_use_id'] for b in last[:2]] == ['t1', 't2']


class _Block:
    def __init__(self, **d):
        self.d = d

    def to_dict(self):
        return dict(self.d)


class _Stream:
    def __init__(self, msg):
        self.msg = msg

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    def __aiter__(self):
        async def gen():
            yield 'event'
        return gen()

    async def get_final_message(self):
        return self.msg


class _Ledger:
    def __init__(self):
        self.rows = []

    def usage(self, *a, **k):
        self.rows.append((a, k))

    def set(self, *a):
        pass

    def event(self, *a):
        pass


def _llm(msg, seen):
    from desk.config import SETTINGS
    s = dataclasses.replace(SETTINGS, provider='anthropic', model='claude-haiku-5-5', flash_model='claude-haiku-5-5',
                            api_key='sk-test', base_url='', effort='medium')
    llm = LLM(s, _Ledger())

    def stream(**kw):
        seen.append(kw)
        return _Stream(msg)
    llm.client = types.SimpleNamespace(beta=types.SimpleNamespace(messages=types.SimpleNamespace(stream=stream)))
    return llm


def test_request_shape_and_result_round_trip():
    thinking = {'type': 'thinking', 'thinking': '', 'signature': 'SIG-abc'}
    msg = types.SimpleNamespace(
        content=[_Block(**thinking), _Block(type='text', text='Checking the board.'),
                 _Block(type='tool_use', id='toolu_1', name='board', input={'n': 5})],
        stop_reason='tool_use',
        usage=types.SimpleNamespace(input_tokens=100, output_tokens=40, cache_read_input_tokens=900,
                                    cache_creation_input_tokens=50))
    seen = []
    llm = _llm(msg, seen)
    tools = [{'type': 'function', 'function': {'name': 'zeta', 'description': 'z', 'parameters': {'type': 'object'}}},
             {'type': 'function', 'function': {'name': 'board', 'description': 'b', 'parameters': {'type': 'object'}}}]
    history = [{'role': 'system', 'content': 'charter'}, {'role': 'user', 'content': 'brief'}]
    res = asyncio.run(llm.chat('a1', history, tools))
    kw = seen[0]
    assert kw['model'] == 'claude-haiku-5-5' and kw['output_config'] == {'effort': 'medium'}
    assert kw['thinking']['type'] == 'adaptive' and kw['thinking']['block_binding']['prefix_mismatch_behavior'] == 'drop_block'
    assert kw['cache_control'] == {'type': 'ephemeral'} and kw['system'] == 'charter'
    assert 'temperature' not in kw and 'budget_tokens' not in json.dumps(kw)
    assert [t['name'] for t in kw['tools']] == ['board', 'zeta'] and 'input_schema' in kw['tools'][0]
    assert res['finish'] == 'tool_calls' and res['content'] == 'Checking the board.'
    assert res['tool_calls'] == [{'id': 'toolu_1', 'name': 'board', 'arguments': '{"n": 5}'}]
    assert res['usage'] == {'prompt_tokens': 1050, 'completion_tokens': 40, 'cached_tokens': 900, 'cache_write_tokens': 50}
    assert llm.ledger.rows[-1][0][1:3] == (1050, 40) and llm.ledger.rows[-1][1]['cached'] == 900

    # the next request replays the assistant turn exactly as received, signature included
    am = assistant_message(res)
    history += [am, {'role': 'tool', 'tool_call_id': 'toolu_1', 'content': 'board rows'}]
    _, out = to_anthropic(history)
    assert out[1]['content'] == [thinking, {'type': 'text', 'text': 'Checking the board.'},
                                 {'type': 'tool_use', 'id': 'toolu_1', 'name': 'board', 'input': {'n': 5}}]
    assert out[2]['content'][0]['tool_use_id'] == 'toolu_1'
    assert 'SIG-abc' not in json.dumps({k: v for k, v in am.items() if k != 'anthropic_content'})
    assert estimate_tokens([am]) < len(json.dumps(am)) // 3            # replay blocks do not inflate the budget

    # an agent that clips old assistant text still sends its thinking and tool calls
    clipped = dict(am, content='Check ...[clipped]')
    _, out = to_anthropic([{'role': 'user', 'content': 'x'}, clipped])
    assert [b['type'] for b in out[1]['content']] == ['thinking', 'text', 'tool_use']
    assert out[1]['content'][1]['text'] == 'Check ...[clipped]'


def test_out_of_credit_pauses_the_firm_instead_of_failing():
    class Err(Exception):
        status_code = 400
    calls = []

    class Client:
        pass
    llm = _llm(None, [])

    async def stream_fail(*a, **k):
        calls.append(1)
        raise Err('Your credit balance is too low to access the Anthropic API.')
    llm._stream = stream_fail

    async def go():
        task = asyncio.create_task(llm.chat('a1', [{'role': 'user', 'content': 'x'}]))
        await asyncio.sleep(0.05)
        task.cancel()
    asyncio.run(go())
    assert calls == [1] and llm.quota_paused_until > 0


def test_config_selects_anthropic_and_haiku(monkeypatch):
    from desk import config as C
    for k in ('LLM_PROVIDER', 'MODEL_NAME', 'DESK_FLASH_MODEL', 'OPENCODE_API_KEY', 'ANTHROPIC_BASE_URL'):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'sk-ant-test')
    assert C._provider() == 'opencode'                     # a key alone never switches on pay-as-you-go billing
    monkeypatch.setenv('LLM_PROVIDER', 'anthropic')
    assert C._provider() == 'anthropic' and C._key() == 'sk-ant-test' and C._base_url() == ''
    monkeypatch.setenv('OPENCODE_API_KEY', 'oc')
    monkeypatch.setenv('LLM_PROVIDER', 'anthropic')
    monkeypatch.setenv('MODEL_NAME', 'auto')
    assert C._default_model() == C._default_flash_model() == 'claude-haiku-5-5'
    monkeypatch.setenv('MODEL_NAME', 'deepseek-v4.1-flash-free')         # another gateway's name is not sent
    assert C._default_model() == 'claude-haiku-5-5'
    monkeypatch.setenv('MODEL_NAME', 'claude-sonnet-5-5')
    assert C._default_model() == 'claude-sonnet-5-5'
    monkeypatch.setenv('LLM_PROVIDER', 'opencode')
    assert C._key() == 'oc'


def test_usd_cap_counts_cached_tokens_at_the_cache_price(tmp_path):
    import time
    from desk.company.ledger import Ledger
    from desk.company.runtime import Org
    L = Ledger(tmp_path / 'desk.sqlite3')
    L.usage('a1', 1_000_000, 100_000, 1.0, model='claude-haiku-5-5', cached=800_000)
    L.usage('a1', 1_000_000, 0, 1.0, model='deepseek-v4.1-flash-free')
    usd = Org.usd_last_hour(types.SimpleNamespace(ledger=L))
    assert abs(usd - (0.8 * 0.01 + 0.2 * 0.10 + 0.1 * 0.50)) < 1e-9
