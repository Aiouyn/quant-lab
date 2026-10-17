"""MiMo gateway: async streaming chat with tool calls, reasoning passback, AIMD adaptive concurrency,
priority scheduling (pods with more budget weight get slots first) and token accounting."""
from __future__ import annotations

import asyncio
import heapq
import itertools
import json
import time

from openai import AsyncOpenAI


class Throttle:
    """Additive-increase / multiplicative-decrease concurrency limit with a priority wait queue."""

    def __init__(self, start=8, cap=32, floor=2, ledger=None):
        self.limit = float(start)
        self.cap, self.floor = cap, floor
        self.inflight = 0
        self.waiters = []
        self.seq = itertools.count()
        self.last_change = time.time()
        self.last_throttle = 0.0
        self.ledger = ledger
        self.errors = 0

    async def acquire(self, priority=1.0):
        fut = asyncio.get_running_loop().create_future()
        heapq.heappush(self.waiters, (-priority, next(self.seq), fut))
        self._dispatch()
        await fut

    def _dispatch(self):
        while self.waiters and self.inflight < int(self.limit):
            _, _, fut = heapq.heappop(self.waiters)
            if not fut.done():
                self.inflight += 1
                fut.set_result(True)

    def release(self, ok=True, throttled=False):
        self.inflight -= 1
        now = time.time()
        if throttled:
            self.limit = max(self.floor, self.limit * 0.6)
            self.last_throttle = now
            self.last_change = now
        elif ok and now - self.last_change > 45 and now - self.last_throttle > 90:
            self.limit = min(self.cap, self.limit + 2)
            self.last_change = now
        if self.ledger is not None:
            self.ledger.set('concurrency', {'limit': round(self.limit, 1), 'inflight': self.inflight,
                                            'waiting': len(self.waiters), 'cap': self.cap})
        self._dispatch()


class LLM:
    def __init__(self, settings, ledger, start=None, cap=None):
        self.s = settings
        self.ledger = ledger
        self.client = AsyncOpenAI(api_key=settings.api_key, base_url=settings.base_url, max_retries=0,
                                  timeout=settings.request_timeout_s)
        self.throttle = Throttle(start or settings.start_concurrency, cap or settings.max_concurrency, ledger=ledger)

    async def chat(self, agent, messages, tools=None, priority=1.0, max_tokens=None, thinking=True, model=None):
        """Returns dict(content, reasoning, tool_calls[list], finish, usage). Raises after retries.
        A '429 quota exhausted' pauses the whole firm (probe every 15 min) without consuming retry attempts."""
        delay = 5.0
        attempt = 0
        while True:
            await self.quota_gate(agent)                  # never hold a slot while the plan quota is exhausted
            await self.throttle.acquire(priority)
            t0 = time.time()
            throttled = ok = False
            try:
                res = await asyncio.wait_for(self._stream(messages, tools, max_tokens, thinking, model), self.s.request_timeout_s)
                ok = True
                u = res['usage']
                self.ledger.usage(agent, u.get('prompt_tokens', 0), u.get('completion_tokens', 0), time.time() - t0,
                                  model=model or self.s.model, cached=u.get('cached_tokens', 0))
                return res
            except Exception as e:
                status = getattr(e, 'status_code', None)
                name = type(e).__name__
                self.ledger.usage(agent, 0, 0, time.time() - t0, f'error:{name}:{status}', model=model or self.s.model)
                if status == 429 and 'quota' in str(e).lower():
                    if time.time() >= getattr(self, 'quota_paused_until', 0):
                        self.quota_paused_until = time.time() + 900
                        self.ledger.set('quota', {'exhausted_at': time.time(), 'next_probe': self.quota_paused_until})
                        self.ledger.event('llm', 'QUOTA_EXHAUSTED', {'agent': agent})
                    continue                              # finally releases the slot; the gate then waits
                throttled = status in (429, 503) or 'RateLimit' in name
                retryable = throttled or (isinstance(status, int) and status >= 500) or name in (
                    'APITimeoutError', 'APIConnectionError', 'TimeoutError', 'ReadTimeout', 'RemoteProtocolError',
                    'StreamStalled', 'IncompleteRead', 'ReadError', 'ConnectError', 'WriteError', 'ConnectionResetError',
                    'RemoteDisconnected', 'ProtocolError')
                attempt += 1
                if not retryable or attempt >= 6:
                    raise
                await asyncio.sleep(delay * (1.5 if throttled else 1.0))
                delay = min(delay * 2, 120)
            finally:
                self.throttle.release(ok=ok, throttled=throttled)

    async def quota_gate(self, agent):
        """While the plan quota is exhausted, everyone waits; the first caller after the pause acts as the probe."""
        while time.time() < getattr(self, 'quota_paused_until', 0):
            await asyncio.sleep(min(60, self.quota_paused_until - time.time() + 1))
        if getattr(self, 'quota_paused_until', 0) and not getattr(self, '_probe_logged', 0) == self.quota_paused_until:
            self._probe_logged = self.quota_paused_until
            self.ledger.event('llm', 'QUOTA_PROBE', {'agent': agent})

    async def _stream(self, messages, tools, max_tokens, thinking, model=None):
        kw = dict(model=model or self.s.model, messages=messages, max_completion_tokens=max_tokens or self.s.max_completion_tokens,
                  stream=True, stream_options={'include_usage': True})
        if tools:
            kw['tools'] = tools
            kw['tool_choice'] = 'auto'
        if thinking:
            kw['extra_body'] = {'thinking': {'type': 'enabled'}}
        content, reason, calls = [], [], {}
        usage, finish = None, None
        stream = await self.client.chat.completions.create(**kw)
        last = time.time()
        async for ch in stream:
            if time.time() - last > 180:
                raise StreamStalled('no tokens for 180s')
            last = time.time()
            if ch.usage:
                usage = ch.usage
            for choice in ch.choices:
                d = choice.delta
                if d is None:
                    continue
                if d.content:
                    content.append(d.content)
                rc = getattr(d, 'reasoning_content', None) or (d.model_extra or {}).get('reasoning_content')
                if rc:
                    reason.append(rc)
                for tc in d.tool_calls or []:
                    e = calls.setdefault(tc.index, {'id': None, 'name': '', 'arguments': ''})
                    if tc.id:
                        e['id'] = tc.id
                    if tc.function and tc.function.name:
                        e['name'] += tc.function.name
                    if tc.function and tc.function.arguments:
                        e['arguments'] += tc.function.arguments
                if choice.finish_reason:
                    finish = choice.finish_reason
        det = getattr(usage, 'prompt_tokens_details', None)
        u = {'prompt_tokens': getattr(usage, 'prompt_tokens', 0) or 0,
             'completion_tokens': getattr(usage, 'completion_tokens', 0) or 0,
             'cached_tokens': (getattr(det, 'cached_tokens', 0) or 0) if det is not None else 0}
        return {'content': ''.join(content), 'reasoning': ''.join(reason),
                'tool_calls': [v for _, v in sorted(calls.items()) if v['name']], 'finish': finish, 'usage': u}


class StreamStalled(TimeoutError):
    pass


def assistant_message(res):
    """Serialize an assistant turn for the next request (MiMo needs reasoning_content passed back)."""
    m = {'role': 'assistant', 'content': res['content'] or (None if (res['tool_calls'] or res.get('reasoning')) else '(no output)')}
    if res.get('reasoning'):
        m['reasoning_content'] = res['reasoning']
    if res['tool_calls']:
        m['tool_calls'] = [{'id': c['id'] or f'call_{i}', 'type': 'function',
                            'function': {'name': c['name'], 'arguments': c['arguments'] or '{}'}}
                           for i, c in enumerate(res['tool_calls'])]
    return m


def estimate_tokens(messages):
    return len(json.dumps(messages, ensure_ascii=False)) // 3
