"""The LLM model is picked at startup from a preference list against the gateway's live model list."""
from __future__ import annotations

import asyncio
import dataclasses
import types

from desk.company.llm import SAFE_FALLBACK_MODEL, resolve_models
from desk.config import SETTINGS


class _Client:
    def __init__(self, ids=None, fail=False):
        self.ids, self.fail = ids or [], fail
        self.models = self

    async def list(self):
        if self.fail:
            raise RuntimeError('401')
        return types.SimpleNamespace(data=[types.SimpleNamespace(id=i) for i in self.ids])


def _s(model='auto', flash='auto'):
    return dataclasses.replace(SETTINGS, model=model, flash_model=flash,
                               model_preference=('minimax-m3.1-flash-free', 'deepseek-v4.1-flash-free'))


def test_first_listed_preference_wins():
    s = asyncio.run(resolve_models(_s(), client=_Client(['big-pickle', 'minimax-m3.1-flash-free',
                                                         'deepseek-v4.1-flash-free'])))
    assert s.model == s.flash_model == 'minimax-m3.1-flash-free'


def test_skips_models_that_left_the_list():
    s = asyncio.run(resolve_models(_s(), client=_Client(['deepseek-v4.1-flash-free'])))
    assert s.model == 'deepseek-v4.1-flash-free'


def test_list_failure_falls_back_and_pinned_models_are_kept():
    s = asyncio.run(resolve_models(_s(), client=_Client(fail=True)))
    assert s.model == SAFE_FALLBACK_MODEL
    pinned = _s(model='minimax-m3', flash='minimax-m3')
    assert asyncio.run(resolve_models(pinned, client=_Client(fail=True))) is pinned
