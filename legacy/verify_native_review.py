"""Actual frozen evidence through the review runtime with operator-authored responses.

No provider is contacted. A temporary SQLite backup and copies of the bound
replay journals isolate all task/decision/lesson mutations from production.
"""
import argparse
import hashlib
import json
import shutil
import sqlite3
import tempfile
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from quantdesk.agents import AgentRuntime, ResearchTools, task_system_prompt
from quantdesk.mission_review import prompt, tool_specs
from quantdesk.research_map import research_map
from quantdesk.settings import Settings
from quantdesk.store import Store
from quantdesk.venue_replay import load

CID = '2b22519842f6'
REPLAY = 'b691d347f0ef4cf4726bfdcf84acd08206739d975a10d7950e461197205ce819'


def snapshot(store):
    queries = {
        'campaign': 'SELECT id,status,token_limit FROM campaigns WHERE id=?',
        'requests': 'SELECT id,status,reserved,actual FROM requests WHERE campaign=? ORDER BY created',
        'cycles': 'SELECT id,status FROM cycles WHERE campaign=? ORDER BY created',
        'decisions': 'SELECT id FROM research_decisions WHERE campaign=? ORDER BY id',
    }
    with store.connect() as db:
        result = {k: [dict(r) for r in db.execute(q, (CID,))] for k, q in queries.items()}
        for table in ('candidates', 'holdouts', 'forward_plans', 'mission_lessons', 'data_exposures'):
            result[table] = [dict(r) for r in db.execute('SELECT * FROM ' + table)]
    return result


def request_reservations(store, tasks):
    with store.connect() as db:
        return [dict(r) for r in db.execute(
            'SELECT requests.agent,requests.reserved,requests.actual FROM requests '
            'JOIN request_tasks ON request_tasks.request_id=requests.id '
            'WHERE request_tasks.task_id IN (' + ','.join('?' for _ in tasks) + ')', tasks)]


class PublicMessage:
    def __init__(self, content=None, call=None):
        self.content = content
        self.tool_calls = [call] if call else None

    def model_dump(self, **kwargs):
        return {'role': 'assistant', 'content': self.content}


def response(name=None, args=None, content=None):
    call = SimpleNamespace(id='operator-authored', function=SimpleNamespace(
        name=name, arguments=json.dumps(args))) if name else None
    return SimpleNamespace(usage=SimpleNamespace(total_tokens=0), choices=[
        SimpleNamespace(message=PublicMessage(content, call), finish_reason='stop')])


def verify():
    settings = Settings()
    production = Store(settings.home)
    before = snapshot(production)
    original = load(production, REPLAY)['report']
    recorded_requests = []
    with tempfile.TemporaryDirectory(prefix='quantlab-native-review-') as temporary:
        directory = Path(temporary)
        with production.connect() as source:
            target = sqlite3.connect(directory / 'desk.sqlite3')
            source.backup(target)
            target.close()
        shutil.copytree(production.home / 'venue_replays' / REPLAY, directory / 'venue_replays' / REPLAY)
        local = Store(directory)
        cycle = local.start_cycle(CID)
        cfg = replace(settings, home=directory, max_agent_steps=1, max_completion_tokens=1000)
        tools = ResearchTools(CID, local, None, None, cfg, cycle)
        diagnostic = original['prediction_diagnostic']
        high = diagnostic['absolute_prediction_bps']['maximum']
        low = diagnostic['modeled_cost_bps']['minimum']
        assert high < low and original['active_signals'] == 0
        reasoning = f'This exact frozen configuration produced no eligible signals: max absolute prediction {high} bps below minimum modeled costs {low} bps before extra margin. Margin reduction alone cannot repair this observed capture.'
        followup = 'Predeclare a different supported mechanism and an outcome-blind cash/ablation comparison on other permitted development observations; keep costs unchanged.'
        limits = 'One retrospective capture; models, venue distribution, execution assumptions and missing funding limit inference. No class verdict, untouched confirmation, calibrated fills or beneficial-learning proof.'
        critique = {'verdict': 'support', 'evidence': [{'replay_id': REPLAY, 'observation': reasoning}],
            'counterarguments': 'Other data, regimes or mechanisms may differ; this observation cannot establish that a model family lacks signal.',
            'limitations': limits, 'next_experiment': followup}

        def authored(**kwargs):
            # This is deliberately an operator-authored fake transport, not MiMo.
            task = json.loads(kwargs['messages'][1]['content'])
            role = 'historian' if task.get('required_mission_decision') else 'falsifier' if task.get('mission_stop') else 'scientist'
            recorded_requests.append({'role': role,
                'serialized_bytes': len(json.dumps({'messages': kwargs['messages'], 'tools': kwargs['tools']}).encode()),
                'tools': [t['function']['name'] for t in kwargs['tools']],
                'thinking_enabled': kwargs['extra_body']['thinking']['type'] == 'enabled'})
            if role == 'scientist':
                return response('stop_mission', {'basis': 'MEASURED_VENUE', 'replay_ids': [REPLAY],
                    'reason': reasoning, 'next_experiment': followup, 'limitations': limits})
            if role == 'falsifier': return response(content=json.dumps(critique))
            return response('record_mission_lesson', {'decision_id': task['required_mission_decision'],
                'lesson': reasoning + ' The operator-authored critique limits this to the exact capture/configuration.',
                'next_experiment': followup})

        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=authored)))
        runtime = AgentRuntime(local, cfg, client=client, model='operator-authored-verification', provider='mimo')
        scientist = cycle + ':scientist:native-review'
        task = {'venue_review': True, 'proposal_workflow': True, 'replay_ids': [REPLAY]}
        decision = runtime.run(CID, 'scientist', task, tools, scientist)
        scoped = {'venue_review': True, 'mission_decision_id': decision['decision_id']}
        runtime.run(CID, 'falsifier', scoped, tools, cycle + ':falsifier:mission')
        runtime.run(CID, 'historian', scoped, tools, cycle + ':historian:mission')
        local.complete_cycle(cycle, 'MISSION_STOP')
        count = len(recorded_requests)
        assert runtime.run(CID, 'scientist', task, tools, scientist)['decision_id'] == decision['decision_id']
        assert runtime.run(CID, 'historian', scoped, tools, cycle + ':historian:mission')['artifact_receipt']
        assert len(recorded_requests) == count == 3
        with local.connect() as db:
            lesson = json.loads(db.execute('SELECT payload FROM mission_lessons WHERE decision_id=?', (decision['decision_id'],)).fetchone()[0])
            payload = json.loads(db.execute('SELECT payload FROM tasks WHERE id=?', (scientist,)).fetchone()[0])
        request_rows = request_reservations(local, [scientist, cycle + ':falsifier:mission', cycle + ':historian:mission'])
        assert len(request_rows) == 3 and all(r['actual'] == 0 for r in request_rows)
        assert lesson['grounding']['decision']['grounding']['observations'][0]['observed_venue_replay'] == original
        assert lesson['grounding']['independent_critique'] == critique
        mapped = research_map(local, CID)
        assert mapped['reviewed_mission_lessons'] == 1
        assert any(r['kind'] == 'MISSION_LESSON' for r in local.recall(4))
        scoped_tools = [t for t in tool_specs('scientist', payload, tools.available('scientist')) if t['function']['name'] == 'stop_mission']
        ordinary_tools = [t for t in tools.available('scientist') if t['function']['name'] in {'submit_proposal', 'stop_mission'}]
        comparison = {}
        for name, system, schemas in [('focused', prompt('scientist'), scoped_tools),
                                      ('ordinary', task_system_prompt('scientist', True), ordinary_tools)]:
            comparison[name] = len(json.dumps({'messages': [{'role': 'system', 'content': system},
                {'role': 'user', 'content': json.dumps(payload)}], 'tools': schemas}).encode())
    assert snapshot(production) == before
    assert load(production, REPLAY)['report'] == original
    remaining = before['campaign'][0]['token_limit'] - sum(r['actual'] if r['actual'] is not None else r['reserved'] for r in before['requests'])
    return {'completed_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'Actual frozen native facts and runtime/services; entirely operator-authored provider responses in a temporary ledger. No real agent judgment, company acceptance or learning-benefit claim.',
        'replay_id': REPLAY, 'source_campaign': lesson['grounding']['decision']['grounding']['observations'][0]['source_campaign'],
        'exact_grounding_and_critique_preserved': True, 'checkpoint_reuse_without_extra_calls': True,
        'production_snapshot_unchanged': True, 'new_production_provider_requests': 0,
        'orders_submitted_to_exchange': 0, 'original_remaining_tokens': remaining,
        'synthetic_request_footprints': recorded_requests, 'synthetic_reservations': request_rows,
        'controlled_same_payload_bytes': comparison,
        'worst_case_sum_of_synthetic_reservations': sum(r['reserved'] for r in request_rows),
        'real_acceptance': False,
        'implementation': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in map(Path, (
            'quantdesk/mission_review.py', 'quantdesk/agents.py', 'quantdesk/proposals.py', 'tests/test_mission_review.py'))}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--recheck', action='store_true')
    args = parser.parse_args()
    out = Settings().home / 'verification' / 'native-review' / 'offline-service.json'
    if out.exists() and not args.recheck:
        print(out.read_text(encoding='utf-8'))
    else:
        value = verify()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(value, indent=2))
