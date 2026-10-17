"""One distinct, idempotent real native allocation-review protocol.

Uses only the unchanged original campaign allowance. One request per role,
no retries or replacement responses. A partial run is not a completed mission.
"""
import json
import os
import shutil
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from quantdesk.agents import AgentRuntime, ResearchTools
from quantdesk.mission_review import brief
from quantdesk.proposals import load
from quantdesk.research_map import research_map
from quantdesk.settings import Settings
from quantdesk.store import Store, process_alive

CID = '2b22519842f6'
REPLAY = 'b691d347f0ef4cf4726bfdcf84acd08206739d975a10d7950e461197205ce819'
CAP = 750000
COMPLETIONS = {'scientist': 1800, 'falsifier': 1400, 'historian': 1400}
TASK_LIMIT = 10000


def snapshot(store):
    with store.connect() as db:
        usage = dict(db.execute('SELECT count(*) requests,coalesce(sum(actual),0) known_reported_tokens,'
            'coalesce(sum(coalesce(actual,reserved)),0) charged_tokens FROM requests WHERE campaign=?', (CID,)).fetchone())
        campaign = dict(db.execute('SELECT id,status,token_limit FROM campaigns WHERE id=?', (CID,)).fetchone())
        protected = {}
        for table in ('candidates', 'holdouts', 'forward_plans', 'data_exposures'):
            protected[table] = [dict(r) for r in db.execute('SELECT * FROM ' + table)]
        cycles = [dict(r) for r in db.execute('SELECT id,status FROM cycles WHERE campaign=? ORDER BY created', (CID,))]
    return {'campaign': campaign, 'usage': usage, 'protected': protected, 'cycles': cycles}


def execute(settings, store, path, runtime_factory=AgentRuntime):
    path = Path(path)
    if path.exists():
        value = json.loads(path.read_text(encoding='utf-8'))
        # No automatic restart of a live, failed, interrupted or completed scope.
        return {'status': value['status'], 'pid': value['pid'],
                'process_alive': process_alive(value['pid']), 'new_request_started': False,
                'receipt': str(path.resolve())}
    before = snapshot(store)
    if before['campaign']['token_limit'] != CAP:
        raise ValueError('The original campaign cap must remain unchanged')
    remaining = CAP - before['usage']['charged_tokens']
    if remaining <= 0:
        raise ValueError('No remaining original allowance; this protocol cannot reset it')
    with store.connect() as db:
        if db.execute("SELECT 1 FROM tasks WHERE campaign=? AND status='RUNNING'", (CID,)).fetchone():
            raise ValueError('An existing research task still owns this campaign')
        if db.execute("SELECT 1 FROM cycles WHERE campaign=? AND status IN ('ACTIVE','PAUSED','FAILED')", (CID,)).fetchone():
            raise ValueError('This distinct review cannot reuse an unfinished company cycle')
    measured = brief(store, REPLAY)  # verify facts/journals/registration before setup
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {'status': 'PREPARING', 'pid': os.getpid(), 'campaign': CID,
        'started_utc': datetime.now(timezone.utc).isoformat(), 'before': before,
        'scope': 'One preselected existing native configuration allocation review; operator supplies the assignment, real agents own decision/critique/lesson. No director, new code/replay, strategy discovery or company-cohort acceptance.',
        'criteria': {'maximum_new_requests': 3, 'maximum_requests_per_role': 1,
            'completion_limits': COMPLETIONS, 'task_token_limit': TASK_LIMIT,
            'request_timeout_seconds': 180, 'cooperative_seconds': 600,
            'original_campaign_limit': CAP, 'starting_remaining_tokens': remaining,
            'budget_increase': False, 'retry': False,
            'completion': 'Exact measured stop, completed separately sourced critique, immutable grounded lesson and DECISION_COMPLETE cycle. Partial work cannot qualify.'},
        'replay_id': REPLAY, 'verified_preflight_evidence': measured,
        'budget_limit': 'A complete three-role run is not guaranteed to fit. Each next reservation must fit actual remaining campaign/task allowance; ambiguous usage remains charged.'}
    with path.open('x', encoding='utf-8') as stream:
        json.dump(state, stream, indent=2)

    def persist():
        state['updated_utc'] = datetime.now(timezone.utc).isoformat()
        stage = path.with_suffix('.partial')
        stage.write_text(json.dumps(state, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        stage.replace(path)

    cycle = None
    try:
        with store.lease('research', seconds=900), store.research_deadline(CID, time.monotonic() + 600):
            cycle = store.start_cycle(CID)
            state.update(cycle_id=cycle, status='RUNNING', phase='scientist')
            persist()
            store.event(CID, 'operator', 'NATIVE_REVIEW_PROTOCOL_PLANNED', {
                'cycle_id': cycle, 'replay_id': REPLAY, 'criteria': state['criteria'], 'scope': state['scope']})
            decision_id = None
            for role in ('scientist', 'falsifier', 'historian'):
                configured = replace(settings, max_agent_steps=1, max_completion_tokens=COMPLETIONS[role], request_timeout=180)
                tools = ResearchTools(CID, store, None, None, configured, cycle)
                agent = runtime_factory(store, configured)
                if agent.provider != 'mimo':
                    raise ValueError('This protocol was declared for the configured MiMo provider')
                task_id = cycle + (':scientist:native-review-v1' if role == 'scientist' else ':' + role + ':mission')
                task = {'venue_review': True, 'task_token_limit': TASK_LIMIT}
                if role == 'scientist':
                    task.update(proposal_workflow=True, replay_ids=[REPLAY],
                        objective='Assess whether further allocation to this exact frozen configuration is justified by its observed native results. Decide a scoped stop/redesign if supported, with a decisive next experiment. You may find the evidence insufficient; do not assume a stop or family verdict is required.')
                else:
                    task['mission_decision_id'] = decision_id
                state.update(phase=role, current_task_id=task_id, model=agent.model, provider=agent.provider)
                persist()
                result = agent.run(CID, role, task, tools, task_id)
                state[role + '_result'] = result
                if role == 'scientist':
                    decision_id = result['decision_id']
                    decision = load(store, decision_id, CID)
                    if decision['cycle'] != cycle or decision['task_id'] != task_id or decision.get('basis') != 'MEASURED_VENUE':
                        raise ValueError('Real decision does not belong to this native assignment')
                    state['decision_id'] = decision_id
                persist()
            with store.connect() as db:
                lesson = db.execute('SELECT payload FROM mission_lessons WHERE decision_id=?', (decision_id,)).fetchone()
                if not lesson: raise ValueError('No immutable reviewed lesson was recorded')
            state['reviewed_lesson'] = json.loads(lesson['payload'])
            store.complete_cycle(cycle, 'MISSION_STOP')
            outcome = {'cycle_id': cycle, 'kind': 'MISSION_STOP', 'decision_id': decision_id, 'candidates': [],
                       'scope': 'Real native allocation review; operator-predeclared mission, no full research-company cohort'}
            store.event(CID, 'desk', 'MISSION_DECISION_COMPLETED', outcome)
            state.update(status='COMPLETE', outcome=outcome,
                interpretation='Real scientist/critic/historian artifact protocol completed. No profitable system, independent simulation certification, beneficial learning or full autonomous company acceptance.')
    except BaseException as error:
        state.update(status='FAILED', error_type=type(error).__name__, error=str(error),
            interpretation='This separately bounded review failed or is partial. Prior commits remain evidence; no completed review/company or economic outcome is claimed. No automatic retry.')
        if cycle:
            store.suspend_cycle(CID, 'FAILED', 'Distinct native review protocol did not complete')
    finally:
        after = snapshot(store)
        state['after'] = after
        state['new_provider_requests'] = after['usage']['requests'] - before['usage']['requests']
        state['new_known_reported_tokens'] = after['usage']['known_reported_tokens'] - before['usage']['known_reported_tokens']
        state['new_charged_tokens'] = after['usage']['charged_tokens'] - before['usage']['charged_tokens']
        state['remaining_original_tokens'] = CAP - after['usage']['charged_tokens']
        state['protected_unchanged'] = after['protected'] == before['protected']
        state['original_cycles_unchanged'] = all(c in after['cycles'] for c in before['cycles'])
        state['orders_submitted_to_exchange'] = 0
        state['completed_utc'] = datetime.now(timezone.utc).isoformat()
        persist()
    return state


def finish_unrequested_historian(settings, store, path, runtime_factory=AgentRuntime):
    """Finish the one remaining request, never retry an already requested role.

    This explicit operator continuation preserves the initial failed receipt and
    all original request/task/campaign caps. It is not unattended acceptance.
    """
    path = Path(path)
    state = json.loads(path.read_text(encoding='utf-8'))
    if state.get('continuation'):
        return {'status': state['status'], 'new_request_started': False, 'receipt': str(path.resolve())}
    if state['status'] != 'FAILED' or state.get('phase') != 'historian' or state['campaign'] != CID:
        raise ValueError('Continuation requires this protocol\'s uncompleted historian phase')
    cycle = state['cycle_id']
    task_id = cycle + ':historian:mission'
    with store.connect() as db:
        requests = [dict(r) for r in db.execute('SELECT requests.agent,requests.actual FROM requests '
            'JOIN request_tasks ON request_tasks.request_id=requests.id WHERE request_tasks.task_id LIKE ?', (cycle + ':%',))]
        tasks = {r['agent']: dict(r) for r in db.execute('SELECT id,campaign,agent,status,result FROM tasks WHERE id LIKE ?', (cycle + ':%',))}
        current_cycle = db.execute('SELECT campaign,status FROM cycles WHERE id=?', (cycle,)).fetchone()
    if sorted(r['agent'] for r in requests) != ['falsifier', 'scientist'] or any(r['actual'] is None for r in requests):
        raise ValueError('Exactly one completed charged request per prior role and zero historian requests are required')
    if not current_cycle or current_cycle['campaign'] != CID or current_cycle['status'] != 'FAILED':
        raise ValueError('This exact interrupted review cycle is required')
    if any(tasks.get(role, {}).get('status') != 'COMPLETE' for role in ('scientist', 'falsifier')):
        raise ValueError('Both prior artifacts must already be complete')
    decision = load(store, state['decision_id'], CID)
    if decision['cycle'] != cycle or decision.get('basis') != 'MEASURED_VENUE' or [r['replay_id'] for r in decision['grounding']['observations']] != [REPLAY]:
        raise ValueError('Existing decision differs from the declared native assignment')
    before = snapshot(store)
    if before['campaign']['token_limit'] != CAP or before['protected'] != state['before']['protected']:
        raise ValueError('Original cap and protected state must remain intact')
    backup = path.with_name('real-run-initial-reservation-failure.json')
    if backup.exists():
        raise ValueError('Initial failure archive already exists; inspect rather than restart')
    shutil.copyfile(path, backup)
    state['continuation'] = {'status': 'PREPARING', 'pid': os.getpid(),
        'started_utc': datetime.now(timezone.utc).isoformat(), 'initial_failure_receipt': str(backup.resolve()),
        'maximum_new_requests': 1, 'prior_requests_repeated': False, 'historian_completion_limit': 1000,
        'reason': 'Prior historian reservation failed before any provider call. Use a focused receipt context retaining the full actual critic and trusted cost-gate/native-mode constraints.',
        'scope': 'Explicit operator intervention; completes remaining artifact stage, not unattended company or beneficial-learning acceptance.'}
    state['status'] = 'CONTINUING_UNREQUESTED_HISTORIAN'
    state['pid'] = os.getpid()

    def persist():
        state['updated_utc'] = datetime.now(timezone.utc).isoformat()
        stage = path.with_suffix('.partial')
        stage.write_text(json.dumps(state, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        stage.replace(path)

    persist()
    try:
        with store.lease('research', seconds=360), store.research_deadline(CID, time.monotonic() + 240):
            with store.connect() as db:
                db.execute("UPDATE cycles SET status='ACTIVE' WHERE id=? AND campaign=? AND status='FAILED'", (cycle, CID))
            configured = replace(settings, max_agent_steps=1, max_completion_tokens=1000, request_timeout=180)
            tools = ResearchTools(CID, store, None, None, configured, cycle)
            agent = runtime_factory(store, configured)
            if agent.provider != 'mimo': raise ValueError('Configured MiMo provider required')
            result = agent.run(CID, 'historian', {'venue_review': True, 'task_token_limit': TASK_LIMIT,
                'mission_decision_id': state['decision_id']}, tools, task_id)
            with store.connect() as db:
                row = db.execute('SELECT payload FROM mission_lessons WHERE decision_id=?', (state['decision_id'],)).fetchone()
            if not row: raise ValueError('No reviewed mission lesson was committed')
            state['historian_result'] = result
            state['reviewed_lesson'] = json.loads(row['payload'])
            if state['reviewed_lesson']['grounding']['independent_critique'] != json.loads(tasks['falsifier']['result']):
                raise ValueError('Actual independent critique was not preserved')
            after = snapshot(store)
            if after['protected'] != before['protected'] or after['campaign']['token_limit'] != CAP:
                raise ValueError('Protected state or original budget changed')
            store.complete_cycle(cycle, 'MISSION_STOP')
            outcome = {'cycle_id': cycle, 'kind': 'MISSION_STOP', 'decision_id': state['decision_id'], 'candidates': [],
                'scope': 'Real native three-role review with explicit operator context repair; no full autonomous company cohort'}
            store.event(CID, 'desk', 'MISSION_DECISION_COMPLETED', outcome)
            state.update(status='COMPLETE_WITH_INTERVENTION', outcome=outcome,
                interpretation='Three real role artifacts completed with an operator context/constraint repair before the previously unrequested historian call. Original scientist/critic guidance remains inspectable. No unattended-company, profitable system, simulation certification or beneficial-learning acceptance.')
            state['continuation']['status'] = 'COMPLETE'
            state['map_reviewed_mission_lessons'] = research_map(store, CID)['reviewed_mission_lessons']
    except BaseException as error:
        state.update(status='FAILED', error_type=type(error).__name__, error=str(error))
        state['continuation'].update(status='FAILED', error_type=type(error).__name__)
        store.suspend_cycle(CID, 'FAILED', 'Unrequested historian continuation did not complete; no retry')
    finally:
        after = snapshot(store)
        state['after'] = after
        state['new_provider_requests'] = after['usage']['requests'] - state['before']['usage']['requests']
        state['new_known_reported_tokens'] = after['usage']['known_reported_tokens'] - state['before']['usage']['known_reported_tokens']
        state['new_charged_tokens'] = after['usage']['charged_tokens'] - state['before']['usage']['charged_tokens']
        state['remaining_original_tokens'] = CAP - after['usage']['charged_tokens']
        state['protected_unchanged'] = after['protected'] == state['before']['protected']
        state['original_cycles_unchanged'] = all(c in after['cycles'] for c in state['before']['cycles'])
        state['continuation']['new_provider_requests'] = after['usage']['requests'] - before['usage']['requests']
        state['continuation']['completed_utc'] = datetime.now(timezone.utc).isoformat()
        state['completed_utc'] = datetime.now(timezone.utc).isoformat()
        persist()
    return state


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--finish-unrequested-historian', action='store_true')
    args = parser.parse_args()
    settings = Settings()
    path = settings.home / 'verification' / 'native-review' / 'real-run.json'
    action = finish_unrequested_historian if args.finish_unrequested_historian else execute
    state = action(settings, Store(settings.home), path)
    print(json.dumps({k: state[k] for k in ('status', 'phase', 'decision_id', 'new_provider_requests',
        'new_known_reported_tokens', 'new_charged_tokens', 'remaining_original_tokens',
        'protected_unchanged', 'original_cycles_unchanged', 'error_type') if k in state}, indent=2))
