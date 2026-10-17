"""Focused allocation review of exact existing native-venue receipts.

This profile does not design/build a new strategy or certify a simulator. The
trusted service supplies facts; each participant owns its interpretation.
"""
import copy
import json
from .store import digest


COMMON = '''Review this research allocation using the supplied verified retrospective evidence. Keep private reasoning private. Costs and source identities are authoritative. One capture, overlapping captures, fill alternatives and implementation versions are not independent confirmation. Zero trades do not establish alpha; negative results do not reject a strategy family. Displayed participation, routing latency, venue transfer and funding settlement remain assumptions. No promotion or beneficial-learning claim is available. You may disagree with the proposed stop or report insufficient evidence; never fabricate a required artifact.'''
INSTRUCTIONS = {
    'scientist': 'Decide whether to stop/redesign further work on these exact frozen configurations. If justified, call stop_mission with MEASURED_VENUE, exact replay IDs, your reason, a decisive next experiment and limitations. Otherwise explain the missing evidence; prose cannot complete a scientific decision task.',
    'falsifier': 'Independently critique the scientist allocation decision. Return one JSON object: verdict (support/disagree/insufficient), evidence (nonempty list of {replay_id, observation}), counterarguments (text), limitations (text), next_experiment (text). Cite supplied exact replay IDs. Separate facts from judgment. A completed critique is not numerical certification or agreement.',
    'historian': 'Record a reusable scoped lesson using record_mission_lesson for the supplied decision ID. Preserve the actual independent critique, including disagreement, source references, uncertainty and a decisive next experiment. Recording is not evidence that memory improves outcomes.',
}


def current_tool_contract(fitted=True):
    """Relevant current capabilities for planning as well as receipt review."""
    result = {
        'native_replay_supported_modes': ['taker'],
        'maker': 'Native replay rejects maker mode; historical maker proxies cannot qualify without trade/queue evidence.',
    }
    if fitted:
        result.update(
            prediction_units='Estimated conditional midpoint return in bps; multiplying predictions is not measured new edge.',
            cost_gate_before_threshold=True,
            threshold_and_signal_scale_can_recover_cost_gated_zero=False,
            gate='abs(predicted_bps) > 2*(taker_fee_bps+slippage_bps)+spread_bps+extra_margin_bps; only then tanh and strategy threshold.',
            inference='Worse descriptive MSE/near-zero IC on one capture is not proof of no information in a family.')
    return result


def prompt(role):
    if role not in INSTRUCTIONS:
        raise ValueError('Venue allocation review supports scientist, falsifier and historian only')
    if role == 'historian':
        return ('Record a scoped lesson from verified facts and the actual critic. Preserve disagreement, counterevidence and tool constraints. '
                'No promotion, numerical certification or learning-benefit claim. Keep private reasoning private. '
                'Call record_mission_lesson for required_mission_decision; report insufficiency instead of inventing a result.')
    return COMMON + '\n' + INSTRUCTIONS[role]


def brief(store, ident):
    from .venue_replay import load
    report = load(store, ident)['report']  # validates digest and all bound journals
    if report.get('format') != 'quantdesk-venue-replay-v1' or report.get('forward_eligible') is not False or report.get('strategy_promoted') is not False or report.get('orders_submitted_to_exchange') != 0:
        raise ValueError('Allocation review requires unpromoted retrospective native receipts')
    candidate = store.get_candidate(report['candidate_id'])
    if report['code_hash'] != digest(candidate['code']) or report['spec_hash'] != digest(candidate['spec']):
        raise ValueError('Venue allocation evidence differs from its immutable candidate')
    with store.connect() as db:
        rows = db.execute("SELECT payload FROM memories WHERE candidate=? AND kind='VENUE_REPLAY' ORDER BY id DESC LIMIT 100", (candidate['id'],)).fetchall()
    if not any(json.loads(r[0]) == {'replay_id': ident, **report} for r in rows):
        raise ValueError('Allocation review requires registered measured venue evidence')
    result = {'replay_id': ident, **{k: report[k] for k in (
        'candidate_id', 'capture_id', 'model_hash', 'code_hash', 'spec_hash',
        'rows', 'dates', 'source_state', 'execution', 'shadow_execution',
        'active_signals', 'forward_eligible', 'strategy_promoted')}}
    result['title'] = candidate['title']
    result['source_campaign'] = candidate['campaign']
    result['configuration'] = {k: candidate['spec'][k] for k in ('mode', 'holding_seconds', 'threshold', 'warmup_rows') if k in candidate['spec']}
    result['current_tool_constraints'] = current_tool_contract(bool(report.get('prediction_diagnostic')))
    result['segments'] = {k: v for k, v in report['segments'].items() if k != 'interpretation'}
    arm_keys = ('trades', 'pnl', 'fees', 'max_drawdown_fraction', 'halted',
                'partial_fills', 'open_quantity', 'cash_position_fee_reconciled')
    for arm in ('quote_arm', 'depth_arm'):
        result[arm] = {k: report[arm][k] for k in arm_keys}
    diagnostic = report.get('prediction_diagnostic')
    if diagnostic:
        result['prediction_diagnostic'] = {k: diagnostic[k] for k in (
            'extra_margin_bps', 'absolute_prediction_bps', 'modeled_cost_bps',
            'rows_above_modeled_costs', 'rows_above_costs_plus_margin')}
        outer = diagnostic['outer_prediction_diagnostics']
        result['prediction_diagnostic']['outer'] = {k: outer[k] for k in (
            'observations', 'mse_bps_squared', 'zero_prediction_mse_bps_squared', 'pearson_ic') if k in outer}
    return result


def prepare(tools, role, task, task_id=None):
    """Replace caller narratives with exact local facts before task reservation."""
    prompt(role)
    if tools.cycle_id is None:
        raise ValueError('Venue review requires a durable current cycle')
    with tools.store.connect() as db:
        cycle = db.execute('SELECT campaign,status FROM cycles WHERE id=?', (tools.cycle_id,)).fetchone()
    if not cycle or cycle['campaign'] != tools.cid or cycle['status'] == 'ABANDONED':
        raise ValueError('Venue review requires this campaign\'s unabandoned cycle')
    if task_id:
        with tools.store.connect() as db:
            previous = db.execute('SELECT campaign,agent,payload FROM tasks WHERE id=?', (task_id,)).fetchone()
        if previous:
            if previous['campaign'] != tools.cid or previous['agent'] != role:
                raise ValueError('Review checkpoint belongs to another campaign/role')
            bound = json.loads(previous['payload'])
            if bound.get('venue_review'):
                if role == 'scientist' and task.get('replay_ids') != bound.get('replay_ids'):
                    raise ValueError('Review checkpoint replay scope is immutable; use a distinct task')
                if role != 'scientist' and task.get('mission_decision_id', task.get('required_mission_decision')) != bound.get('mission_decision_id'):
                    raise ValueError('Review checkpoint decision scope is immutable; use a distinct task')
    result = {k: task[k] for k in ('task_token_limit', 'objective') if k in task}
    result['venue_review'] = True
    if role == 'scientist':
        if task.get('proposal_workflow') is not True:
            raise ValueError('Venue scientist review requires proposal_workflow=true')
        ids = task.get('replay_ids')
        result['proposal_workflow'] = True
    else:
        from .proposals import load
        ident = task.get('mission_decision_id', task.get('required_mission_decision'))
        decision = load(tools.store, ident, tools.cid)
        if decision['cycle'] != tools.cycle_id or decision['kind'] != 'MISSION_STOP' or decision.get('basis') != 'MEASURED_VENUE':
            raise ValueError('Venue review decision must be this cycle\'s native allocation stop')
        ids = [r['replay_id'] for r in decision['grounding']['observations']]
        result['mission_decision_id'] = ident
        result['mission_stop'] = {'decision_id': ident, **{k: decision[k] for k in (
            'scope', 'reason', 'next_experiment', 'limitations')}}
        if role == 'historian':
            with tools.store.connect() as db:
                review = db.execute('SELECT status,campaign,agent,result FROM tasks WHERE id=?', (tools.cycle_id + ':falsifier:mission',)).fetchone()
            if not review or review['status'] != 'COMPLETE' or review['campaign'] != tools.cid or review['agent'] != 'falsifier':
                raise ValueError('Independent mission review must complete before historian context')
            result['required_mission_decision'] = ident
            result['critique'] = json.loads(review['result'])
    if not isinstance(ids, list) or not 1 <= len(ids) <= 3 or any(not isinstance(i, str) for i in ids) or len(set(ids)) != len(ids):
        raise ValueError('Venue review needs 1..3 exact unique replay IDs')
    result['replay_ids'] = ids
    result['verified_venue_evidence'] = [brief(tools.store, ident) for ident in ids]
    configurations = [(r['candidate_id'], r['capture_id'], r['model_hash'],
                       digest(r['execution']), digest(r['shadow_execution']))
                      for r in result['verified_venue_evidence']]
    if len(set(configurations)) != len(configurations):
        raise ValueError('Repeated replay versions cannot become separate review observations')
    if role == 'historian':
        validate_review(result, result['critique'])
        # The full immutable decision/critique/grounding is retained by the
        # lesson service. Avoid repeating the scientist narrative in a receipt
        # task; keep the entire actual critic, exact IDs and material facts.
        result['mission_stop'] = {'decision_id': ident, 'scope': decision['scope']}
        compact = []
        for report in result['verified_venue_evidence']:
            evidence = {k: report[k] for k in ('replay_id', 'candidate_id', 'capture_id', 'dates', 'rows',
                'source_state', 'active_signals', 'current_tool_constraints')}
            for arm in ('quote_arm', 'depth_arm'):
                evidence[arm] = {k: report[arm][k] for k in ('trades', 'pnl', 'open_quantity')}
            if report.get('prediction_diagnostic'):
                diagnostic = report['prediction_diagnostic']
                evidence['prediction_diagnostic'] = {k: diagnostic[k] for k in (
                    'extra_margin_bps', 'modeled_cost_bps', 'rows_above_modeled_costs', 'outer')}
                evidence['prediction_diagnostic']['maximum_absolute_prediction_bps'] = diagnostic['absolute_prediction_bps']['maximum']
            compact.append(evidence)
        result['verified_venue_evidence'] = compact
    return result


def tool_specs(role, task, available):
    names = {'read_venue_replay', 'read_decision'}
    if role == 'scientist': names.add('stop_mission')
    if role == 'historian': names.add('record_mission_lesson')
    selected = copy.deepcopy([t for t in available if t['function']['name'] in names])
    for tool in selected:
        if tool['function']['name'] == 'stop_mission':
            params = tool['function']['parameters']
            params['properties'] = {k: v for k, v in params['properties'].items() if k in (
                'basis', 'replay_ids', 'reason', 'next_experiment', 'limitations')}
            params['properties']['basis']['enum'] = ['MEASURED_VENUE']
            params['properties']['replay_ids']['items'] = {'type': 'string', 'enum': task['replay_ids']}
            params['required'] = ['basis', 'replay_ids', 'reason', 'next_experiment', 'limitations']
    return selected


def validate_call(task, name, args):
    if name == 'stop_mission' and (args.get('basis') != 'MEASURED_VENUE' or
            not isinstance(args.get('replay_ids'), list) or
            not set(args['replay_ids']) <= set(task['replay_ids'])):
        raise ValueError('This review can ground only the supplied native replay IDs')
    if name == 'read_venue_replay' and args.get('replay_id') not in task['replay_ids']:
        raise ValueError('This review reads only its exact replay scope')
    if name in {'read_decision', 'record_mission_lesson'} and args.get('decision_id') != task.get('mission_decision_id'):
        raise ValueError('This review reads/records only its exact mission decision')


def validate_review(task, result):
    if result.get('verdict') not in {'support', 'disagree', 'insufficient'}:
        raise ValueError('Critique requires verdict support/disagree/insufficient')
    for key in ('counterarguments', 'limitations', 'next_experiment'):
        if not isinstance(result.get(key), str) or not result[key].strip():
            raise ValueError('Critique requires nonempty ' + key)
    evidence = result.get('evidence')
    if not isinstance(evidence, list) or not evidence or any(
            not isinstance(r, dict) or r.get('replay_id') not in task['replay_ids'] or
            not isinstance(r.get('observation'), str) or not r['observation'].strip()
            for r in evidence):
        raise ValueError('Critique requires observations pointing to supplied exact replay IDs')
