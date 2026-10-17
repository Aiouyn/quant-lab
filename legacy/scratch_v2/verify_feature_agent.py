"""Real MiMo integration: authored causal feature draft, explicitly untested.

This single-role capability check does not claim a complete autonomous campaign
or profitable strategy. Replay follows under the normal research lease.
"""
import json
import os
from dataclasses import replace
from pathlib import Path
from quantdesk.settings import Settings
from quantdesk.store import Store
from quantdesk.data import Catalog
from quantdesk.evaluation import Evaluator
from quantdesk.agents import AgentRuntime,ResearchTools


class DraftTools(ResearchTools):
    def available(self,role):
        # Drafting may profile earlier data and measure descriptive features.
        # Expensive replay is staged after the active full-day benchmark.
        allowed={'inspect_data','measure_feature','recall','research_map',
                 'list_candidates','read_candidate','register_candidate'}
        return [tool for tool in super().available(role) if tool['function']['name'] in allowed]


def main():
    settings=replace(Settings(),max_completion_tokens=18000,max_agent_steps=3,request_timeout=900)
    store=Store(settings.home); catalog=Catalog(settings.data_dir,settings.home)
    directory=settings.home/'verification'/'feature-agent'; directory.mkdir(parents=True,exist_ok=True)
    path=directory/'state.json'
    if path.exists():
        state=json.loads(path.read_text())
        if state.get('status')=='COMPLETE':
            print(json.dumps(state,indent=2)); return
    else:
        cid=store.campaign(catalog.plan(),160000)
        state={'campaign':cid,'status':'RUNNING','phase':'Real MiMo feature-program capability check',
               'replay_verified':False,'profitable_strategy_claim':False}
    state.update(pid=os.getpid(),status='RUNNING')
    state['attempts']=state.get('attempts',1)+1 if state.get('error_type') else state.get('attempts',1)
    path.write_text(json.dumps(state,indent=2),encoding='utf-8')
    tools=DraftTools(state['campaign'],store,catalog,Evaluator(store,catalog,settings),settings,'feature-capability')
    task={'mission':'Demonstrate thoughtful agent-authored feature construction using the new fitted contract. Build a falsifiable microstructure mechanism and register one complete numerical feature program for later testing.',
        'requirements':['Use model.custom_features=true and compute_signal(df) returning 3..8 bounded named causal feature columns.',
            'Choose a defensible ridge, quadratic_ridge or xgboost recipe. For a threshold/regime mechanism, bounded shallow trees are available.',
            'Inspect actual development data and measure at least one descriptive feature before implementing.',
            'Use only development indices 0..19: these complete days are being cached. Do not request late dates or sealed test data.',
            'The existing full-day ridge benchmark has produced no trades under configured costs on its first three outer dates. A fixed depth-flow baseline lost money; cash is a required comparison.',
            'Estimated gross prediction must exceed actual fees/spread/slippage; do not manufacture a lower fee tier or claim profits from IC.',
            'Prefix the title INTEGRATION DRAFT. Specify the mechanism, decisive ablation and falsification. Register the actual code; root replay will independently test it.',
            'Use max_train_rows <=30000, train_days <=12, min_train_days=10, inner_days=3. Keep the first experiment computationally bounded.'],
        'execution':settings.execution_dict(),'falsification':'Reject if the custom features add no outer-date net value or fail causality/schema checks',
        'required_deliverable':'Register one exact, implementable feature-program candidate. No backtest has been performed in this task; label the result as an untested draft.'}
    try:
        result=AgentRuntime(store,settings).run(state['campaign'],'engineer',task,tools,'feature-capability:engineer')
        with store.connect() as db:
            submissions=db.execute("SELECT payload FROM events WHERE campaign=? AND kind='SUBMITTED' ORDER BY id",(state['campaign'],)).fetchall()
        ids=list(dict.fromkeys(json.loads(r[0])['id'] for r in submissions))
        if not ids:
            raise RuntimeError('Real agent did not submit a draft')
        state.update(status='COMPLETE',candidate_ids=ids,receipt=result)
        state.pop('error_type',None)
        path.write_text(json.dumps(state,indent=2,allow_nan=False),encoding='utf-8')
        print(json.dumps(state,indent=2),flush=True)
    except BaseException as error:
        state.update(status='FAILED',error_type=type(error).__name__)
        path.write_text(json.dumps(state,indent=2),encoding='utf-8')
        raise


if __name__=='__main__':
    main()
