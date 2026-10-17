"""Real falsifier/historian handoff grounded in the completed feature replay."""
import json
import os
from dataclasses import replace
from quantdesk.settings import Settings
from quantdesk.store import Store
from quantdesk.data import Catalog
from quantdesk.evaluation import Evaluator
from quantdesk.agents import AgentRuntime,ResearchTools


def main():
    settings=replace(Settings(),max_completion_tokens=16000,max_agent_steps=3,request_timeout=900)
    store=Store(settings.home); root=settings.home/'verification'/'feature-replay'
    report=json.loads((root/'report.json').read_text())
    frontier=json.loads((root/'prediction-frontier.json').read_text())
    candidate=store.get_candidate(report['candidate']); cid=candidate['campaign']
    state={'status':'RUNNING','pid':os.getpid(),'campaign':cid,'candidate':candidate['id'],
        'phase':'Real critique and measured organizational lesson','learning_improvement_proven':False}
    path=root/'review-state.json'
    path.write_text(json.dumps(state,indent=2))
    runtime=AgentRuntime(store,settings)
    class ReviewTools(ResearchTools):
        def available(self,role):
            permitted={'read_candidate','record_lesson'}
            return [t for t in super().available(role) if t['function']['name'] in permitted]
    tools=ReviewTools(cid,store,Catalog(settings.data_dir,settings.home),Evaluator(store,Catalog(settings.data_dir,settings.home),settings),settings,'feature-review')
    try:
        with store.lease('research',seconds=3600):
            critique=runtime.run(cid,'falsifier',{'candidate':{k:candidate[k] for k in ('id','code','spec','status')},
                'development_replay':report,'prediction_frontier':frontier,
                'required_deliverable':'Return one concise JSON critique: measured findings, unsupported causal claims, cost/margin semantics, and ONE decisive economical next experiment with stop criteria. Existing measurements are bounded first-10000-row descriptive samples; outer replay is full-day. No code or retrieval is needed: exact source and measured evidence are attached. Do not generalize three dates to a whole strategy class, propose a fee reduction, or suggest bypassing funding safeguards. Acknowledge that repeated organizational learning is not yet proven.'},tools,'feature-review:falsifier')
            (root/'critique.json').write_text(json.dumps(critique,indent=2,allow_nan=False))
            state['phase']='Record measured lesson'; path.write_text(json.dumps(state,indent=2))
            lesson=runtime.run(cid,'historian',{'cohort':[{'candidate_id':candidate['id']}],
                'measured_replay':report,'prediction_frontier':frontier,'critique':critique,
                'required_deliverable':'Call record_lesson once for this exact candidate with measured negative evidence, limitations, uncertainty and the decisive next experiment. Do not call read_candidate: the evidence is already attached. No validation or holdout has occurred. The tool attaches authoritative walk-forward grounding. A stored lesson alone does not establish improving research performance.'},tools,'feature-review:historian')
            state.update(status='COMPLETE',phase='Measured handoff recorded',receipt=lesson)
            path.write_text(json.dumps(state,indent=2,allow_nan=False))
            print(json.dumps(state,indent=2),flush=True)
    except BaseException as error:
        state.update(status='FAILED',error_type=type(error).__name__)
        path.write_text(json.dumps(state,indent=2))
        raise


if __name__=='__main__':
    main()
