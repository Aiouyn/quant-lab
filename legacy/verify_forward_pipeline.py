"""Actual short public-source receipt smoke; deliberately cannot qualify."""
import json
import hashlib
from datetime import datetime,timezone
from pathlib import Path
from quantdesk.settings import Settings
from quantdesk.store import Store,digest
from quantdesk.paper import PaperSession
from quantdesk.forward import file_hash,assess_forward


def main():
    settings=Settings(); store=Store(settings.home)
    bundle=settings.home/'bundles'/'37e7854f12d976e8'
    manifest=json.loads((bundle/'manifest.json').read_text(encoding='utf-8'))
    assert manifest['status']!='PAPER_READY' and manifest['spec']['warmup_rows']>=6
    session=PaperSession(bundle,settings.home,settings,True)
    summary=session.run(samples=6,interval=1)
    metadata=json.loads((session.directory/'session.json').read_text(encoding='utf-8'))
    assert metadata['data_source']=='hyperliquid_public_mainnet'
    assert summary['journal_sha256']==file_hash(session.journal)
    assert summary['session_hash']==digest(metadata)
    assert summary['orders_submitted_to_exchange']==0 and not metadata.get('forward_plan_id')
    events=[json.loads(line) for line in session.journal.read_text(encoding='utf-8').splitlines()]
    assert events[-1]['summary']=={k:v for k,v in summary.items() if k!='journal_sha256'}
    assessment=assess_forward(store,manifest,settings)
    assert not assessment['eligible']
    result={'created_utc':datetime.now(timezone.utc).isoformat(),'actual_public_probe':summary,
        'integrity_checked':True,'metadata':metadata,'forward_assessment':assessment,
        'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('quantdesk').glob('*.py')},
        'interpretation':'Six-sample exploratory public-source receipt smoke only. No matured inference, alpha, qualified forward window, portfolio acceptance or exchange order.'}
    output=settings.home/'verification'/'forward-gate'; output.mkdir(parents=True,exist_ok=True)
    path=output/'public-probe.json'; path.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({'receipt':str(path),'summary':summary,'qualifies':assessment['eligible']},indent=2))


if __name__=='__main__':main()
