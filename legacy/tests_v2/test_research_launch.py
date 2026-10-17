"""Research launches must use fresh modules and preserve execution configuration."""
import json
import os
import subprocess
import sys
from dataclasses import asdict,replace
from pathlib import Path
import pytest
from quantdesk.settings import Settings,ROOT
from quantdesk.store import Store
from quantdesk.launch import start_research
from quantdesk.operations import job_snapshot


def test_launcher_uses_argument_list_and_preserves_full_settings(tmp_path,monkeypatch):
    store=Store(tmp_path); cid=store.campaign({},1000)
    settings=replace(Settings(),home=tmp_path,data_dir=tmp_path/'data with spaces',taker_fee_bps=7.1)
    captured={}
    class Process:
        pid=1073741823
    def launch(args,**kwargs):
        captured.update(args=args,kwargs=kwargs)
        return Process()
    monkeypatch.setattr(subprocess,'Popen',launch)
    process,directory=start_research(store,settings,cid,2,4,1)
    request=json.loads((directory/'request.json').read_text())
    assert request['settings']['taker_fee_bps']==7.1
    assert request['settings']['data_dir']==str(settings.data_dir.resolve())
    assert request['cycles']==2 and request['max_retries']==1
    assert captured['args']==[sys.executable,'-u','-m','quantdesk.research_worker',str(directory/'request.json')]
    assert captured['kwargs']['shell'] is False and captured['kwargs']['cwd']==ROOT
    assert not any('key' in k.lower() for k in request['settings'])


def test_live_research_lease_prevents_another_child(tmp_path,monkeypatch):
    store=Store(tmp_path); cid=store.campaign({},1000)
    with store.lease('research'):
        with pytest.raises(ValueError,match='already running'):
            start_research(store,replace(Settings(),home=tmp_path),cid)
    assert not (tmp_path/'jobs').exists()


def test_actual_fresh_worker_fails_unknown_campaign_before_provider_use(tmp_path):
    directory=tmp_path/'jobs'/'actual-probe'; directory.mkdir(parents=True)
    settings=replace(Settings(),home=tmp_path,data_dir=tmp_path/'data')
    configuration={k:str(v) if isinstance(v,Path) else v for k,v in asdict(settings).items()}
    path=directory/'request.json'
    path.write_text(json.dumps({'campaign':'missing-campaign','cycles':1,'hours':1,'max_retries':0,'settings':configuration}))
    result=subprocess.run([sys.executable,'-m','quantdesk.research_worker',str(path)],cwd=ROOT,
        capture_output=True,text=True,timeout=30,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    assert result.returncode!=0
    state=json.loads((directory/'status.json').read_text())
    assert state['status']=='FAILED' and state['error']=='Unknown campaign'
    assert state['source_sha256']['research_worker.py']
    store=Store(tmp_path)
    assert store.status()['usage']==[] and not store.candidates()
    snapshot=next(j for j in job_snapshot(store) if j['kind']=='research_job')
    assert snapshot['status']=='FAILED' and not snapshot['running']
