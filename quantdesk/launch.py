"""Launch research in a fresh interpreter with a durable, nonsecret request."""
import json
import os
import subprocess
import sys
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from .settings import ROOT
from .store import process_alive


def start_research(store,settings,cid,cycles=1,hours=8,max_retries=2):
    campaign=store.get_campaign(cid)
    if campaign['status']=='SEALED':
        raise ValueError('Campaign is sealed')
    if type(cycles) is not int or not 1<=cycles<=10 or not 0<hours<=72 or type(max_retries) is not int or not 0<=max_retries<=5:
        raise ValueError('Cycles 1..10, hours (0,72], retries 0..5')
    with store.connect() as db:
        lease=db.execute("SELECT pid,expires FROM leases WHERE name='research'").fetchone()
    if lease and (process_alive(lease['pid']) if lease['pid'] else lease['expires']>time.time()):
        raise ValueError('Research already running')
    directory=store.home/'jobs'/uuid.uuid4().hex[:12]
    directory.mkdir(parents=True)
    configuration=asdict(settings)
    configuration={key:str(value.resolve()) if isinstance(value,Path) else value for key,value in configuration.items()}
    request={'campaign':cid,'cycles':cycles,'hours':hours,'max_retries':max_retries,'settings':configuration}
    path=directory/'request.json'
    path.write_text(json.dumps(request,indent=2,allow_nan=False),encoding='utf-8')
    flags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0
    with (directory/'stdout.log').open('w',encoding='utf-8') as stdout,(directory/'stderr.log').open('w',encoding='utf-8') as stderr:
        process=subprocess.Popen([sys.executable,'-u','-m','quantdesk.research_worker',str(path)],
            cwd=ROOT,stdout=stdout,stderr=stderr,shell=False,creationflags=flags)
    return process,directory
