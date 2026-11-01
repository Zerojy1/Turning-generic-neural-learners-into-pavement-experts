import argparse,fcntl,subprocess,sys,os
from common import *
STAGES={k:[k] for k in ['main','components','transfer']}
def run(stage):
 freeze_check();path=ROOT/'queues'/stage;path.mkdir(parents=True,exist_ok=True);lock=(path/'manager.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);todo=[s for s in specs() if s['block'] in STAGES[stage]];pending=[];done=[];prov=provenance()
 for s in todo:
  c=ROOT/'runs'/s['job_id']/'complete.json'
  if c.exists():assert json.loads(c.read_text())['provenance']==prov;done.append(s['job_id'])
  else:pending.append(s)
 running={};failed=[];started=stamp()
 def save():write(path/'state.json',{'stage':stage,'started':started,'updated':stamp(),'manager_pid':os.getpid(),'total':len(todo),'completed':done,'failed':failed,'running':{k:{'pid':v['process'].pid,'family':v['spec']['family'],'block':v['spec']['block'],'started_epoch':v['epoch'],'device':v['device']} for k,v in running.items()},'pending':len(pending),'workers':{'cuda':2,'cpu':2}})
 while pending or running:
  for tag,v in list(running.items()):
   code=v['process'].poll()
   if code is None:continue
   v['log'].close();del running[tag]
   if code==0 and (ROOT/'runs'/tag/'complete.json').exists():done.append(tag)
   else:failed.append({'tag':tag,'exit_code':code,'log':str(path/(tag+'.log'))})
  if not failed:
   for device,limit in [('cuda',2),('cpu',2)]:
    free=limit-sum(v['device']==device for v in running.values())
    while free:
     idx=next((i for i,s in enumerate(pending) if (s['family']=='xgb')==(device=='cpu')),None)
     if idx is None:break
     s=pending.pop(idx);log=(path/(s['job_id']+'.log')).open('w');env=os.environ.copy();env.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1');proc=subprocess.Popen([sys.executable,str(ROOT/'run_job.py'),'--tag',s['job_id'],'--device',device],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=env);running[s['job_id']]={'process':proc,'spec':s,'epoch':time.time(),'device':device,'log':log};free-=1
  save()
  if failed and not running:break
  time.sleep(3)
 save()
 if failed:raise RuntimeError(f'{stage}: {failed}')
 assert len(done)==len(todo);write(path/'complete.json',{'stage':stage,'completed':len(done),'finished':stamp()});print('QUEUE_COMPLETE',stage,len(done),flush=True)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('stage',choices=list(STAGES));run(a.parse_args().stage)
