import fcntl,os,subprocess,sys
from common import *

def main():
 folder=ROOT/'queues/supplement';folder.mkdir(parents=True,exist_ok=True);lock=(folder/'manager.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);freeze_check();assert json.loads((ROOT/'checks/preflight_complete.json').read_text())['passed'];done=[]
 stages=[('training',['queue_train.py','main']),('analysis',['supplement_analysis.py']),('scenarios',['supplement_evaluation.py']),('verification',['verify_supplement.py']),('report',['report_supplement.py'])]
 try:
  for stage,args in stages:
   write(folder/'state.json',{'stage':stage,'pid':os.getpid(),'updated':stamp(),'completed_stages':done})
   with (folder/(stage+'.log')).open('a') as log:subprocess.run([sys.executable,*args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
   done.append(stage)
  write(folder/'state.json',{'stage':'awaiting_visual_review_and_package','pid':os.getpid(),'updated':stamp(),'completed_stages':done});write(folder/'compute_complete.json',{'time':stamp(),'jobs':100,'still_required':['actual figure review','scientific review','reproduction package','Windows checksum verification','GitHub delivery index update']})
 except Exception as e:write(folder/'failure.json',{'time':stamp(),'error':repr(e),'completed_stages':done});raise
if __name__=='__main__':main()
