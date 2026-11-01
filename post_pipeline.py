import fcntl,os,subprocess,sys
from common import *
def main():
 folder=ROOT/'queues/post';folder.mkdir(parents=True,exist_ok=True);lock=(folder/'manager.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);done=[]
 try:
  while not (ROOT/'queues/training/complete.json').exists():
   if (ROOT/'queues/training/failure.json').exists():raise RuntimeError('training failed; logs retained, no postprocess')
   write(folder/'state.json',{'pid':os.getpid(),'stage':'waiting_training','updated':stamp()});time.sleep(20)
  for script,mark in [('analysis.py','analysis_complete'),('evaluation.py','evaluation_complete'),('data_checks.py','data_checks_complete'),('audit_responses.py','responses_complete'),('attribution.py','attribution_complete'),('finish_analysis.py','finish_analysis_complete'),('verify_results.py','independent_verification')]:
   write(folder/'state.json',{'pid':os.getpid(),'stage':script,'completed':done,'updated':stamp()})
   if not (ROOT/f'checks/{mark}.json').exists():
    with (ROOT/'logs'/('post_'+script+'.log')).open('w') as log:subprocess.run([sys.executable,str(ROOT/script)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
   done.append(script)
  write(folder/'complete.json',{'finished':stamp(),'scripts':done});write(folder/'state.json',{'pid':os.getpid(),'stage':'numerical_complete_reporting_pending','completed':done,'updated':stamp()})
 except Exception as e:write(folder/'failure.json',{'time':stamp(),'error':repr(e),'completed':done});raise
if __name__=='__main__':main()
