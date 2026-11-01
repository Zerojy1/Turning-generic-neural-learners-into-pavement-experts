import fcntl,os,subprocess,sys
from common import *
def main():
 path=ROOT/'queues/training';path.mkdir(parents=True,exist_ok=True);lock=(path/'manager.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);freeze_check();assert json.loads((ROOT/'checks/preflight_complete.json').read_text())['passed'];completed=[]
 try:
  for stage in ['main','components','transfer']:
   write(path/'state.json',{'manager_pid':os.getpid(),'stage':stage,'completed_stages':completed,'updated':stamp()});subprocess.run([sys.executable,str(ROOT/'queue_train.py'),stage],check=True,cwd=ROOT);completed.append(stage)
  write(path/'complete.json',{'finished':stamp(),'jobs':605});write(path/'state.json',{'manager_pid':os.getpid(),'stage':'complete','completed_stages':completed,'updated':stamp()})
 except Exception as e:write(path/'failure.json',{'time':stamp(),'error':repr(e),'completed_stages':completed});raise
if __name__=='__main__':main()
