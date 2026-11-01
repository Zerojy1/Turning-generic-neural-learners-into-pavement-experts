import subprocess,sys
from common import *
from data import Data,Preprocess
from models import Network,tensors

def main():
 torch.set_num_threads(2);d=Data();jobs=specs();assert len(jobs)==100 and not (ROOT/'runs').exists();rows=[]
 for family in P['families']:
  s=next(x for x in jobs if x['family']==family);tr,va,fi,te=split(d,s);r=d.raw(tr[:24]);pre=Preprocess(d,tr,s);other=dict(s,arm='A2',output='residual');pre2=Preprocess(d,tr,other)
  for key in pre.a:
   if key!='target_center':np.testing.assert_array_equal(pre.a[key],pre2.a[key])
  np.testing.assert_array_equal(offset(r,s),np.zeros(len(r['memory'])));np.testing.assert_array_equal(offset(r,other),r['memory'][:,0])
  seed_all(42);direct=Network(s,pre.a['temp_sigma'],pre.a['target_scale']).eval();seed_all(42);residual=Network(other,pre2.a['temp_sigma'],pre2.a['target_scale']).eval()
  assert direct.state_dict().keys()==residual.state_dict().keys()
  for key in direct.state_dict():torch.testing.assert_close(direct.state_dict()[key],residual.state_dict()[key],rtol=0,atol=0)
  assert direct.head[0].in_features==87 and direct.head[0].out_features==57
  with torch.no_grad():torch.testing.assert_close(direct(tensors(pre.transform(r,s),'cpu')),residual(tensors(pre2.transform(r,other),'cpu')),atol=0,rtol=0)
  original=d.y[te].copy();d.y[te]+=1000;changed=Preprocess(d,tr,s);d.y[te]=original
  for key in pre.a:np.testing.assert_array_equal(pre.a[key],changed.a[key])
  # Ensure current y_t is absent from inputs, using only training targets for this check.
  j=int(tr[10]);before=d.raw([j]);saved=d.y[j];d.y[j]+=1000;after=d.raw([j]);d.y[j]=saved
  for key in before:np.testing.assert_array_equal(before[key],after[key])
  assert not set(tr)&set(va) and not set(fi)&set(te)
  rows.append({'family':family,'parameters':sum(x.numel() for x in direct.parameters()),'same_A2_parameters_and_initial_weights':True,'test_outcomes_do_not_affect_preprocessing':True,'output':'depth'})
 for s in jobs:
  assert s['output']=='depth' and s['arm']=='A0';tr,va,fi,te=split(d,s);assert not set(fi)&set(te)
 assert len(set(x['job_id'] for x in jobs))==100
 write(ROOT/'checks/unit_checks.json',{'passed':True,'time':stamp(),'rows':rows,'jobs':100,'outer_performance_used':False})
 print('UNIT_CHECKS_PASS',json.dumps(rows))
 benchmark=[]
 for family in P['families']:
  s=next(x for x in jobs if x['family']==family);subprocess.run([sys.executable,'run_job.py','--tag',s['job_id'],'--device','cuda','--smoke-epochs','20'],check=True,cwd=ROOT)
  folder=ROOT/'checks/smoke'/s['job_id'];times=[pd.read_csv(folder/f'selection_{i}/history.csv').seconds.iloc[-1]/20 for i in range(3)]
  per=float(np.mean(times));benchmark.append({'family':family,'inner_epoch_seconds':per,'jobs':25,'serial_training_minutes_low':25*(per*665+3)/60,'serial_training_minutes_high':25*(per*880*1.7+8)/60})
 write(ROOT/'checks/benchmark.json',{'time':stamp(),'training_side_only':True,'rows':benchmark,'two_worker_training_minutes':[sum(x['serial_training_minutes_low'] for x in benchmark)/2,sum(x['serial_training_minutes_high'] for x in benchmark)/2],'postprocessing_minutes':[15,40]})
 write(ROOT/'checks/preflight_complete.json',{'passed':True,'time':stamp(),'jobs':100,'unit_checks_sha256':digest(ROOT/'checks/unit_checks.json'),'benchmark_sha256':digest(ROOT/'checks/benchmark.json')});print('PREFLIGHT_COMPLETE',json.dumps(benchmark))
if __name__=='__main__':main()
