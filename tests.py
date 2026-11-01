from common import *
from data import Data,Preprocess,tree_features
from models import Network,tensors
from scenarios import Scenarios,freeze_manifests
from calendar_temperature import CalendarTemperature
from attribution import shapley,permutation_reference

def main():
 torch.set_num_threads(2);d=Data();jobs=specs();assert len(jobs)==605
 base=next(s for s in jobs if s['arm']=='A1' and s['family']=='mlp' and s['split']=='fold1' and s['seed']==42);tr,val,final,te=split(d,base);ids=tr[:32];pre=Preprocess(d,tr,base);raw=d.raw(ids);results=[]
 # Train preprocessing must not depend on heldout outcomes.
 saved=d.y[te].copy();d.y[te]+=100;changed=Preprocess(d,tr,base);d.y[te]=saved
 for k in pre.a:np.testing.assert_array_equal(pre.a[k],changed.a[k])
 for family in P['families']:
  for arm in ['A1','A2','A3','A6','A7']:
   s=dict(base,family=family,arm=arm,variant=arm);seed_all(42);net=Network(s,pre.a['temp_sigma'],pre.a['target_scale']);net.eval();inp=tensors(pre.transform(raw,s),'cpu')
   if arm!='A2':
    # Nonzero context path to prevent a trivial zero-head isolation test.
    with torch.no_grad():net.head[-1].weight.normal_(0,.02)
   with torch.no_grad():y,z=net(inp,True)
   assert torch.isfinite(y).all();params=sum(p.numel() for p in net.parameters());backbone=sum(p.numel() for p in net.backbone.parameters())+(sum(p.numel() for p in net.projection.parameters()) if family=='transformer' else 0)
   expected={'A1':5043,'A2':5074,'A3':5043,'A6':5042,'A7':5046}[arm];assert params-backbone==expected,(family,arm,params-backbone)
   for kind,value in [('load',1.1),('load',1.5),('temperature',5)]:
    rr={k:v.copy() for k,v in raw.items()}
    if kind=='load':rr['daily_load']*=value;rr['load'][:,1]*=value;rr['load'][:,3]*=value
    else:rr['daily_temperature']+=value
    with torch.no_grad():yp,zp=net(tensors(pre.transform(rr,s),'cpu'),True)
    if arm!='A2':
     for key in ['context','base_normalized','a_mm_per_10k']:torch.testing.assert_close(z[key],zp[key],rtol=0,atol=0)
     if arm!='A3':assert (yp>=y-1e-6).all()
     if arm=='A6' and kind=='temperature':torch.testing.assert_close(yp,y,atol=0,rtol=0)
   if arm!='A2':
    n=raw['daily_load']*raw['daily_mask'];q=np.maximum(raw['daily_temperature']-25,0)/float(pre.a['temp_sigma']);a=z['a_mm_per_10k'].detach().numpy();gamma=z['gamma'].detach().numpy();ref=a*(n*(1+gamma[:,None]*q)).sum(1)
    np.testing.assert_allclose(z['G_mm'].numpy(),ref,atol=3e-6,rtol=2e-6)
    cur=pre.current(raw);np.testing.assert_allclose(ref,a*cur[:,0]*(1+gamma*cur[:,1]),atol=3e-6,rtol=2e-6)
   results.append({'family':family,'arm':arm,'nonbackbone_parameters':expected})
  # Pair effective initial functions, even though their raw softplus parameters differ.
  nets=[]
  for arm in ['A1','A3']:
   seed_all(42);s=dict(base,family=family,arm=arm);net=Network(s,pre.a['temp_sigma'],pre.a['target_scale']);net.eval()
   with torch.no_grad():nets.append(net(tensors(pre.transform(raw,s),'cpu')).numpy())
  np.testing.assert_allclose(*nets,atol=2e-7,rtol=1e-6)
 for s in jobs:
  train,validation,refit,test=split(d,s);assert not set(train)&set(validation) and not set(refit)&set(test)
  if s['split'].startswith('class'):
   support=(d.a['thickness'][refit]>0).any(0);need=(d.a['thickness'][test]>0).any(0);assert (support|~need).all()
 freeze_manifests();old=json.loads((ROOT/'checks/reference_origin_manifest.json').read_text());new=json.loads((ROOT/'checks/origin_manifest.json').read_text());assert old==new
 expected=json.loads((ROOT/'checks/reference_explanation_manifest.json').read_text());assert expected==json.loads((ROOT/'checks/explanation_manifest.json').read_text())
 cal=CalendarTemperature();manifest=json.loads((ROOT/'checks/calendar_manifest.json').read_text());checked=0
 for sid,origins in manifest.items():
  for o in origins[::13]:
   k=o['target_ids'][0];day=int(d.meta.start_day.iloc[k]);a=cal.estimate(sid,day,o['origin_day'],'calendar_past')
   if a is not None:
    origin_year=pd.Timestamp(o['origin_day']*86400,unit='s').year;assert max(a['reference_years'])<origin_year;assert max(a['support_days'])<o['origin_day'];assert a['max_available_day']<o['origin_day'];checked+=1
 sc=Scenarios(d,['STR1']);sc.origins=sc.origins[:1];sc.missing=[]
 predictor=lambda r:r['memory'][:,0]+.01*r['daily_load'].sum(1)+.0001*r['history'][:,-1,2]
 before=sc.predict(predictor);last_order=sc.origins[0]['origin_order'];dy=d.y.copy();fr=d.frame.copy();d.y[d.groups=='STR1']+=999;d.frame.loc[(d.frame.STR_name=='STR1')&(d.frame.Observation_Order>last_order),'Rutting_Depth_mm']+=999
 sc2=Scenarios(d,['STR1']);sc2.origins=sc2.origins[:1];sc2.missing=[];after=sc2.predict(predictor);np.testing.assert_array_equal(before.y_pred_mm,after.y_pred_mm);d.y=dy;d.frame=fr
 loads=before.pivot(index='horizon',columns='scenario',values='input_deltaN_10k');assert np.allclose(loads.to_numpy(),loads.iloc[:,0].to_numpy()[:,None])
 vals=np.random.default_rng(42).normal(size=(10,16));np.testing.assert_allclose(shapley(vals),permutation_reference(vals),atol=1e-14)
 write(ROOT/'checks/tests.json',{'passed':True,'time':stamp(),'models':results,'split_jobs':605,'source_queries_checked':checked,'same_load_across_temperature_scenarios':True,'future_rutting_leakage_test':True,'reference_origin_and_explanation_manifests_identical':True,'test_scope':'implementation and source checks, no outer model performance or training'})
 print('TESTS_PASS',len(results),checked)
if __name__=='__main__':main()
