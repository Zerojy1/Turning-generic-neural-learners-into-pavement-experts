from common import *
from data import Data
from models import Predictor
from analysis import structure_metrics,summarize,paired,FAMILIES
from evaluation import predictions,radius,scopes,interval_tables

def main():
 torch.set_num_threads(2);freeze_check();d=Data();folder=ROOT/'evaluation';folder.mkdir(exist_ok=True);rows=[];cal=[];jobs=specs();configs=[s for s in jobs if s['seed']==42]
 for count,s in enumerate(configs):
  tag=s['arm_id']+'_'+s['split'];path=folder/(tag+'.csv.gz');mark=folder/(tag+'.json')
  if not mark.exists():
   start=time.monotonic();ss=[x for x in jobs if x['arm_id']==s['arm_id'] and x['split']==s['split']];assert len(ss)==5
   outer=[Predictor(x,ROOT/'runs'/x['job_id'],'cuda') for x in ss];inner=[Predictor(x,ROOT/'runs'/x['job_id'],'cuda',selection=True) for x in ss];_,val,_,test=split(d,s);a=predictions(d,test,outer);b=predictions(d,val,inner);b.to_csv(folder/(tag+'_inner.csv.gz'),index=False)
   r=[dict(scenario=scenario,horizon=int(h),radius_mm=radius(g),calibration_n=len(g),calibration_structures=g.STR_name.nunique()) for (scenario,h),g in b.groupby(['scenario','horizon'])]
   a=a.merge(pd.DataFrame(r),on=['scenario','horizon'],how='left',validate='many_to_one');a['lower_mm']=a.y_pred_mm-a.radius_mm;a['upper_mm']=a.y_pred_mm+a.radius_mm
   for k in ['arm_id','arm','family','split']:a[k]=s[k]
   a.to_csv(path,index=False);write(mark,{'time':stamp(),'seconds':time.monotonic()-start,'calibration':r});del outer,inner
  rows.append(pd.read_csv(path))
  for r in json.loads(mark.read_text())['calibration']:cal.append(dict(arm_id=s['arm_id'],split=s['split'],**r))
  write(folder/'state.json',{'updated':stamp(),'completed':count+1,'total':len(configs),'last':tag})
 old=pd.read_csv(ROOT/'base_reference/results/scenario_predictions.csv.gz');old=old[old.arm.isin(['A1','A2','A4']) & old.arm_id.str.startswith('main_')];new=pd.concat(rows,ignore_index=True)
 keys=['scenario','horizon','STR_name','origin_order','sample_id','target_order','split']
 for f in FAMILIES:
  aa=new[new.family==f]
  for arm in ['A1','A2']:
   bb=old[(old.family==f)&(old.arm==arm)];pair=aa.merge(bb,on=keys,suffixes=('_new','_old'),validate='one_to_one');assert len(pair)==len(aa)==len(bb);np.testing.assert_allclose(pair.y_true_mm_new,pair.y_true_mm_old,atol=1e-9);np.testing.assert_allclose(pair.persistence_mm_new,pair.persistence_mm_old,atol=1e-9)
 a=pd.concat([old,new],ignore_index=True);base=old[old.arm=='A4'].copy();base.y_pred_mm=base.persistence_mm;base.arm='A5';base.arm_id='main_A5_persistence';base.family='persistence';base[['lower_mm','upper_mm','radius_mm']]=np.nan;a=pd.concat([a,base],ignore_index=True);a.to_csv(ROOT/'results/scenario_predictions.csv.gz',index=False)
 out=ROOT/'results/tables';pd.DataFrame(cal).to_csv(out/'new_calibration_radii.csv',index=False);pieces=[]
 for scope,z in scopes(a):z=z.copy();z['scope']=scope;pieces.append(z)
 aa=pd.concat(pieces,ignore_index=True);keys=['scope','scenario','horizon','arm_id','arm','family'];m=structure_metrics(aa,keys);m.to_csv(out/'scenario_structure_metrics.csv',index=False);summarize(m,keys).to_csv(out/'scenario_summary.csv',index=False);iv,ivs=interval_tables(aa[aa.arm!='A5']);iv.to_csv(out/'interval_structures.csv',index=False);ivs.to_csv(out/'interval_summary.csv',index=False)
 contrasts=[]
 for (scope,scenario,h),z in m.groupby(['scope','scenario','horizon']):
  for f in FAMILIES:
   for left,right in [('A1','A0'),('A2','A0'),('A1','A2')]:
    x=z[(z.arm==left)&(z.family==f)];y=z[(z.arm==right)&(z.family==f)]
    if len(x) and len(x)==len(y):contrasts.append(dict(scope=scope,scenario=scenario,horizon=h,family=f,comparison=f'{left}_minus_{right}',**paired(x,y)))
 pd.DataFrame(contrasts).drop(columns=['reference_p','boundary']).to_csv(out/'scenario_contrasts.csv',index=False);write(ROOT/'checks/supplement_evaluation_complete.json',{'passed':True,'time':stamp(),'new_configs':20,'rows':len(a),'same_targets_and_persistence_checked':True})
if __name__=='__main__':main()
