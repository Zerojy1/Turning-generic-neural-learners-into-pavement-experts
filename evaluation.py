"""Inference-only scenarios; inner selected checkpoints supply empirical calibration."""
from common import *
from data import Data
from models import Predictor
from scenarios import Scenarios
from analysis import analysis_check,structure_metrics,summarize,paired,FAMILIES

def radius(a,level=.95):
 if len(a)==0:return None
 z=a.copy();z['err']=abs(z.y_pred_mm-z.y_true_mm);counts=z.groupby('STR_name').size();z['w']=[1/(len(counts)*counts[s]) for s in z.STR_name];z=z.sort_values('err',kind='stable');i=min(np.searchsorted(z.w.cumsum().to_numpy(),level,side='left'),len(z)-1);return float(z.err.iloc[i])

def predictions(d,ids,models):
 def pred(raw):return np.mean([m.predict_raw(raw) for m in models],axis=0)
 a=Scenarios(d,np.unique(d.groups[ids])).predict(pred);oneids=ids[d.meta.eligible.iloc[ids].to_numpy(bool)];raw=d.raw(oneids)
 one=pd.DataFrame({'scenario':'one_step','horizon':1,'STR_name':d.groups[oneids],'origin_order':d.meta.Observation_Order.iloc[oneids].to_numpy()-1,'sample_id':oneids,'target_order':d.meta.Observation_Order.iloc[oneids].to_numpy(),'y_true_mm':d.y[oneids],'persistence_mm':raw['memory'][:,0],'y_pred_mm':pred(raw),'calendar_common':False,'calendar_two_year':False,'loyo_common':False})
 return pd.concat([one,a],ignore_index=True)

def scopes(a):
 yield 'full',a
 for scope,col in [('past_common','calendar_common'),('past_two_year','calendar_two_year'),('loyo_common','loyo_common')]:
  z=a[a[col].fillna(False).astype(bool)]
  if len(z):yield scope,z

def interval_tables(a):
 rows=[];keys=['scope','scenario','horizon','arm_id','arm','family']
 for key,g in a.groupby(keys+['STR_name'],dropna=False):
  v=g[g.lower_mm.notna()&g.upper_mm.notna()];n=len(v);covered=(v.y_true_mm>=v.lower_mm)&(v.y_true_mm<=v.upper_mm);width=v.upper_mm-v.lower_mm;mis=width+40*(np.maximum(v.lower_mm-v.y_true_mm,0)+np.maximum(v.y_true_mm-v.upper_mm,0))
  rows.append(dict(zip(keys+['STR_name'],key),n=len(g),interval_n=n,coverage=float(covered.mean()) if n else None,MPIW_mm=float(width.mean()) if n else None,MIS95_mm=float(mis.mean()) if n else None))
 frame=pd.DataFrame(rows);summary=frame.groupby(keys,as_index=False,dropna=False).agg(structures=('STR_name','nunique'),structures_with_intervals=('coverage','count'),n=('n','sum'),interval_n=('interval_n','sum'),mean_structure_coverage=('coverage','mean'),worst_structure_coverage=('coverage','min'),MPIW_mm=('MPIW_mm','mean'),MIS95_mm=('MIS95_mm','mean'));return frame,summary

def main():
 torch.set_num_threads(2);analysis_check();freeze_check();d=Data();folder=ROOT/'evaluation';folder.mkdir(exist_ok=True);allrows=[];cal=[]
 jobs=specs();configs=[s for s in jobs if s['seed']==42 and s['block']!='transfer']
 for count,s in enumerate(configs):
  tag=s['arm_id']+'_'+s['split'];path=folder/(tag+'.csv.gz');mark=folder/(tag+'.json')
  if not mark.exists():
   start=time.monotonic();ss=[x for x in jobs if x['arm_id']==s['arm_id'] and x['split']==s['split']];assert len(ss)==5
   outer=[Predictor(x,ROOT/'runs'/x['job_id'],'cpu' if x['family']=='xgb' else 'cuda') for x in ss];inner=[Predictor(x,ROOT/'runs'/x['job_id'],'cpu' if x['family']=='xgb' else 'cuda',selection=True) for x in ss]
   _,val,_,test=split(d,s);a=predictions(d,test,outer);b=predictions(d,val,inner);b.to_csv(folder/(tag+'_inner.csv.gz'),index=False);r=[]
   for (scenario,h),g in b.groupby(['scenario','horizon']):r.append(dict(scenario=scenario,horizon=int(h),radius_mm=radius(g),calibration_n=len(g),calibration_structures=g.STR_name.nunique()))
   a=a.merge(pd.DataFrame(r),on=['scenario','horizon'],how='left',validate='many_to_one');a['lower_mm']=a.y_pred_mm-a.radius_mm;a['upper_mm']=a.y_pred_mm+a.radius_mm
   for k in ['arm_id','arm','family','split']:a[k]=s[k]
   a.to_csv(path,index=False);write(mark,{'time':stamp(),'seconds':time.monotonic()-start,'tag':tag,'rows':len(a),'calibration':r});del outer,inner
  a=pd.read_csv(path);allrows.append(a)
  for r in json.loads(mark.read_text())['calibration']:cal.append(dict(arm_id=s['arm_id'],split=s['split'],**r))
  write(folder/'state.json',{'updated':stamp(),'completed':count+1,'total':len(configs),'last':tag})
 a=pd.concat(allrows,ignore_index=True);a.to_csv(ROOT/'results/scenario_predictions.csv.gz',index=False);out=ROOT/'results/tables';pd.DataFrame(cal).to_csv(out/'calibration_radii.csv',index=False)
 multi=[]
 for scope,z in scopes(a):z=z.copy();z['scope']=scope;multi.append(z)
 aa=pd.concat(multi,ignore_index=True);keys=['scope','scenario','horizon','arm_id','arm','family'];m=structure_metrics(aa,keys);m.to_csv(out/'scenario_structure_metrics.csv',index=False);summarize(m,keys).to_csv(out/'scenario_summary.csv',index=False)
 iv,ivs=interval_tables(aa);iv.to_csv(out/'interval_structures.csv',index=False);ivs.to_csv(out/'interval_summary.csv',index=False)
 desc=[]
 for (scope,scenario,h),z in m.groupby(['scope','scenario','horizon']):
  for f in FAMILIES:
   x=z[(z.arm=='A1')&(z.family==f)]
   if not len(x):continue
   for arm,fam in [('A2',f),('A4','xgb')]:
    y=z[(z.arm==arm)&(z.family==fam)]
    if len(y)==len(x):desc.append(dict(scope=scope,scenario=scenario,horizon=h,family=f,comparison='A1_minus_'+arm,**paired(x,y)))
 pd.DataFrame(desc).drop(columns=['reference_p','boundary']).to_csv(out/'scenario_contrasts.csv',index=False)
 temp=[]
 for (armid,f,h),z in m[m.scope=='past_common'].groupby(['arm_id','family','horizon']):
  x=z[z.scenario=='calendar_past'];y=z[z.scenario=='recent_temperature']
  if len(x) and len(x)==len(y):temp.append(dict(arm_id=armid,family=f,horizon=h,comparison='calendar_past_minus_recent_temperature',**paired(x,y)))
 pd.DataFrame(temp).drop(columns=['reference_p','boundary']).to_csv(out/'temperature_scenario_contrasts.csv',index=False)
 # Explicit persistence baseline on identical prediction targets at every horizon/scope.
 base=aa[aa.arm=='A4'].copy();base.y_pred_mm=base.persistence_mm;base.arm='A5';base.arm_id='main_A5_persistence';base.family='persistence';summarize(structure_metrics(base,keys),keys).to_csv(out/'scenario_persistence.csv',index=False)
 write(ROOT/'checks/evaluation_complete.json',{'time':stamp(),'configs':len(configs),'rows':len(a),'calibration':'selected-inner empirical equal-structure absolute-residual quantiles; reused selection data; no coverage guarantee','temperature_scenarios':'all identical given loads; frozen models; strict-past availability and retrospective LOYO separated'})
if __name__=='__main__':main()
