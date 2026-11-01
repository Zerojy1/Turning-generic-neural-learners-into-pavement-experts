"""Covariate-only descriptions; no post-hoc model or threshold choices."""
from common import *
from data import Data
from analysis import analysis_check

def correlation(a,x,y):
 if len(a)<3 or a[x].std()==0 or a[y].std()==0:return None,None
 return float(a[x].corr(a[y])),float(a[x].rank().corr(a[y].rank()))

def main():
 analysis_check();d=Data();a=d.meta[d.meta.eligible].copy();out=ROOT/'results/tables';out.mkdir(parents=True,exist_ok=True)
 # Same end date may contain different start dates; report both definitions.
 d1=[]
 for grouping in [['end_day'],['start_day','end_day']]:
  for key,g in a.groupby(grouping):
   vals=key if isinstance(key,tuple) else (key,);span=float(g.Delta_Load_10k.max()-g.Delta_Load_10k.min());tol=1e-6*max(1,float(abs(g.Delta_Load_10k).max()))
   d1.append(dict(zip(grouping,vals),definition='+'.join(grouping),structures=g.STR_name.nunique(),start_dates=g.start_day.nunique(),min_deltaN_10k=float(g.Delta_Load_10k.min()),max_deltaN_10k=float(g.Delta_Load_10k.max()),span_10k=span,equal_within_tolerance=span<=tol,tolerance=tol))
 pd.DataFrame(d1).to_csv(out/'D1_load_synchrony.csv',index=False)
 mask=d.a['daily_mask'];a['daily_mean_heat25_proxy']=((np.maximum(d.a['daily_temperature']-25,0)*mask).sum(1)*24)[a.sample_id.to_numpy()]
 rows=[]
 for scope,group in [('all19',a),('test15',a[~a.STR_name.isin(P['splits']['anchors'])])]:
  for sid,g in [('pooled',group)]+list(group.groupby('STR_name')):
   for x in ['days','degree_hours25','daily_mean_heat25_proxy']:
    pear,spear=correlation(g,'Delta_Load_10k',x);rows.append(dict(scope=scope,STR_name=sid,n=len(g),variable=x,Pearson=pear,Spearman=spear,interpretation='descriptive; common calendar and duration can confound correlations'))
 pd.DataFrame(rows).to_csv(out/'D2_load_correlations.csv',index=False)
 rates=a.copy();rates['load_per_day']=rates.Delta_Load_10k/rates.days;rates['heat_per_day']=rates.degree_hours25/rates.days;rr=[]
 for sid,g in [('pooled',rates)]+list(rates.groupby('STR_name')):
  pear,spear=correlation(g,'load_per_day','heat_per_day');rr.append(dict(STR_name=sid,n=len(g),Pearson=pear,Spearman=spear,interpretation='duration-normalized descriptive correlation; no causal interpretation'))
 pd.DataFrame(rr).to_csv(out/'D2_intensity_correlations.csv',index=False)
 e=pd.read_csv(ROOT/'results/scenario_predictions.csv.gz');e=e[(e.arm_id=='main_A1_mlp')&(e.scenario!='one_step')&(e.scenario!='missing_one')]
 # Temperature input errors do not depend on model: use one fixed arm to avoid duplication.
 temp=[]
 for scope,z in [('full',e),('past_common',e[e.calendar_common])]:
  for (scenario,h,sid),g in z.groupby(['scenario','horizon','STR_name']):temp.append(dict(scope=scope,scenario=scenario,horizon=h,STR_name=sid,n=len(g),temperature_MAE_C=float(abs(g.temp_mean_error_C).mean()),temperature_bias_C=float(g.temp_mean_error_C.mean()),heat_proxy_MAE_C=float(abs(g.heat_proxy_error_C).mean())))
 pd.DataFrame(temp).to_csv(out/'temperature_input_errors.csv',index=False)
 e['target_year']=[pd.Timestamp(int(d.meta.end_day.iloc[k])*86400,unit='s').year for k in e.sample_id];e['abs_temperature_error']=abs(e.temp_mean_error_C);e['abs_heat_error']=abs(e.heat_proxy_error_C)
 e.groupby(['scenario','horizon','STR_name','target_year'],as_index=False).agg(n=('sample_id','size'),temperature_MAE_C=('abs_temperature_error','mean'),temperature_bias_C=('temp_mean_error_C','mean'),heat_proxy_MAE_C=('abs_heat_error','mean'),heat_proxy_bias_C=('heat_proxy_error_C','mean')).to_csv(out/'temperature_input_errors_by_year.csv',index=False)
 write(ROOT/'checks/data_checks_complete.json',{'time':stamp(),'dated_targets_all19':len(a),'D1':'date and interval matched; original load units 10000 repetitions','D2':'pooled plus within structure; correlations not causal explanations of negative contribution','temperature_errors':'one fixed arm; descriptive only; no scenario selection'})
if __name__=='__main__':main()
