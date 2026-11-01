"""Independent checks of split membership, saved models, metrics and scenario constraints."""
from common import *
from data import Data
from models import Predictor
from analysis import analysis_check

def main():
 torch.set_num_threads(2);freeze_check();analysis_check();d=Data();checks=[];maxerror=0.
 for i,s in enumerate(specs()):
  folder=ROOT/'runs'/s['job_id'];tr,val,final,test=split(d,s);member=pd.read_csv(folder/'membership.csv')
  for name,ids in [('inner_train',tr),('inner_validation',val),('final_train',final),('outer_test',test)]:assert member.loc[member[name],'sample_id'].tolist()==ids.tolist()
  pre=dict(np.load(folder/'refit/preprocessing.npz'));assert set(pre['train_ids'])==set(final)
  sel=json.loads((folder/'selection.json').read_text());expected=min(sel['candidates'],key=lambda x:(x['score'],x['candidate']));assert sel['selected']==expected
  m=Predictor(s,folder,'cpu' if s['family']=='xgb' else 'cuda');p=m.predict(d,test);saved=pd.read_csv(folder/'predictions.csv');err=float(abs(p-saved.y_pred_mm.to_numpy()).max());assert err<4e-5;maxerror=max(maxerror,err)
  checks.append(dict(job_id=s['job_id'],replay_max_abs_mm=err,inner_structures=len(set(d.groups[tr])),outer_structures=len(set(d.groups[test]))));del m
  if (i+1)%25==0:write(ROOT/'checks/verification_progress.json',{'time':stamp(),'models':i+1,'total':605})
 a=pd.read_csv(ROOT/'results/ensemble_predictions.csv.gz');table=pd.read_csv(ROOT/'results/tables/one_step_summary.csv');metric_error=0.
 for r in table[table.arm!='A5'].itertuples():
  z=a[(a.arm_id==r.arm_id)&(a.block==r.block)]
  if r.scope=='primary2089':z=z[z.eligible]
  mse=[];pmse=[];mae=[]
  for sid,g in z.groupby('STR_name'):
   ee=g.y_true_mm.to_numpy()-g.y_pred_mm.to_numpy();pp=g.y_true_mm.to_numpy()-g.persistence_mm.to_numpy();mse.append(float(np.dot(ee,ee)/len(ee)));pmse.append(float(np.dot(pp,pp)/len(pp)));mae.append(float(sum(abs(ee))/len(ee)))
  assert len(mse)==r.structures and len(z)==r.n
  err=max(abs(np.sqrt(sum(mse)/len(mse))-r.macro_RMSE_mm),abs(1-sum(mse)/sum(pmse)-r.skill),abs(sum(mae)/len(mae)-r.macro_MAE_mm));assert err<1e-9;metric_error=max(metric_error,err)
 # Primary tests recomputed directly from ensemble errors, without analysis.paired.
 primary=pd.read_csv(ROOT/'results/tables/primary_backbone_contrasts.csv');family_ps={}
 for r in primary.itertuples():
  xa=a[(a.block=='main')&(a.arm=='A1')&(a.family==r.family)&a.eligible];xb=a[(a.block=='main')&(a.arm==('A2' if r.comparison=='P1' else 'A3'))&(a.family==r.family)&a.eligible];v=[]
  for sid in sorted(xa.STR_name.unique()):
   xx=xa[xa.STR_name==sid].sort_values('sample_id');yy=xb[xb.STR_name==sid].sort_values('sample_id');assert xx.sample_id.tolist()==yy.sample_id.tolist();v.append([np.mean((xx.y_true_mm-xx.y_pred_mm)**2),np.mean((yy.y_true_mm-yy.y_pred_mm)**2),np.mean((xx.y_true_mm-xx.persistence_mm)**2)])
  v=np.array(v);index=np.random.default_rng(20261101).integers(0,len(v),(10000,len(v)));means=v[index].mean(1);delta=(v[:,1].mean()-v[:,0].mean())/v[:,2].mean();boot=(means[:,1]-means[:,0])/means[:,2];p=(1+sum(boot-delta>=delta-r.boundary))/10001;assert abs(delta-r.delta_skill)<1e-9 and abs(p-r.reference_p)<1e-9;family_ps.setdefault(r.comparison,[]).append(float(p))
 families=pd.read_csv(ROOT/'results/tables/primary_conjunctions.csv');ps=np.array([max(family_ps[k]) for k in families.comparison]);order=np.argsort(ps);adjusted=np.empty(2);adjusted[order[0]]=min(1.,2*ps[order[0]]);adjusted[order[1]]=min(1.,max(adjusted[order[0]],ps[order[1]]));np.testing.assert_allclose(adjusted,families.holm_p,atol=1e-9);assert ((adjusted<=.05)==families.passed).all()
 # All folds/arms compared at exactly the same target keys; ensemble updates, peak max.
 scen=pd.read_csv(ROOT/'results/scenario_predictions.csv.gz');one=scen[scen.scenario=='one_step'];ref=a[a.eligible&(a.block!='transfer')];merged=one.merge(ref,on=['arm_id','sample_id'],suffixes=('_s','_p'),validate='one_to_one');assert len(merged)==len(one);assert abs(merged.y_pred_mm_s-merged.y_pred_mm_p).max()<4e-5
 roll=scen[scen.scenario.isin(['realized','calendar_past','recent_temperature','calendar_loyo_retrospective'])].copy()
 for _,g in roll.groupby(['arm_id','STR_name','origin_order','scenario']):
  g=g.sort_values('horizon');assert g.horizon.tolist()==[1,2,3,4];np.testing.assert_allclose(g.input_latest_mm.iloc[1:],g.y_pred_mm.iloc[:-1],atol=3e-5,rtol=1e-6);np.testing.assert_allclose(g.input_peak_mm.iloc[1:],np.maximum(g.input_peak_mm.iloc[:-1].to_numpy(),g.y_pred_mm.iloc[:-1].to_numpy()),atol=3e-5,rtol=1e-6)
 for _,g in roll.groupby(['arm_id','STR_name','origin_order','horizon']):assert g.input_deltaN_10k.max()-g.input_deltaN_10k.min()<1e-6 and g.input_days.nunique()==1
 origin_counts={}
 for armid,g in roll[roll.scenario=='calendar_past'].groupby('arm_id'):origin_counts[armid]=g[['STR_name','origin_order']].drop_duplicates().shape[0]
 summary=json.loads((ROOT/'checks/calendar_summary.json').read_text());assert set(origin_counts.values())=={summary['strict_past_common_origins']}
 # Source-date/year audit, all provenance records rather than only sampled examples.
 from datetime import date,timedelta
 source_count=0
 with (ROOT/'checks/calendar_day_provenance.jsonl').open() as fh:
  for line in fh:
   r=json.loads(line);source_count+=1;origyear=(date(1970,1,1)+timedelta(days=r['origin_day'])).year;targetyear=(date(1970,1,1)+timedelta(days=r['target_day'])).year;years=[(date(1970,1,1)+timedelta(days=x)).year for x in r['support_days']]
   if not r['retrospective']:assert max(r['support_days'])<r['origin_day'] and r['max_available_day']<r['origin_day'] and max(years)<origyear
   else:assert targetyear not in years
 audit=pd.read_csv(ROOT/'results/response_rows.csv.gz');assert len(audit)==250680 and not audit.loc[audit.arm=='A1','decrease'].any()
 shap=pd.read_csv(ROOT/'results/attributions.csv.gz');assert abs(shap.additivity_error_mm).max()<1e-8
 for _,g in shap.groupby(['arm_id','reference','STR_name','sample_id','seed']):assert len(g)==4 and abs(g.phi_mm.sum()+g.baseline_mm.iloc[0]-g.increment_prediction_mm.iloc[0])<1e-8
 pd.DataFrame(checks).to_csv(ROOT/'results/tables/independent_model_replay.csv',index=False)
 write(ROOT/'checks/independent_verification.json',{'passed':True,'time':stamp(),'models_reloaded':len(checks),'max_model_replay_error_mm':maxerror,'max_independent_metric_error':metric_error,'scenario_ensemble_updates_checked':True,'same_given_loads_checked':True,'strict_past_origins_per_arm':origin_counts,'calendar_provenance_records_checked':source_count,'attribution_additivity_checked':True,'training_freeze':digest(ROOT/'training_freeze.json'),'analysis_freeze':digest(ROOT/'analysis_freeze.json')});print('INDEPENDENT_VERIFICATION_PASS',len(checks))
if __name__=='__main__':main()
