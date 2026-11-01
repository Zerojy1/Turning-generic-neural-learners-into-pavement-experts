"""Independent fitted-model replay and independent metric/interval recomputation."""
from common import *
from data import Data
from models import Predictor

def main():
 torch.set_num_threads(2);freeze_check();d=Data();checks=[];jobs=specs();rows=[]
 for i,s in enumerate(jobs):
  folder=ROOT/'runs'/s['job_id'];c=json.loads((folder/'complete.json').read_text());assert c['provenance']==provenance() and c['smoke_epochs']==0
  ids=split(d,s)[-1];saved=pd.read_csv(folder/'predictions.csv');assert saved.sample_id.to_list()==ids.tolist();model=Predictor(s,folder,'cuda');p=model.predict(d,ids);np.testing.assert_allclose(p,saved.y_pred_mm,atol=3e-5,rtol=1e-6)
  tr,va,fi,te=split(d,s);np.testing.assert_array_equal(model.pre.a['train_ids'],fi);choice=json.loads((folder/'selection.json').read_text());expected=min(choice['candidates'],key=lambda x:(x['score'],x['candidate']));assert expected==choice['selected'];assert all(x['epoch'] in [1]+list(range(5,221,5)) for x in choice['candidates'])
  for ci in range(3):
   stored=np.load(folder/f'selection_{ci}/preprocessing.npz');np.testing.assert_array_equal(stored['train_ids'],tr)
  checks.append({'job':s['job_id'],'replay_max_abs_mm':float(np.max(abs(p-saved.y_pred_mm))),'passed':True});saved['family']=s['family'];saved['seed']=s['seed'];rows.append(saved)
  write(ROOT/'checks/replay_progress.json',{'completed':i+1,'total':100,'time':stamp()});del model
 seeds=pd.concat(rows);independent=seeds.groupby(['family','sample_id']).agg(y_pred_mm=('y_pred_mm','mean'),y_true_mm=('y_true_mm','first'),persistence_mm=('y_previous_mm','first'),STR_name=('STR_name','first'),eligible=('eligible','first')).reset_index();summary=pd.read_csv(ROOT/'results/tables/one_step_summary.csv');metric_checks=[]
 for family in P['families']:
  for scope,eligible in [('primary2089',True),('all2164',False)]:
   a=independent[independent.family==family];a=a[a.eligible] if eligible else a;per=[]
   for sid,g in a.groupby('STR_name'):
    error=g.y_pred_mm.to_numpy()-g.y_true_mm.to_numpy();delta=g.y_true_mm.to_numpy()-g.persistence_mm.to_numpy();per.append((np.mean(error**2),np.mean(abs(error)),np.mean(delta**2),np.var(delta)))
   v=np.array(per);m,p=v[:,0].mean(),v[:,2].mean();values=[np.sqrt(m),v[:,1].mean(),1-m/p,1-m/v[:,3].mean()];published=summary[(summary.family==family)&(summary.arm=='A0')&(summary.scope==scope)].iloc[0];np.testing.assert_allclose(values,published[['macro_RMSE_mm','macro_MAE_mm','skill','increment_R2']].to_numpy(float),atol=1e-9,rtol=1e-9);metric_checks.append({'family':family,'scope':scope,'n':len(a),'passed':True})
 # Independently rebuild the paired bootstrap from saved structure errors.
 metrics=pd.read_csv(ROOT/'results/tables/structure_metrics.csv');primary=pd.read_csv(ROOT/'results/tables/supplementary_primary_contrasts.csv');pvals=[]
 for f in P['families']:
  z=metrics[(metrics.scope=='primary2089')&(metrics.family==f)].sort_values('STR_name');aa=z[z.arm=='A1'];bb=z[z.arm=='A0'];assert aa.STR_name.tolist()==bb.STR_name.tolist();am=aa.MSE_mm2.to_numpy();bm=bb.MSE_mm2.to_numpy();den=aa.persistence_MSE_mm2.to_numpy();delta=(bm.mean()-am.mean())/den.mean();indices=np.random.default_rng(20261101).integers(0,15,(10000,15));boot=(bm[indices].mean(1)-am[indices].mean(1))/den[indices].mean(1);pv=(1+((boot-delta)>=delta).sum())/10001;pvals.append(pv);r=primary[primary.family==f].iloc[0];np.testing.assert_allclose([delta,*np.quantile(boot,[.025,.975]),pv],r[['delta_skill','skill_CI_low','skill_CI_high','reference_p']].to_numpy(float),atol=1e-9)
 order=np.argsort(pvals);adjusted=np.zeros(4);last=0
 for rank,j in enumerate(order):last=max(last,(4-rank)*pvals[j]);adjusted[j]=min(last,1)
 for f,p in zip(P['families'],adjusted):np.testing.assert_allclose(primary[primary.family==f].holm_p.iloc[0],p,atol=1e-12)
 # Recompute all twenty calibration tables from their held-in inner predictions.
 radii=pd.read_csv(ROOT/'results/tables/new_calibration_radii.csv');cal_checks=0
 for path in sorted((ROOT/'evaluation').glob('*_inner.csv.gz')):
  tag=path.name.removesuffix('_inner.csv.gz');b=pd.read_csv(path);armid,fold=tag.rsplit('_',1)
  for (scenario,h),g in b.groupby(['scenario','horizon']):
   err=abs(g.y_pred_mm-g.y_true_mm).to_numpy();ns=g.groupby('STR_name').size().to_dict();weights=np.array([1/(len(ns)*ns[s]) for s in g.STR_name]);ix=np.argsort(err,kind='stable');k=min(np.searchsorted(np.cumsum(weights[ix]),.95,side='left'),len(ix)-1);r=radii[(radii.arm_id==armid)&(radii.split==fold)&(radii.scenario==scenario)&(radii.horizon==h)];assert len(r)==1;np.testing.assert_allclose(err[ix[k]],r.radius_mm.iloc[0],atol=1e-8);cal_checks+=1
 # Existing model scores are exact reused artifacts, not retrained estimates.
 old=pd.read_csv(ROOT/'base_reference/results/tables/one_step_summary.csv');pairs=summary[summary.arm.isin(['A1','A2','A4'])].merge(old[old.block=='main'],on=['scope','arm_id','arm','family'],suffixes=('_new','_old'),validate='one_to_one');assert len(pairs)==18
 for col in ['macro_RMSE_mm','macro_MAE_mm','skill','increment_R2']:np.testing.assert_allclose(pairs[col+'_new'],pairs[col+'_old'],atol=1e-10)
 write(ROOT/'checks/independent_supplement_verification.json',{'passed':True,'time':stamp(),'new_models_replayed':100,'replays':checks,'metrics':metric_checks,'bootstrap_and_holm_independently_recomputed':True,'calibration_groups_recomputed':cal_checks,'reused_base_summaries_unchanged':True});print('INDEPENDENT_VERIFICATION_PASS')
if __name__=='__main__':main()
