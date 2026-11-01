"""Frozen structure-balanced analysis; all non-primary contrasts are descriptive."""
from common import *
FAMILIES=['mlp','gru','lstm','transformer']
KEYS=['arm_id','block','arm','family','split']

def analysis_check():
 f=ROOT/'analysis_freeze.json'
 if f.exists():
  for n,h in json.loads(f.read_text())['files'].items():assert digest(ROOT/n)==h,('changed analysis file',n)

def aggregate():
 from data import Data
 d=Data();rows=[]
 for s in specs():
  folder=ROOT/'runs'/s['job_id'];c=json.loads((folder/'complete.json').read_text());assert matches_recorded_provenance(c['provenance']) and c['smoke_epochs']==0
  a=pd.read_csv(folder/'predictions.csv');ids=split(d,s)[-1];assert a.sample_id.tolist()==ids.tolist();assert np.isfinite(a.y_pred_mm).all()
  np.testing.assert_allclose(a.y_true_mm,d.y[ids],atol=1e-6);assert not a.sample_id.duplicated().any()
  for k in KEYS+['seed']:a[k]=s[k]
  rows.append(a)
 allrows=pd.concat(rows,ignore_index=True);g=allrows.groupby(KEYS+['sample_id'],sort=False,dropna=False)
 assert (g.seed.nunique()==5).all() and (g.size()==5).all()
 a=g.agg(STR_name=('STR_name','first'),Observation_Order=('Observation_Order','first'),eligible=('eligible','first'),y_true_mm=('y_true_mm','first'),persistence_mm=('y_previous_mm','first'),y_pred_mm=('y_pred_mm','mean'),seed_sd_mm=('y_pred_mm','std')).reset_index()
 for _,z in a[a.block!='transfer'].groupby('arm_id'):assert len(z)==2164 and z.eligible.sum()==2089
 return allrows,a

def structure_metrics(a,keys):
 rows=[]
 for key,g in a.groupby(keys+['STR_name'],dropna=False,sort=True):
  vals=key if isinstance(key,tuple) else (key,);e=g.y_pred_mm.to_numpy()-g.y_true_mm.to_numpy();inc=g.y_true_mm.to_numpy()-g.persistence_mm.to_numpy()
  rows.append(dict(zip(keys+['STR_name'],vals),n=len(g),MSE_mm2=float(np.mean(e*e)),MAE_mm=float(np.mean(abs(e))),persistence_MSE_mm2=float(np.mean(inc*inc)),increment_variance_mm2=float(np.mean((inc-inc.mean())**2))))
 return pd.DataFrame(rows)

def summarize(a,keys):
 rows=[]
 for key,g in a.groupby(keys,dropna=False,sort=True):
  vals=key if isinstance(key,tuple) else (key,);m=g.MSE_mm2.mean();p=g.persistence_MSE_mm2.mean();v=g.increment_variance_mm2.mean()
  rows.append(dict(zip(keys,vals),structures=len(g),n=int(g.n.sum()),macro_RMSE_mm=float(np.sqrt(m)),macro_MAE_mm=float(g.MAE_mm.mean()),skill=float(1-m/p) if p>0 else None,increment_R2=float(1-m/v) if v>0 else None,wins_vs_persistence=int((g.MSE_mm2<g.persistence_MSE_mm2).sum())))
 return pd.DataFrame(rows)

def paired(a,b,seed=20261101,boundary=0.,reps=10000):
 x=a[['STR_name','MSE_mm2','persistence_MSE_mm2','n']].merge(b[['STR_name','MSE_mm2','persistence_MSE_mm2','n']],on='STR_name',suffixes=('_a','_b'),validate='one_to_one').sort_values('STR_name')
 assert len(x)==len(a)==len(b)>0 and (x.n_a==x.n_b).all();np.testing.assert_allclose(x.persistence_MSE_mm2_a,x.persistence_MSE_mm2_b,atol=1e-9)
 am=x.MSE_mm2_a.to_numpy();bm=x.MSE_mm2_b.to_numpy();pm=x.persistence_MSE_mm2_a.to_numpy();den=pm.mean();assert den>0
 delta=float((bm.mean()-am.mean())/den);rng=np.random.default_rng(seed);ix=rng.integers(0,len(x),(reps,len(x)));ds=(bm[ix].mean(1)-am[ix].mean(1))/pm[ix].mean(1);rd=np.sqrt(am[ix].mean(1))-np.sqrt(bm[ix].mean(1))
 # Boundary-centered bootstrap reference, not a distribution-free exact test.
 p=float((1+np.count_nonzero(ds-delta>=delta-boundary))/(reps+1))
 return dict(structures=len(x),delta_skill=delta,skill_CI_low=float(np.quantile(ds,.025)),skill_CI_high=float(np.quantile(ds,.975)),RMSE_difference_mm=float(np.sqrt(am.mean())-np.sqrt(bm.mean())),RMSE_CI_low=float(np.quantile(rd,.025)),RMSE_CI_high=float(np.quantile(rd,.975)),wins=int((am<bm).sum()),boundary=boundary,reference_p=p)

def holm(p):
 order=np.argsort(p);out=np.zeros(len(p));running=0.
 for i,j in enumerate(order):running=max(running,(len(p)-i)*p[j]);out[j]=min(1.,running)
 return out.tolist()

def main():
 analysis_check();freeze_check();out=ROOT/'results/tables';out.mkdir(parents=True,exist_ok=True);_,a=aggregate();a.to_csv(ROOT/'results/ensemble_predictions.csv.gz',index=False)
 metrics=[];tables=[];baselines=[]
 for scope,z in [('primary2089',a[a.eligible]),('all2164',a)]:
  m=structure_metrics(z,['arm_id','block','arm','family','split']);m['scope']=scope;metrics.append(m)
  # Outer folds are not independent units. Pool held-out structures before summarizing.
  s=summarize(m,['arm_id','block','arm','family']);s['scope']=scope;tables.append(s)
  base=z[(z.arm=='A4')&(z.block=='main')].copy();base.y_pred_mm=base.persistence_mm;base.arm_id='main_A5_persistence';base.arm='A5';base.family='persistence';b=summarize(structure_metrics(base,['arm_id','block','arm','family']),['arm_id','block','arm','family']);b['scope']=scope;baselines.append(b)
 m=pd.concat(metrics);m.to_csv(out/'structure_metrics.csv',index=False);pd.concat(tables+baselines).to_csv(out/'one_step_summary.csv',index=False)
 primary=m[(m.scope=='primary2089')&(m.block=='main')];contrasts=[];familyp=[]
 for name,other,boundary in [('P1','A2',0.),('P2','A3',-.03)]:
  pp=[]
  for f in FAMILIES:
   aa=primary[(primary.arm=='A1')&(primary.family==f)];bb=primary[(primary.arm==other)&(primary.family==f)];r=paired(aa,bb,boundary=boundary);budget=-boundary*aa.persistence_MSE_mm2.mean();r['MSE_margin_mm2']=budget;r['RMSE_margin_at_comparator_mm']=float(np.sqrt(bb.MSE_mm2.mean()+budget)-np.sqrt(bb.MSE_mm2.mean()));contrasts.append(dict(comparison=name,family=f,**r));pp.append(r['reference_p'])
  familyp.append(dict(comparison=name,conjunction_p=max(pp),boundary=boundary))
 adj=holm([r['conjunction_p'] for r in familyp])
 for r,p in zip(familyp,adj):r.update(holm_p=p,passed=p<=.05)
 pd.DataFrame(contrasts).to_csv(out/'primary_backbone_contrasts.csv',index=False);pd.DataFrame(familyp).to_csv(out/'primary_conjunctions.csv',index=False)
 desc=[];mm=m[m.scope=='primary2089']
 for f in FAMILIES:
  aa=mm[(mm.block=='main')&(mm.arm=='A1')&(mm.family==f)]
  for other in ['A4']+(['A6','A7'] if f in ['mlp','gru'] else []):
   bb=mm[(mm.arm==other)&(mm.family==('xgb' if other=='A4' else f))&(mm.block!='transfer')];desc.append(dict(comparison=f'A1_minus_{other}',family=f,**paired(aa,bb)))
 pd.DataFrame(desc).drop(columns=['reference_p','boundary']).to_csv(out/'descriptive_components_tree.csv',index=False)
 cc=mm[mm.block=='transfer'];desc=[]
 for sp in sorted(cc.split.unique()):
  for f in FAMILIES:
   aa=cc[(cc.split==sp)&(cc.family==f)&(cc.arm=='A1')]
   for other,fam in [('A7',f),('A4','xgb')]:
    bb=cc[(cc.split==sp)&(cc.family==fam)&(cc.arm==other)];desc.append(dict(split=sp,family=f,comparison=f'A1_minus_{other}',**paired(aa,bb)))
 pd.DataFrame(desc).drop(columns=['reference_p','boundary']).to_csv(out/'class_contrasts.csv',index=False)
 from data import Data
 d=Data();limits=d.frame.groupby('STR_name').Cum_Load_10k.max()*.75;load=d.frame.set_index(['STR_name','Observation_Order']).Cum_Load_10k
 late=a[a.eligible & (a.block!='transfer')].copy();keep=[float(load.loc[(r.STR_name,r.Observation_Order)])>limits[r.STR_name] for r in late.itertuples()];late=late.loc[keep]
 lm=structure_metrics(late,['arm_id','arm','family']);lm.to_csv(out/'late_load_structures.csv',index=False);summarize(lm,['arm_id','arm','family']).to_csv(out/'late_load_summary.csv',index=False)
 write(ROOT/'checks/analysis_complete.json',{'time':stamp(),'jobs':605,'ensemble_rows':len(a),'primary_decisions':familyp,'scope':'Further frozen exploratory comparison; structure bootstrap conditional on shared-site sample and overlapping training sets.'})
if __name__=='__main__':main()
