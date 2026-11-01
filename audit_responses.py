from common import *
from data import Data
from models import Predictor
from analysis import analysis_check

def main():
 torch.set_num_threads(2);analysis_check();freeze_check();d=Data();rows=[];parameters=[]
 for s in specs():
  if s['block']!='main' or s['arm'] not in ['A1','A3']:continue
  ids=split(d,s)[-1];ids=ids[d.meta.eligible.iloc[ids].to_numpy(bool)];m=Predictor(s,ROOT/'runs'/s['job_id'],'cuda');raw=d.raw(ids);base=m.predict_raw(raw);details=m.intermediates(raw)
  trainraw=d.raw(m.pre.a['train_ids']);tn=trainraw['daily_load'].sum(1);tt=trainraw['daily_temperature'][trainraw['daily_mask']];nmin,nmax=float(tn.min()),float(tn.max());tmin,tmax=float(tt.min()),float(tt.max())
  for j,k in enumerate(ids):parameters.append({'job_id':s['job_id'],'arm':s['arm'],'family':s['family'],'fold':s['split'],'seed':s['seed'],'STR_name':d.groups[k],'sample_id':int(k),**{n:float(details[n][j]) for n in ['a_mm_per_10k','gamma','b_mm','G_mm','current_Q']}})
  for perturb in ['load10','load50','temp5']:
   r={k:v.copy() for k,v in raw.items()}
   if perturb.startswith('load'):factor=1.1 if perturb=='load10' else 1.5;r['daily_load']*=factor;r['load'][:,1]*=factor;r['load'][:,3]*=factor
   else:r['daily_temperature']+=5;r['temperature'][:,:2]+=5
   pred=m.predict_raw(r);delta=pred-base;tol=1e-6*np.maximum(1.,abs(base));changed=m.intermediates(r)
   for name in ['context','b_mm','a_mm_per_10k']:np.testing.assert_allclose(changed[name],details[name],atol=1e-7,rtol=0)
   for j,k in enumerate(ids):
    temp=r['daily_temperature'][j,r['daily_mask'][j]];dn=float(r['daily_load'][j].sum());rows.append({'job_id':s['job_id'],'arm':s['arm'],'family':s['family'],'seed':s['seed'],'fold':s['split'],'STR_name':d.groups[k],'sample_id':int(k),'perturbation':perturb,'prediction_mm':float(base[j]),'change_mm':float(delta[j]),'tolerance_mm':float(tol[j]),'raw_decrease':bool(delta[j]<0),'decrease':bool(delta[j]<-tol[j]),'required_monotonic':s['arm']=='A1','load_within_training_range':nmin<=dn<=nmax,'temperature_within_training_range':bool(temp.min()>=tmin and temp.max()<=tmax),'training_deltaN_min':nmin,'training_deltaN_max':nmax,'training_temp_min':tmin,'training_temp_max':tmax})
 a=pd.DataFrame(rows);a.to_csv(ROOT/'results/response_rows.csv.gz',index=False);pd.DataFrame(parameters).to_csv(ROOT/'results/engineering_terms.csv.gz',index=False)
 summary=a.groupby(['arm','family','perturbation'],as_index=False).agg(n=('change_mm','size'),decreases=('decrease','sum'),raw_decreases=('raw_decrease','sum'),mean_change_mm=('change_mm','mean'),min_change_mm=('change_mm','min'),median_change_mm=('change_mm','median'),max_change_mm=('change_mm','max'));summary['violation_fraction']=summary.decreases/summary.n;summary.to_csv(ROOT/'results/tables/response_audit.csv',index=False)
 a.groupby(['arm','family','STR_name','perturbation'],as_index=False).agg(n=('change_mm','size'),decreases=('decrease','sum'),mean_change_mm=('change_mm','mean'),min_change_mm=('change_mm','min')).to_csv(ROOT/'results/tables/response_structure_audit.csv',index=False)
 assert len(a)==2*4*5*2089*3;assert not a[a.required_monotonic].decrease.any();write(ROOT/'checks/responses_complete.json',{'passed':True,'time':stamp(),'rows':len(a),'scope':'fixed history, duration, grid and preprocessing; conditional final-output property, not causal mechanism or globally monotone rollout'})
if __name__=='__main__':main()
