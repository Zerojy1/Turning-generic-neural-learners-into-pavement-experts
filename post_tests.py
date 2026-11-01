from common import *
from analysis import structure_metrics,summarize,paired,holm
from evaluation import radius,interval_tables
from models import Predictor
from data import Data

def main():
 a=pd.DataFrame({'model':['a']*5,'STR_name':['s1']+['s2']*4,'y_true_mm':[2,4,4,4,4],'persistence_mm':[0]*5,'y_pred_mm':[1,2,2,2,2]})
 m=structure_metrics(a,['model']);s=summarize(m,['model']).iloc[0];assert abs(s.macro_RMSE_mm-np.sqrt(2.5))<1e-12 and abs(s.skill-.75)<1e-12
 b=m.copy();b.MSE_mm2*=2;r=paired(m,b,seed=19,reps=10000);assert abs(r['delta_skill']-.25)<1e-12
 # Independent resampling calculation of the boundary-centered reference p.
 rng=np.random.default_rng(19);ix=rng.integers(0,2,(10000,2));aa=m.sort_values('STR_name');bb=b.sort_values('STR_name');ds=[]
 for row in ix:ds.append(float((bb.MSE_mm2.to_numpy()[row].mean()-aa.MSE_mm2.to_numpy()[row].mean())/aa.persistence_MSE_mm2.to_numpy()[row].mean()))
 np.testing.assert_allclose([r['skill_CI_low'],r['skill_CI_high']],np.quantile(ds,[.025,.975]),atol=1e-12)
 np.testing.assert_allclose(holm([.04,.01]),[.04,.02]);np.testing.assert_allclose(holm([.03,.04]),[.06,.06])
 # Equal structure weights, not a pooled empirical quantile.
 c=pd.DataFrame({'STR_name':['s1']*100+['s2'],'y_true_mm':[0.]*101,'y_pred_mm':[.1]*100+[10.]});assert radius(c)==10.
 assert radius(c.iloc[:0]) is None
 iv=a.rename(columns={'model':'arm_id'});iv['scope']='full';iv['scenario']='test';iv['horizon']=1;iv['arm']='A1';iv['family']='x';iv['lower_mm']=iv.y_pred_mm-1;iv['upper_mm']=iv.y_pred_mm+1
 rows,sums=interval_tables(iv);assert sums.mean_structure_coverage.iloc[0]==.5 and sums.MPIW_mm.iloc[0]==2 and sums.MIS95_mm.iloc[0]==22
 # Selected checkpoint loaders exercised on all training-side smoke fits.
 d=Data();bench=json.loads((ROOT/'checks/benchmark/complete.json').read_text());replay=[]
 for r in bench['rows']:
  specrow=spec(r['tag']);m=Predictor(specrow,ROOT/'checks/smoke'/r['tag'],'cpu',selection=True);tr,val,_,_=split(d,specrow);assert set(m.pre.a['train_ids'])==set(tr)
  pred=m.predict(d,val);assert np.isfinite(pred).all();replay.append(dict(family=specrow['family'],n=len(pred)))
 write(ROOT/'checks/post_tests.json',{'passed':True,'time':stamp(),'structure_weighting':True,'boundary_bootstrap_reference':True,'Holm':True,'empirical_radius_equal_structure':True,'interval_score':True,'selected_checkpoint_replay':replay,'outer_predictions_used':False});print('POST_TESTS_PASS')
if __name__=='__main__':main()
