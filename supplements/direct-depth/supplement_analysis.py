"""Pre-frozen supplemental comparisons; original EKI decisions remain unchanged."""
from common import *
from analysis import aggregate,structure_metrics,summarize,paired,holm,FAMILIES
from data import Data

def main():
 freeze_check();out=ROOT/'results/tables';out.mkdir(parents=True,exist_ok=True);seeds,new=aggregate();seeds.to_csv(ROOT/'results/new_seed_predictions.csv.gz',index=False)
 old=pd.read_csv(ROOT/'base_reference/results/ensemble_predictions.csv.gz');old=old[(old.block=='main')&old.arm.isin(['A1','A2','A4'])].copy()
 for family in FAMILIES:
  a=new[new.family==family].sort_values('sample_id')
  for arm in ['A1','A2']:
   b=old[(old.arm==arm)&(old.family==family)].sort_values('sample_id');assert a.sample_id.tolist()==b.sample_id.tolist();np.testing.assert_allclose(a.y_true_mm,b.y_true_mm,atol=1e-9);np.testing.assert_allclose(a.persistence_mm,b.persistence_mm,atol=1e-9)
 a=pd.concat([old,new],ignore_index=True);base=a[(a.arm=='A4')].copy();base.y_pred_mm=base.persistence_mm;base.arm='A5';base.arm_id='main_A5_persistence';base.family='persistence';a=pd.concat([a,base],ignore_index=True);a.to_csv(ROOT/'results/ensemble_predictions.csv.gz',index=False)
 mets=[];sums=[];contrasts=[]
 for scope,z in [('primary2089',a[a.eligible]),('all2164',a)]:
  m=structure_metrics(z,['arm_id','arm','family']);m['scope']=scope;mets.append(m);sm=summarize(m,['arm_id','arm','family']);sm['scope']=scope;sums.append(sm)
  for family in FAMILIES:
   for left,right in [('A1','A0'),('A2','A0'),('A1','A2')]:
    x=m[(m.arm==left)&(m.family==family)];y=m[(m.arm==right)&(m.family==family)];r=paired(x,y);r['RMSE_reduction_percent']=100*(1-np.sqrt(x.MSE_mm2.mean()/y.MSE_mm2.mean()));contrasts.append(dict(scope=scope,family=family,comparison=f'{left}_minus_{right}',**r))
 pd.concat(mets).to_csv(out/'structure_metrics.csv',index=False);summary=pd.concat(sums);summary.to_csv(out/'one_step_summary.csv',index=False);co=pd.DataFrame(contrasts);co.to_csv(out/'paired_contrasts.csv',index=False)
 primary=co[(co.scope=='primary2089')&(co.comparison=='A1_minus_A0')].copy();primary['holm_p']=holm(primary.reference_p.to_list());primary['positive_and_adjusted_passing']=(primary.delta_skill>0)&(primary.holm_p<=.05);primary.to_csv(out/'supplementary_primary_contrasts.csv',index=False)
 write(ROOT/'results/supplementary_decision.json',{'scope':'new exploratory supplement only; does not replace original P1/P2','passed_all_four':bool(primary.positive_and_adjusted_passing.all()),'backbones':primary.to_dict('records')})
 table=[]
 for f in FAMILIES:
  s=summary[(summary.scope=='primary2089')&(summary.family==f)].set_index('arm');r=primary[primary.family==f].iloc[0];table.append(dict(family=f,direct_RMSE_mm=s.loc['A0','macro_RMSE_mm'],residual_RMSE_mm=s.loc['A2','macro_RMSE_mm'],EKI_RMSE_mm=s.loc['A1','macro_RMSE_mm'],EKI_vs_direct_RMSE_reduction_percent=r.RMSE_reduction_percent,EKI_vs_direct_delta_skill=r.delta_skill,delta_skill_CI_low=r.skill_CI_low,delta_skill_CI_high=r.skill_CI_high,holm_p=r.holm_p,wins_out_of_15=r.wins))
 pd.DataFrame(table).to_csv(out/'three_level_main_table.csv',index=False)
 d=Data();limits=d.frame.groupby('STR_name').Cum_Load_10k.max()*.75;loads=d.frame.set_index(['STR_name','Observation_Order']).Cum_Load_10k;z=a[a.eligible].copy();z=z.loc[[loads.loc[(r.STR_name,r.Observation_Order)]>limits[r.STR_name] for r in z.itertuples()]];m=structure_metrics(z,['arm_id','arm','family']);m.to_csv(out/'late_load_structures.csv',index=False);summarize(m,['arm_id','arm','family']).to_csv(out/'late_load_summary.csv',index=False)
 write(ROOT/'checks/supplement_analysis_complete.json',{'passed':True,'time':stamp(),'new_models':100,'matched_primary_targets':2089,'all_targets':2164})
if __name__=='__main__':main()
