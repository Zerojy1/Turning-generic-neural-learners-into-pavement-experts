from common import *
from analysis import analysis_check,summarize

def main():
 analysis_check();out=ROOT/'results/tables';rows=[]
 for s in specs():
  folder=ROOT/'runs'/s['job_id'];c=json.loads((folder/'complete.json').read_text());sel=json.loads((folder/'selection.json').read_text())['selected'];rows.append(dict(job_id=s['job_id'],arm_id=s['arm_id'],arm=s['arm'],family=s['family'],block=s['block'],split=s['split'],seed=s['seed'],seconds=c['seconds'],parameter_count=c['parameter_count'],peak_gpu_MiB=c['peak_gpu_bytes']/2**20,inference_256_ms=float(np.median(c['inference_256_seconds'])*1000),selected_candidate=sel['candidate'],selected_epochs_or_trees=sel.get('epoch',sel.get('trees')),replay_max_abs_mm=c['replay_max_abs']))
 a=pd.DataFrame(rows);a.to_csv(out/'execution_costs_and_selection.csv',index=False);a.groupby(['arm_id','arm','family','block'],as_index=False).agg(jobs=('job_id','size'),mean_seconds=('seconds','mean'),total_seconds=('seconds','sum'),parameters=('parameter_count','mean'),max_gpu_MiB=('peak_gpu_MiB','max'),median_inference_256_ms=('inference_256_ms','median')).to_csv(out/'cost_summary.csv',index=False)
 m=pd.read_csv(out/'structure_metrics.csv');m=m[m.block=='transfer'];summarize(m,['scope','arm_id','arm','family','split']).to_csv(out/'class_summary.csv',index=False)
 e=pd.read_csv(ROOT/'results/engineering_terms.csv.gz');e['abs_b_mm']=abs(e.b_mm);e['abs_G_mm']=abs(e.G_mm)
 e.groupby(['arm','family','seed','fold'] if 'fold' in e else ['arm','family','seed'],as_index=False).agg(mean_a=('a_mm_per_10k','mean'),median_a=('a_mm_per_10k','median'),gamma=('gamma','mean'),mean_abs_b_mm=('abs_b_mm','mean'),mean_abs_G_mm=('abs_G_mm','mean')).to_csv(out/'engineering_terms_summary.csv',index=False)
 write(ROOT/'checks/finish_analysis_complete.json',{'time':stamp(),'jobs':len(a),'costs':'training selection plus refit; parameter budget and latency descriptive','interpretation':'b and G are model terms, not measured recoverable/permanent strain; attribution not causal'})
if __name__=='__main__':main()
