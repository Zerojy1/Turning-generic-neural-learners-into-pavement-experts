"""Four-group finite background game, incremental output, no causal interpretation."""
import itertools,math,os
from common import *
from data import Data
from models import Predictor
from analysis import analysis_check
GROUPS=['memory','load','temperature','structure']
def shapley(values):
 out=np.zeros((len(values),4))
 for j in range(4):
  for mask in range(16):
   if mask&(1<<j):continue
   k=mask.bit_count();out[:,j]+=math.factorial(k)*math.factorial(3-k)/24*(values[:,mask|(1<<j)]-values[:,mask])
 return out
def permutation_reference(values):
 out=np.zeros((len(values),4))
 for perm in itertools.permutations(range(4)):
  mask=0
  for j in perm:nextmask=mask|(1<<j);out[:,j]+=(values[:,nextmask]-values[:,mask])/24;mask=nextmask
 return out
def hybrid(query,donor,mask):
 r={k:v.copy() for k,v in query.items()};dt=r['duration'][:,0];days=r['daily_mask'].sum(1).astype(int)
 if not mask&1:r['memory']=donor['memory'].copy()
 if not mask&2:
  r['history'][:,:,:2]=donor['history'][:,:,:2];r['load'][:,0:2]=donor['load'][:,0:2];r['load'][:,2]=dt;r['load'][:,3]=r['load'][:,1]/dt;r['daily_load']=r['daily_mask']*(r['load'][:,1]/dt)[:,None]
 if not mask&4:
  r['history'][:,:,2:]=donor['history'][:,:,2:];r['temperature']=donor['temperature'].copy();r['temperature'][:,2]=donor['temperature'][:,2]/donor['duration'][:,0]*dt
  r['daily_temperature'][:]=0
  for i,n in enumerate(days):
   z=donor['daily_temperature'][i,donor['daily_mask'][i]];r['daily_temperature'][i,:n]=np.interp((np.arange(n)+.5)/n,(np.arange(len(z))+.5)/len(z),z)
 if not mask&8:r['thickness']=donor['thickness'].copy()
 return r
def coalitions(d,m,qids,bids):
 n,b=bids.shape;query=d.raw(np.repeat(qids,b));donor=d.raw(bids.reshape(-1));out=np.zeros((n,16))
 for mask in range(16):
  r=hybrid(query,donor,mask);pred=m.predict_raw(r)-r['memory'][:,0];out[:,mask]=pred.reshape(n,b).mean(1,dtype=np.float64)
 np.testing.assert_allclose(out[:,15],m.predict_raw(d.raw(qids))-d.a['memory'][qids,0],atol=2e-5,rtol=2e-6);return out
def main():
 torch.set_num_threads(2);torch.set_num_interop_threads(1);analysis_check();freeze_check();d=Data();meta=json.loads((ROOT/'checks/explanation_manifest.json').read_text());jobs=specs();todo=[s for s in jobs if s['block']=='main' and s['arm'] in ['A1','A2'] and s['seed']==42];assert len(todo)==40;out=ROOT/'attribution';out.mkdir(exist_ok=True);frames=[];timings=[]
 for i,s in enumerate(todo):
  tag=s['arm_id']+'_'+s['split'];path=out/(tag+'.csv.gz');marker=out/(tag+'.json');begin=time.monotonic();write(ROOT/'queues/attribution/state.json',{'pid':os.getpid(),'completed':i,'total':40,'current':tag,'timings':timings,'updated':stamp()})
  if path.exists() and marker.exists():frames.append(pd.read_csv(path));timings.append(json.loads(marker.read_text()));continue
  f=int(s['split'][4:]);qids=np.array(sorted(q for sid in P['splits']['outer_groups'][f-1] for q in meta['selected'][sid]));background=np.tile(meta['backgrounds'][str(f)],(len(qids),1));matched=np.array([meta['matched'][str(f)][str(q)] for q in qids]);arrays={'query_ids':qids,'main_donors':background,'matched_donors':matched};records=[]
  for ref,bids in [('main',background),('load_matched',matched)]:
   values=[];phis=[]
   for seed in P['seeds']:
    row=next(r for r in jobs if r['arm_id']==s['arm_id'] and r['split']==s['split'] and r['seed']==seed);m=Predictor(row,ROOT/'runs'/row['job_id'],'cuda');v=coalitions(d,m,qids,bids);phi=shapley(v);np.testing.assert_allclose(phi,permutation_reference(v),atol=1e-10,rtol=1e-10);error=v[:,0]+phi.sum(1)-v[:,15];assert abs(error).max()<1e-8
    values.append(v);phis.append(phi)
    for qi,q in enumerate(qids):
     for gi,g in enumerate(GROUPS):records.append({'arm_id':s['arm_id'],'family':s['family'],'variant':s['variant'],'fold':f,'sample_id':int(q),'STR_name':d.groups[q],'seed':seed,'reference':ref,'group':g,'phi_mm':float(phi[qi,gi]),'baseline_mm':float(v[qi,0]),'increment_prediction_mm':float(v[qi,15]),'additivity_error_mm':float(error[qi])})
    del m
   arrays[ref+'_coalitions']=np.array(values);arrays[ref+'_phi']=np.array(phis)
  a=pd.DataFrame(records);a.to_csv(path,index=False);np.savez_compressed(out/(tag+'.npz'),**arrays);timing={'tag':tag,'seconds':time.monotonic()-begin,'queries':len(qids)};write(marker,timing);frames.append(a);timings.append(timing)
 whole=pd.concat(frames,ignore_index=True);whole.to_csv(ROOT/'results/attributions.csv.gz',index=False);ens=whole.groupby(['arm_id','reference','STR_name','sample_id','group'],as_index=False).agg(phi_mm=('phi_mm','mean'),seed_sd_mm=('phi_mm','std'));ens['abs_phi_mm']=abs(ens.phi_mm);st=ens.groupby(['arm_id','reference','STR_name','group'],as_index=False)[['phi_mm','abs_phi_mm','seed_sd_mm']].mean();st.to_csv(ROOT/'results/attribution_structure.csv',index=False);st.groupby(['arm_id','reference','group'],as_index=False)[['phi_mm','abs_phi_mm','seed_sd_mm']].mean().to_csv(ROOT/'results/tables/attribution_global.csv',index=False)
 stability=[]
 for (arm,ref,sid,q),g in whole.groupby(['arm_id','reference','STR_name','sample_id']):
  a=g.pivot(index='seed',columns='group',values='phi_mm').reindex(columns=GROUPS).to_numpy();avg=a.mean(0);diff=float(abs(a-avg).sum(1).mean());stability.append({'arm_id':arm,'reference':ref,'STR_name':sid,'sample_id':q,'seed_L1_mm':diff,'normalized_seed_L1':diff/max(abs(avg).sum(),.01)})
 stable=pd.DataFrame(stability);stable.to_csv(ROOT/'results/attribution_stability_rows.csv',index=False);stable.groupby(['arm_id','reference','STR_name'],as_index=False)[['seed_L1_mm','normalized_seed_L1']].mean().groupby(['arm_id','reference'],as_index=False)[['seed_L1_mm','normalized_seed_L1']].mean().to_csv(ROOT/'results/tables/attribution_seed_stability.csv',index=False)
 diff=[]
 for (arm,sid),g in st.groupby(['arm_id','STR_name']):
  a=g[g.reference=='main'].set_index('group').reindex(GROUPS).abs_phi_mm.to_numpy();b=g[g.reference=='load_matched'].set_index('group').reindex(GROUPS).abs_phi_mm.to_numpy();diff.append({'arm_id':arm,'STR_name':sid,'L1_mm':float(abs(a-b).sum()),'normalized_L1':float(abs(a-b).sum()/max(abs(a).sum(),.01))})
 pd.DataFrame(diff).to_csv(ROOT/'results/attribution_background_rows.csv',index=False);pd.DataFrame(diff).groupby('arm_id',as_index=False)[['L1_mm','normalized_L1']].mean().to_csv(ROOT/'results/tables/attribution_background_stability.csv',index=False)
 write(ROOT/'checks/attribution_complete.json',{'finished':stamp(),'models':8,'queries_per_model':150,'seeds':5,'backgrounds':2,'coalitions':16,'independent_24_permutations_passed':True,'causal_claim':False});write(ROOT/'queues/attribution/state.json',{'status':'complete','completed':40,'total':40,'timings':timings,'updated':stamp()})
if __name__=='__main__':main()
