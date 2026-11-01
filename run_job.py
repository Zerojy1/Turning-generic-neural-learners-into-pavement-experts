import argparse,os,resource
from common import *
from data import Data,tree_features,Preprocess
from models import Fit,Predictor
import xgboost as xgb

def run(tag,device,smoke_epochs=0):
 begin=time.monotonic();freeze_check();s=spec(tag);d=Data();train,val,final,test=split(d,s);prov=provenance();folder=ROOT/('checks/smoke' if smoke_epochs else 'runs')/tag;folder.mkdir(parents=True,exist_ok=True)
 if (folder/'complete.json').exists():assert matches_recorded_provenance(json.loads((folder/'complete.json').read_text())['provenance']);print('CACHED',tag);return
 if smoke_epochs:final=train;test=val
 if device=='cuda':torch.cuda.reset_peak_memory_stats()
 write(folder/'job.json',{'spec':s,'pid':os.getpid(),'started':stamp(),'provenance':prov,'smoke_epochs':smoke_epochs})
 membership=d.meta[['sample_id','STR_name','Observation_Order']].copy()
 for n,ids in [('inner_train',train),('inner_validation',val),('final_train',final),('outer_test',test)]:membership[n]=np.isin(d.ids,ids)
 membership.to_csv(folder/'membership.csv',index=False);candidates=[]
 if s['family']=='xgb':
  raw=d.raw(d.ids);pre=Preprocess(d,train,s);X=tree_features(raw,pre);off=offset(raw,s);target=d.y-off;cfg=P['training']['xgb'];params={k:v for k,v in cfg.items() if k not in ['depths','rounds','early_stopping_rounds']};params.update(objective='reg:squarederror',seed=s['seed']);dt=xgb.DMatrix(X[train],label=target[train]);dv=xgb.DMatrix(X[val],label=target[val])
  def metric(pred,matrix):return 'macro_mse',macro_mse(matrix.get_label(),pred,d.groups[val])
  for ci,depth in enumerate(cfg['depths']):
   hist={};model=xgb.train(dict(params,max_depth=depth),dt,num_boost_round=10 if smoke_epochs else cfg['rounds'],evals=[(dv,'inner')],custom_metric=metric,early_stopping_rounds=3 if smoke_epochs else cfg['early_stopping_rounds'],maximize=False,verbose_eval=False,evals_result=hist);n=int(model.best_iteration)+1;score=macro_mse(d.y[val],model.predict(dv,iteration_range=(0,n))+off[val],d.groups[val]);write(folder/f'selection_{ci}.json',hist);model.save_model(folder/f'selection_{ci}_model.json');candidates.append({'candidate':ci,'max_depth':depth,'trees':n,'score':score})
  selected=min(candidates,key=lambda x:(x['score'],x['candidate']));pre.save(folder/'selection_preprocessing.npz');pre_final=Preprocess(d,final,s);Xf=tree_features(raw,pre_final);(folder/'refit').mkdir(exist_ok=True);pre_final.save(folder/'refit/preprocessing.npz');model=xgb.train(dict(params,max_depth=selected['max_depth']),xgb.DMatrix(Xf[final],label=target[final]),num_boost_round=selected['trees']);model.save_model(folder/'model.json');pred=model.predict(xgb.DMatrix(Xf[test]))+off[test];nparams=None
 else:
  maximum=smoke_epochs or 220
  for ci,lr in enumerate(P['training']['nn']['learning_rates']):
   fit=Fit(d,train,s,lr,device,maximum);res=fit.train(maximum,folder/f'selection_{ci}',val);candidates.append(dict(res,candidate=ci,learning_rate=lr));del fit
  selected=min(candidates,key=lambda x:(x['score'],x['candidate']));fit=Fit(d,final,s,selected['learning_rate'],device,maximum);fit.train(selected['epoch'],folder/'refit');fit.save(folder/'refit');pred=fit.predict(d,test);nparams=sum(p.numel() for p in fit.net.parameters());del fit
 assert np.isfinite(pred).all();other=Predictor(s,folder,device);replay=other.predict(d,test);np.testing.assert_allclose(pred,replay,atol=3e-5,rtol=1e-6)
 raw=d.raw(test);frame=d.meta.iloc[test].copy();frame['y_true_mm']=d.y[test];frame['y_previous_mm']=raw['memory'][:,0];frame['p_previous_mm']=raw['memory'][:,1];frame['y_pred_mm']=pred;frame.to_csv(folder/'predictions.csv',index=False)
 details=other.intermediates(raw);np.savez_compressed(folder/'intermediates.npz',sample_id=test,**details)
 lat=[]
 for _ in range(4):
  tick=time.perf_counter();other.predict_raw({k:v[:min(256,len(v))] for k,v in raw.items()});lat.append(time.perf_counter()-tick)
 write(folder/'selection.json',{'candidates':candidates,'selected':selected,'reason':'minimum inner structure balanced MSE; ties candidate order; outer never used'})
 write(folder/'complete.json',{'tag':tag,'family':s['family'],'block':s['block'],'finished':stamp(),'seconds':time.monotonic()-begin,'provenance':prov,'parameter_count':nparams,'smoke_epochs':smoke_epochs,'replay_max_abs':float(np.max(abs(replay-pred))),'peak_gpu_bytes':int(torch.cuda.max_memory_allocated()) if device=='cuda' else 0,'peak_host_rss_KiB':int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),'inference_256_seconds':lat[1:]});print('COMPLETE',tag,round(time.monotonic()-begin,2),flush=True)
if __name__=='__main__':
 torch.set_num_threads(2);torch.set_num_interop_threads(1);a=argparse.ArgumentParser();a.add_argument('--tag',required=True);a.add_argument('--device',default='cuda');a.add_argument('--smoke-epochs',type=int,default=0);v=a.parse_args();run(v.tag,v.device,v.smoke_epochs)
