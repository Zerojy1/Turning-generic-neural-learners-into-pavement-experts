"""Outcome-independent scenario selection; recursive prediction never consumes future rutting."""
from common import *
from data import Data

class Scenarios:
 def __init__(self,d,names):
  self.d=d;self.origins=[];self.missing=[];lookup={(r.STR_name,int(r.Observation_Order)):int(r.sample_id) for r in d.meta.itertuples()};daily=pd.read_csv(ROOT/'data/daily.csv');self.daymap={(r.STR_name,int(r.day)):r for r in daily.itertuples()}
  for sid in names:
   g=d.frame[d.frame.STR_name==sid].sort_values('Observation_Order').reset_index(drop=True);env=g[ENV].to_numpy(np.float32);y=g.Rutting_Depth_mm.to_numpy(np.float32);orders=g.Observation_Order.to_numpy(int)
   for i in range(2,len(g)-4):
    ids=[lookup[(sid,int(orders[j]))] for j in range(i,i+5)]
    if not all(d.meta.eligible.iloc[k] for k in ids):continue
    r=d.raw([ids[1]]);temps=[d.a['temperature'][lookup[(sid,int(orders[j]))],0] for j in range(max(1,i-2),i+1)];self.origins.append({'STR_name':sid,'origin_order':int(orders[i]),'target_ids':ids[1:],'history':r['history'][0],'memory':np.array([y[i],y[:i+1].max()],np.float32),'load_rate':float(d.a['load'][ids[0],3]),'temperature':float(np.mean(temps))})
   for j in range(2,len(g)):
    k=lookup[(sid,int(orders[j]))];previous=lookup[(sid,int(orders[j-1]))]
    if not d.meta.eligible.iloc[k] or not d.meta.eligible.iloc[previous]:continue
    raw=d.raw([k]);lo=int(d.meta.start_day.iloc[previous]);hi=int(d.meta.end_day.iloc[k]);raw['history'][0]=env[np.maximum(np.arange(j-7,j-1),0)];raw['memory'][0]=[y[j-2],y[:j-1].max()];raw=self.interval(raw,sid,lo,hi,float(env[j-2,0]),float(env[j,0]-env[j-2,0]))
    self.missing.append({'STR_name':sid,'origin_order':int(orders[j-2]),'sample_id':k,'raw':raw})
 def interval(self,raw,sid,lo,hi,nprev,dn):
  rr={k:v.copy() for k,v in raw.items()};a=[self.daymap[(sid,dd)] for dd in range(lo,hi)];dt=len(a);assert dt>0
  rr['temperature'][0]=[np.mean([x.temp_C for x in a]),max(x.max_C for x in a),sum(x.degree_hours25 for x in a),np.mean([x.observed_coverage for x in a]),float(any(not x.internal for x in a))];rr['load'][0]=[nprev,dn,dt,dn/dt];rr['duration'][0]=dt
  rr['daily_temperature']=np.array([[x.temp_C for x in a]],np.float32);rr['daily_load']=np.full((1,dt),dn/dt,np.float32);rr['daily_mask']=np.ones((1,dt),bool);return rr
 def manifest(self):return {'origins':[{k:r[k] for k in ['STR_name','origin_order','target_ids']} for r in self.origins],'missing':[{k:r[k] for k in ['STR_name','origin_order','sample_id']} for r in self.missing]}
 def predict(self,predict):
  from scenario_predict import predict_scenarios
  return predict_scenarios(self,predict)


def stack_raw(rows):
 out={}
 for k in rows[0]:
  if k.startswith('daily_'):
   size=max(a[k].shape[1] for a in rows);out[k]=np.concatenate([np.pad(a[k],((0,0),(0,size-a[k].shape[1]))) for a in rows])
  else:out[k]=np.concatenate([a[k] for a in rows])
 return out

def freeze_manifests():
 d=Data();origins={str(f):Scenarios(d,P['splits']['outer_groups'][f-1]).manifest() for f in range(1,6)};write(ROOT/'checks/origin_manifest.json',origins)
 rng=np.random.default_rng(20261101);selected={};background={};matched={}
 for sid in sorted(set(d.groups)-set(P['splits']['anchors']),key=lambda s:int(s[3:])):
  ids=d.ids[(d.groups==sid)&d.meta.eligible.to_numpy(bool)];selected[sid]=sorted(rng.choice(ids,10,replace=False).tolist())
 for f in range(1,6):
  s=next(s for s in specs() if s['split']==f'fold{f}');tr=split(d,s)[2];rr=np.random.default_rng(20261101+f);donors=[];matched[str(f)]={};qids=[q for sid in P['splits']['outer_groups'][f-1] for q in selected[sid]]
  for sid in sorted(np.unique(d.groups[tr]),key=lambda s:int(s[3:])):donors.extend(rr.choice(tr[d.groups[tr]==sid],2,replace=False).tolist())
  background[str(f)]=donors
  for q in qids:
   ids=[]
   for sid in sorted(np.unique(d.groups[tr]),key=lambda s:int(s[3:])):
    pool=tr[d.groups[tr]==sid];ids.append(int(pool[np.argmin(abs(d.a['load'][pool,0]-d.a['load'][q,0]))]))
   matched[str(f)][str(q)]=ids
 write(ROOT/'checks/explanation_manifest.json',{'selected':selected,'backgrounds':background,'matched':matched,'seed':20261101,'queries':150,'selection_uses_outcomes':False})
 print('MANIFESTS',sum(len(v['origins']) for v in origins.values()),sum(len(v['missing']) for v in origins.values()))
if __name__=='__main__':freeze_manifests()
