from common import *
KEYS=['history','memory','thickness','temperature','load','duration','daily_temperature','daily_load','daily_mask']

def make_arrays(frame=None):
 frame=(pd.read_csv(ROOT/'data/panel.csv') if frame is None else frame.copy()).sort_values(['STR_num','Observation_Order']);audit=pd.read_csv(ROOT/'data/daily_target_audit.csv');dates={(r.STR_name,int(r.Observation_Order)):int(r.date_day) if pd.notna(r.date_day) else -1 for r in audit.itertuples()}
 daily=pd.read_csv(ROOT/'data/daily.csv');daymap={(r.STR_name,int(r.day)):r for r in daily.itertuples()};rows=[];meta=[];maxdays=1
 for sid,g in frame.groupby('STR_name',sort=False):
  g=g.sort_values('Observation_Order').reset_index(drop=True);env=g[ENV].to_numpy(np.float32);yy=g.Rutting_Depth_mm.to_numpy(np.float32)
  for i in range(1,len(g)):
   r=g.iloc[i];lo=dates[(sid,int(g.Observation_Order.iloc[i-1]))];hi=dates[(sid,int(r.Observation_Order))];eligible=lo>=0 and hi>lo
   days=list(range(lo,hi)) if eligible else []
   if days:
    a=[daymap[(sid,dd)] for dd in days];temp=np.array([x.temp_C for x in a]);mx=max(x.max_C for x in a);heat=sum(x.degree_hours25 for x in a);cov=np.mean([x.observed_coverage for x in a]);fallback=float(any(not x.internal for x in a))
   else:temp=np.array([float(r.Avg_Temp_C)]);mx=float(temp[0]);heat=max(mx-25,0)*24;cov=0.;fallback=1.
   dt=len(temp);dn=float(r.Cum_Load_10k-g.Cum_Load_10k.iloc[i-1]);assert dn>=0
   row={'history':env[np.maximum(np.arange(i-6,i),0)],'memory':np.array([yy[i-1],yy[:i].max()]),'thickness':r[STRUCT].to_numpy(np.float32),'temperature':np.array([temp.mean(),mx,heat,cov,fallback]),'load':np.array([g.Cum_Load_10k.iloc[i-1],dn,dt,dn/dt]),'duration':np.array([dt]),'daily_temperature':temp,'daily_load':np.full(dt,dn/dt),'daily_mask':np.ones(dt,bool),'y':yy[i]};rows.append(row);maxdays=max(maxdays,dt)
   meta.append({'sample_id':len(rows)-1,'STR_name':sid,'Observation_Order':int(r.Observation_Order),'origin_order':int(g.Observation_Order.iloc[i-1]),'eligible':bool(eligible),'start_day':lo,'end_day':hi,'days':dt,'Cum_Load_10k':float(r.Cum_Load_10k),'Delta_Load_10k':dn,'temp_mean_C':float(temp.mean()),'temp_max_C':float(mx),'degree_hours25':float(heat),'coverage':float(cov),'fallback':bool(fallback),'date_proxy':True})
 arrays={}
 for k in KEYS+['y']:
  if k.startswith('daily_'):
   arrays[k]=np.array([np.pad(r[k],(0,maxdays-len(r[k]))) for r in rows],dtype=bool if k=='daily_mask' else np.float32)
  else:arrays[k]=np.array([r[k] for r in rows],dtype=np.float32)
 return arrays,pd.DataFrame(meta)

class Data:
 def __init__(self):
  self.a=dict(np.load(ROOT/'data/arrays.npz'));self.y=self.a.pop('y');self.meta=pd.read_csv(ROOT/'data/targets.csv');self.ids=np.arange(len(self.y));self.groups=self.meta.STR_name.to_numpy();self.frame=pd.read_csv(ROOT/'data/panel.csv')
  assert len(self.y)==2734 and np.array_equal(self.ids,self.meta.sample_id)
 def raw(self,ids):return {k:v[ids].copy() for k,v in self.a.items()}

def ablate(raw,s):return {k:v.copy() for k,v in raw.items()}

class Preprocess:
 def __init__(self,d=None,ids=None,s=None,saved=None):
  if saved is not None:self.a=dict(np.load(saved));return
  r=d.raw(ids);names=d.groups[ids];first=np.array([np.flatnonzero(names==g)[0] for g in np.unique(names)]);positive=r['thickness'][first];positive=positive[positive>0];self.a={'h0':np.array(float(np.median(positive)))}
  daily=pd.read_csv(ROOT/'data/daily.csv');daily=daily[daily.STR_name.isin(np.unique(names))];internal=daily[daily.internal.astype(bool)];chosen=internal if len(internal) else daily
  self.a['temp_sigma']=np.array(max(float(chosen.temp_C.to_numpy().std()),1.));self.a['sigma_uses_internal']=np.array(bool(len(internal)))
  memory=np.c_[r['memory'][:,0],r['memory'][:,1]-r['memory'][:,0]]
  for k,v in [('history',r['history'].reshape(-1,5)),('memory',memory),('duration',r['duration']),('flat',np.log1p(r['thickness']/self.a['h0'])),('current',self.current(r))]:
   self.a[k+'_mean']=v.mean(0,dtype=np.float64);sd=v.std(0,dtype=np.float64);self.a[k+'_scale']=np.where(sd>1e-8,sd,1.)
  self.a['target_scale']=np.array(max(float(d.y[ids].std(dtype=np.float64)),1e-6));self.a['target_center']=np.array(float(((d.y[ids]-offset(r,s))/self.a['target_scale']).mean(dtype=np.float64)));self.a['train_ids']=np.asarray(ids)
 def current(self,r):
  mask=r['daily_mask'];q=(np.maximum(r['daily_temperature']-25,0)*mask).sum(1)/mask.sum(1)/self.a['temp_sigma'];dn=(r['daily_load']*mask).sum(1)
  return np.c_[dn,q]
 def transform(self,raw,s):
  r=raw;m=np.c_[r['memory'][:,0],r['memory'][:,1]-r['memory'][:,0]];out={}
  for k,v in [('history',r['history']),('memory',m),('duration',r['duration']),('flat',np.log1p(r['thickness']/self.a['h0'])),('current',self.current(r))]:out[k]=((v-self.a[k+'_mean'])/self.a[k+'_scale']).astype(np.float32)
  out.update(thickness=(r['thickness']/self.a['h0']).astype(np.float32),daily_temperature=r['daily_temperature'],daily_load=r['daily_load'],daily_mask=r['daily_mask'].astype(np.float32));return out
 def save(self,path):np.savez(path,**self.a)

def tree_features(raw,pre):
 r=pre.transform(raw,{'variant':'A4'});return np.c_[r['history'].reshape(len(raw['history']),-1),r['memory'],r['flat'],r['duration'],r['current']].astype(np.float32)
