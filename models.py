import math,time
from common import *
from data import Preprocess,tree_features
from torch import nn
import torch.nn.functional as F
import xgboost as xgb

class Structure(nn.Module):
 def __init__(self):
  super().__init__();self.token=nn.Sequential(nn.Linear(19,32),nn.ReLU(),nn.Linear(32,16));self.fuse=nn.Linear(69,16)
  self.register_buffer('ident',torch.eye(18));self.register_buffer('positions',torch.tensor([s['position_id'] for s in SLOTS]))
 def forward(self,h):
  tokens=self.token(torch.cat([self.ident.unsqueeze(0).expand(len(h),-1,-1),torch.log1p(h).unsqueeze(-1)],-1));pools=[];heights=[]
  for p in range(4):
   w=h*(self.positions==p);height=w.sum(1,keepdim=True);pools.append((tokens*w.unsqueeze(-1)).sum(1)/height.clamp_min(1e-12));heights.append(height)
  hh=torch.cat(heights,1);return self.fuse(torch.cat(pools+[torch.log1p(hh),torch.log1p(h.sum(1,keepdim=True))],1))

class Network(nn.Module):
 def __init__(self,s,temp_sigma=1.,target_scale=1.):
  super().__init__();self.s=s;self.generic=s['arm']=='A2';f=s['family'];self.family=f
  if f=='mlp':self.backbone=nn.Sequential(nn.Flatten(),nn.Linear(30,128),nn.ReLU(),nn.Dropout(.05),nn.Linear(128,64))
  elif f=='gru':self.backbone=nn.GRU(5,64,batch_first=True)
  elif f=='lstm':self.backbone=nn.LSTM(5,64,batch_first=True)
  elif f=='transformer':
   self.projection=nn.Linear(5,64);layer=nn.TransformerEncoderLayer(64,4,128,.05,'relu',batch_first=True,norm_first=True);self.backbone=nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
   p=torch.arange(6,dtype=torch.float32)[:,None];div=torch.exp(torch.arange(0,64,2,dtype=torch.float32)*(-math.log(10000.)/64));pe=torch.zeros(1,6,64);pe[0,:,0::2]=torch.sin(p*div);pe[0,:,1::2]=torch.cos(p*div);self.register_buffer('positions',pe)
  else:raise ValueError(f)
  if self.generic:self.head=nn.Sequential(nn.Linear(87,57),nn.ReLU(),nn.Dropout(.05),nn.Linear(57,1))
  else:
   self.structure=nn.Sequential(nn.Linear(18,65),nn.ReLU(),nn.Linear(65,16)) if s['arm']=='A7' else Structure()
   self.head=nn.Sequential(nn.Linear(83,32),nn.ReLU(),nn.Dropout(.05),nn.Linear(32,2))
   nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)
   with torch.no_grad():self.head[-1].bias[1]=1. if s['arm']=='A3' else math.log(math.expm1(1.))
   if s['arm']=='A6':self.register_buffer('gamma_raw',torch.tensor(0.))
   else:self.gamma_raw=nn.Parameter(torch.tensor(.1 if s['arm']=='A3' else math.log(math.expm1(.1))))
  self.register_buffer('temp_sigma',torch.tensor(float(temp_sigma)));self.register_buffer('target_scale',torch.tensor(float(target_scale)))
 def forward(self,r,details=False):
  x=r['history']
  if self.family=='mlp':h=self.backbone(x)
  elif self.family=='gru':h=self.backbone(x)[1][-1]
  elif self.family=='lstm':h=self.backbone(x)[1][0][-1]
  else:h=self.backbone(self.projection(x)+self.positions)[:,-1]
  if self.generic:
   pred=self.head(torch.cat([h,r['memory'],r['flat'],r['duration'],r['current']],1)).squeeze(1);out={'history':h}
  else:
   z=self.structure(torch.log1p(r['thickness'])) if self.s['arm']=='A7' else self.structure(r['thickness'])
   context=torch.cat([h,r['memory'],z,r['duration']],1);b,ar=self.head(context).unbind(1)
   a=.01*(ar if self.s['arm']=='A3' else F.softplus(ar));gamma=self.gamma_raw if self.s['arm'] in ['A3','A6'] else F.softplus(self.gamma_raw)
   q=F.relu(r['daily_temperature']-25)/self.temp_sigma;factor=1+gamma*q;G=(a[:,None]*factor*r['daily_load']*r['daily_mask']).sum(1);pred=b+G/self.target_scale
   out={'history':h,'structure':z,'context':context,'base_normalized':b,'a_mm_per_10k':a,'gamma':gamma.expand_as(a),'G_mm':G,'current_Q':(q*r['daily_mask']).sum(1)/r['daily_mask'].sum(1)}
  return (pred,out) if details else pred

def tensors(r,device):return {k:torch.as_tensor(v,dtype=torch.float32,device=device) for k,v in r.items()}
class Predictor:
 def __init__(self,s,folder,device='cpu',selection=False):
  self.s=s;self.device=device
  chosen=json.loads((folder/'selection.json').read_text())['selected'] if selection else None
  modeldir=folder/f"selection_{chosen['candidate']}" if selection else folder/'refit'
  self.pre=Preprocess(saved=(folder/'selection_preprocessing.npz' if s['family']=='xgb' and selection else modeldir/'preprocessing.npz'))
  self.tree_limit=int(chosen['trees']) if s['family']=='xgb' and selection else 0
  if s['family']=='xgb':self.booster=xgb.Booster();self.booster.load_model(folder/f"selection_{chosen['candidate']}_model.json" if selection else folder/'model.json');self.booster.set_param({'nthread':2})
  else:
   self.net=Network(s,self.pre.a['temp_sigma'],self.pre.a['target_scale']).to(device);self.net.load_state_dict(torch.load(modeldir/('best_weights.pt' if selection else 'weights.pt'),map_location=device,weights_only=True));self.net.eval()
 def predict_raw(self,raw):
  if self.s['family']=='xgb':return self.booster.predict(xgb.DMatrix(tree_features(raw,self.pre)),iteration_range=(0,self.tree_limit))+offset(raw,self.s)
  self.net.eval();out=[]
  with torch.no_grad():
   for start in range(0,len(raw['memory']),512):
    r={k:v[start:start+512] for k,v in raw.items()};out.append(self.net(tensors(self.pre.transform(r,self.s),self.device)).cpu().numpy())
  return (np.concatenate(out)+self.pre.a['target_center'])*self.pre.a['target_scale']+offset(raw,self.s)
 def predict(self,d,ids):return self.predict_raw(d.raw(ids))
 def intermediates(self,raw):
  if self.s['family']=='xgb':return {}
  self.net.eval();out={}
  with torch.no_grad():
   for start in range(0,len(raw['memory']),512):
    rr={k:v[start:start+512] for k,v in raw.items()};_,z=self.net(tensors(self.pre.transform(rr,self.s),self.device),details=True)
    for k,v in z.items():out.setdefault(k,[]).append(v.detach().cpu().numpy())
  out={k:np.concatenate(v) for k,v in out.items()}
  if 'base_normalized' in out:out['b_mm']=(out['base_normalized']+self.pre.a['target_center'])*self.pre.a['target_scale']
  return out

class Fit(Predictor):
 def __init__(self,d,ids,s,lr,device='cuda',maximum=220):
  seed_all(s['seed']);self.s=s;self.device=device;self.data=d;self.ids=np.asarray(ids);self.pre=Preprocess(d,ids,s);self.net=Network(s,self.pre.a['temp_sigma'],self.pre.a['target_scale']).to(device)
  self.opt=torch.optim.AdamW(self.net.parameters(),lr=lr,weight_decay=1e-4);self.scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(self.opt,T_max=maximum,eta_min=lr*.05)
  raw=d.raw(ids);self.arrays=tensors(self.pre.transform(raw,s),device);self.target=torch.as_tensor((d.y[ids]-offset(raw,s))/self.pre.a['target_scale']-self.pre.a['target_center'],dtype=torch.float32,device=device)
 def train(self,epochs,folder,val=None):
  folder.mkdir(parents=True,exist_ok=True);history=[];best=float('inf');best_epoch=0;begin=time.monotonic();self.pre.save(folder/'preprocessing.npz')
  for epoch in range(1,epochs+1):
   self.net.train();order=torch.randperm(len(self.ids),device=self.device);total=0.
   for ix in order.split(64):
    self.opt.zero_grad(set_to_none=True);pred=self.net({k:v[ix] for k,v in self.arrays.items()});loss=((pred-self.target[ix])**2).mean()
    if not torch.isfinite(loss):raise ValueError('nonfinite loss')
    loss.backward();nn.utils.clip_grad_norm_(self.net.parameters(),1.);self.opt.step();total+=float(loss.detach())*len(ix)
   self.scheduler.step();item={'epoch':epoch,'loss':total/len(self.ids),'seconds':time.monotonic()-begin}
   if val is not None and (epoch==1 or epoch%5==0 or epoch==epochs):
    score=macro_mse(self.data.y[val],self.predict(self.data,val),self.data.groups[val]);item['inner_macro_MSE_mm2']=score
    if score<best:best=score;best_epoch=epoch;torch.save(self.net.state_dict(),folder/'best_weights.pt')
   history.append(item)
   if epoch==1 or epoch%20==0 or epoch==epochs:write(folder/'progress.json',dict(item,selected_epoch=best_epoch))
  pd.DataFrame(history).to_csv(folder/'history.csv',index=False)
  return {'score':best if val is not None else None,'epoch':best_epoch if val is not None else epochs,'seconds':time.monotonic()-begin}
 def save(self,folder):self.pre.save(folder/'preprocessing.npz');torch.save(self.net.state_dict(),folder/'weights.pt')
