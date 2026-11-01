from pathlib import Path
import csv,hashlib,json,random,time
import numpy as np,pandas as pd,torch
ROOT=Path(__file__).resolve().parent
P=json.loads((ROOT/'protocol.json').read_text(encoding='utf-8'))
ENV=['Cum_Load_10k','Delta_Load_10k','Avg_Temp_C','Delta_Temp_C','Temp_MA3_C']
SLOTS=json.loads((ROOT/'slot_mapping.json').read_text(encoding='utf-8'))['slots']
STRUCT=[s['column'] for s in SLOTS]
def stamp():
 from datetime import datetime,timezone,timedelta
 return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')
def digest(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(path,obj):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);q=path.with_suffix(path.suffix+'.tmp');q.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');q.replace(path)
def seed_all(seed):
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
def specs():
 a=list(csv.DictReader((ROOT/'run_matrix.csv').open(encoding='utf-8-sig')))
 for s in a:s['seed']=int(s['seed'])
 return a
def spec(tag):return next(s for s in specs() if s['job_id']==tag)
def split(d,s):
 if s['split'].startswith('fold'):
  f=int(s['split'][4:]);held=P['splits']['outer_groups'][f-1];val=P['splits']['outer_groups'][f%5];final=[g for g in np.unique(d.groups) if g not in held];train=[g for g in final if g not in val]
 else:
  a=P['class_splits'][s['split'][6:]];train,val,final,held=[a[k] for k in ['inner_train','inner_validation','final_train','heldout']]
 return tuple(d.ids[np.isin(d.groups,n)] for n in [train,val,final,held])
def macro_mse(y,p,g):return float(np.mean([np.mean((np.asarray(p)[g==s]-np.asarray(y)[g==s])**2) for s in np.unique(g)]))
def offset(raw,s):return raw['memory'][:,0] if s['output']=='residual' else np.zeros(len(raw['memory']),np.float32)
def freeze_check():
 f=ROOT/'training_freeze.json'
 if f.exists():
  for n,h in json.loads(f.read_text())['files'].items():assert digest(ROOT/n)==h,('changed frozen file',n)
def provenance():
 names=['common.py','data.py','models.py','run_job.py','protocol.json','run_matrix.csv','slot_mapping.json','target_manifest.csv','data/arrays.npz','data/panel.csv','data/daily.csv','data/targets.csv','execution_addendum.md','calendar_temperature.py','data/archive_daily.csv','checks/calendar_manifest.json']
 return {n:digest(ROOT/n) for n in names}


def matches_recorded_provenance(recorded):
 """Accept exact original records only for the hash-verified distribution snapshot."""
 current=provenance()
 if recorded==current:return True
 path=ROOT/'release_provenance.json'
 if not path.exists():return False
 ledger=json.loads(path.read_text(encoding='utf-8'))
 if recorded!=ledger['original_training_provenance']:return False
 for name,expected in ledger['packaged_runtime_files'].items():
  if digest(ROOT/name)!=expected:return False
 return True
