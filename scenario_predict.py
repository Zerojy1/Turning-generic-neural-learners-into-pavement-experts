from common import *
from calendar_temperature import CalendarTemperature

def predict_scenarios(sc,predict):
 from scenarios import stack_raw
 cal=CalendarTemperature();manifest=json.loads((ROOT/'checks/calendar_manifest.json').read_text());lookup={(s,a['origin_order']):a for s,items in manifest.items() for a in items};out=[]
 for scenario in ['realized','calendar_past','recent_temperature','calendar_loyo_retrospective']:
  origins=[r for r in sc.origins if not scenario.startswith('calendar') or lookup[(r['STR_name'],r['origin_order'])][scenario]['eligible']]
  if not origins:continue
  memory=np.stack([r['memory'] for r in origins]);initial=memory[:,0].copy();hist=np.stack([r['history'] for r in origins])
  for horizon in range(1,5):
   ids=np.array([r['target_ids'][horizon-1] for r in origins]);raw=sc.d.raw(ids);actual=raw['daily_temperature'].copy();raw['memory']=memory.copy();raw['history']=hist.copy();mask=raw['daily_mask'];days=mask.sum(1)
   if scenario=='recent_temperature':raw['daily_temperature']=mask*np.array([r['temperature'] for r in origins])[:,None]
   elif scenario.startswith('calendar'):
    raw['daily_temperature'][:]=0
    for i,r in enumerate(origins):
     info=lookup[(r['STR_name'],r['origin_order'])];lo=int(sc.d.meta.start_day.iloc[ids[i]]);hi=int(sc.d.meta.end_day.iloc[ids[i]])
     estimates=[cal.estimate(r['STR_name'],day,info['origin_day'],scenario) for day in range(lo,hi)];assert all(a is not None for a in estimates)
     raw['daily_temperature'][i,:len(estimates)]=[a['temperature_C'] for a in estimates]
   # Every scenario shares the original given load grid. Never extrapolate loads.
   temp=(raw['daily_temperature']*mask).sum(1)/days;realtemp=(actual*mask).sum(1)/days
   heat=(np.maximum(raw['daily_temperature']-25,0)*mask).sum(1)/days;realheat=(np.maximum(actual-25,0)*mask).sum(1)/days
   pred=predict(raw)
   for i,r in enumerate(origins):
    info=lookup[(r['STR_name'],r['origin_order'])];out.append({'scenario':scenario,'horizon':horizon,'STR_name':r['STR_name'],'origin_order':r['origin_order'],'sample_id':int(ids[i]),'target_order':int(sc.d.meta.Observation_Order.iloc[ids[i]]),'y_true_mm':float(sc.d.y[ids[i]]),'persistence_mm':float(initial[i]),'y_pred_mm':float(pred[i]),'input_latest_mm':float(memory[i,0]),'input_peak_mm':float(memory[i,1]),'input_deltaN_10k':float(raw['load'][i,1]),'input_temp_mean_C':float(temp[i]),'input_days':float(raw['duration'][i,0]),'calendar_common':info['calendar_past']['eligible'],'calendar_two_year':info['calendar_past']['eligible'] and info['calendar_past']['minimum_reference_years']>=2,'loyo_common':info['calendar_loyo_retrospective']['eligible'],'temp_mean_error_C':float(temp[i]-realtemp[i]),'heat_proxy_error_C':float(heat[i]-realheat[i])})
   memory[:,0]=pred;memory[:,1]=np.maximum(memory[:,1],pred)
   nxt=np.c_[raw['load'][:,0]+raw['load'][:,1],raw['load'][:,1],temp,temp-hist[:,-1,2],(hist[:,-2:,2].sum(1)+temp)/3].astype(np.float32);hist=np.concatenate([hist[:,1:],nxt[:,None]],1)
 for start in range(0,len(sc.missing),256):
  batch=sc.missing[start:start+256];r=stack_raw([a['raw'] for a in batch]);pred=predict(r)
  for i,a in enumerate(batch):
   k=a['sample_id'];out.append({'scenario':'missing_one','horizon':2,'STR_name':a['STR_name'],'origin_order':a['origin_order'],'sample_id':k,'target_order':int(sc.d.meta.Observation_Order.iloc[k]),'y_true_mm':float(sc.d.y[k]),'persistence_mm':float(r['memory'][i,0]),'y_pred_mm':float(pred[i]),'input_latest_mm':float(r['memory'][i,0]),'input_peak_mm':float(r['memory'][i,1]),'input_deltaN_10k':float(r['load'][i,1]),'input_temp_mean_C':float(r['temperature'][i,0]),'input_days':float(r['duration'][i,0]),'calendar_common':False,'calendar_two_year':False,'loyo_common':False})
 return pd.DataFrame(out)
