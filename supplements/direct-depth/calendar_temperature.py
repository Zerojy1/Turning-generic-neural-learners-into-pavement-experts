from common import *
from functools import lru_cache
from datetime import date,timedelta

class CalendarTemperature:
 def __init__(self):
  daily=pd.read_csv(ROOT/'data/daily.csv');arc=pd.read_csv(ROOT/'data/archive_daily.csv');self.daymap={};self.archive={};self.years={}
  for r in daily.itertuples():
   self.daymap[(r.STR_name,int(r.day))]=(float(r.temp_C),bool(r.internal));self.years.setdefault(r.STR_name,set()).add(date.fromordinal(date(1970,1,1).toordinal()+int(r.day)).year)
  for r in arc.itertuples():self.archive[(r.STR_name,int(r.day))]=(float(r.archive_temp_C),int(r.available_day),bool(r.supported))
 @lru_cache(maxsize=None)
 def estimate(self,sid,day,origin,mode):
  target=date(1970,1,1)+timedelta(days=int(day));orig=date(1970,1,1)+timedelta(days=int(origin));retrospective=mode=='calendar_loyo_retrospective'
  years=sorted(y for y in self.years[sid] if (y!=target.year if retrospective else y<orig.year))
  for source in ['internal','archive']:
   means=[];refs=[];supports=[];avail=[]
   for year in years:
    centers=[date(year,target.month,target.day)] if not(target.month==2 and target.day==29) else [date(year,2,28),date(year,3,1)]
    vals_by_center=[];year_support=[];year_avail=[]
    for center in centers:
     vals=[];support=[];av=[]
     for shift in range(-3,4):
      dd=(center+timedelta(days=shift)-date(1970,1,1)).days
      support_year=(date(1970,1,1)+timedelta(days=dd)).year
      if (support_year==target.year if retrospective else support_year>=orig.year):continue
      if not retrospective and dd>=origin:continue
      if source=='internal':
       entry=self.daymap.get((sid,dd));ok=entry is not None and entry[1];temp=entry[0] if entry else 0;available=dd+1
      else:
       entry=self.archive.get((sid,dd));ok=entry is not None and entry[2];temp=entry[0] if entry else 0;available=entry[1] if entry else -1
      if ok and (retrospective or available<origin):vals.append(temp);support.append(dd);av.append(available)
     if len(vals)<4:break
     vals_by_center.append(float(np.mean(vals)));year_support+=support;year_avail+=av
    if len(vals_by_center)==len(centers):means.append(float(np.mean(vals_by_center)));refs.append(year);supports+=year_support;avail+=year_avail
   if means:
    return {'temperature_C':float(np.mean(means)),'source':source,'reference_years':refs,'support_days':sorted(set(supports)),'max_available_day':max(avail),'reference_year_mean_temperatures':means,'retrospective':retrospective}
  return None

def freeze_calendar_manifest():
 from data import Data
 from scenarios import Scenarios
 d=Data();cal=CalendarTemperature();rows=[];manifests={};details=[]
 # Inner calibration, outer evaluation and classes use the same source rule.
 for sid in sorted(set(d.groups),key=lambda x:int(x[3:])):
  sc=Scenarios(d,[sid]);manifests[sid]=[]
  for origin in sc.origins:
   first=origin['target_ids'][0];origin_day=int(d.meta.start_day.iloc[first]);item={k:origin[k] for k in ['STR_name','origin_order','target_ids']};item['origin_day']=origin_day
   for mode in ['calendar_past','calendar_loyo_retrospective']:
    estimate_days=[];missing_days=[];minimum_years=999;archive_days=0
    for k in origin['target_ids']:
     for day in range(int(d.meta.start_day.iloc[k]),int(d.meta.end_day.iloc[k])):
      a=cal.estimate(sid,day,origin_day,mode)
      if a is None:missing_days.append(day)
      else:
       estimate_days.append(day);minimum_years=min(minimum_years,len(a['reference_years']));archive_days+=a['source']=='archive'
       details.append({'STR_name':sid,'origin_order':origin['origin_order'],'origin_day':origin_day,'target_day':day,'mode':mode,**a})
    item[mode]={'eligible':not missing_days,'minimum_reference_years':minimum_years if estimate_days else 0,'estimated_days':len(estimate_days),'archive_days':archive_days,'missing_days':missing_days}
   manifests[sid].append(item)
   rows.append({'STR_name':sid,'origin_order':origin['origin_order'],'origin_day':origin_day,'past_eligible':item['calendar_past']['eligible'],'loyo_eligible':item['calendar_loyo_retrospective']['eligible'],'past_min_years':item['calendar_past']['minimum_reference_years'],'past_archive_days':item['calendar_past']['archive_days'],'past_missing_days':len(item['calendar_past']['missing_days'])})
 write(ROOT/'checks/calendar_manifest.json',manifests);pd.DataFrame(rows).to_csv(ROOT/'checks/calendar_availability.csv',index=False)
 with (ROOT/'checks/calendar_day_provenance.jsonl').open('w') as f:
  for row in details:f.write(json.dumps(row)+'\n')
 test=pd.DataFrame(rows);test=test[~test.STR_name.isin(P['splits']['anchors'])]
 write(ROOT/'checks/calendar_summary.json',{'time':stamp(),'primary_rollout_origins':len(test),'strict_past_common_origins':int(test.past_eligible.sum()),'retrospective_common_origins':int(test.loyo_eligible.sum()),'strict_at_least2years_origins':int((test.past_eligible&(test.past_min_years>=2)).sum()),'source_outcomes_used':False,'fallback':'archive support at interval end; strictly earlier than origin','by_structure':test.groupby('STR_name')[['past_eligible','loyo_eligible']].sum().reset_index().to_dict('records')})
 print('CALENDAR_READY',len(test),int(test.past_eligible.sum()),int(test.loyo_eligible.sum()))
if __name__=='__main__':freeze_calendar_manifest()
