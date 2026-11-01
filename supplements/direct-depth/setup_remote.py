import csv,json,pathlib,shutil,hashlib
root=pathlib.Path(__file__).resolve().parent;base=pathlib.Path('/home/zyl105/Zerojy/projects/rutting-eki-v8-20261101');old=base.parent/'rutting-engineering-modules-v7-20261101'
assert not (root/'runs').exists() and not (root/'training_freeze.json').exists()
index={r['path']:r for r in csv.DictReader((root/'base_FILE_INDEX.csv').open())}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
names=['data/arrays.npz','data/panel.csv','data/daily.csv','data/targets.csv','data/archive_daily.csv','data/daily_target_audit.csv','data/source_manifest.json','checks/calendar_manifest.json','checks/origin_manifest.json','checks/calendar_summary.json']
references=['results/ensemble_predictions.csv.gz','results/scenario_predictions.csv.gz','results/tables/one_step_summary.csv','results/tables/scenario_summary.csv','results/tables/primary_conjunctions.csv','checks/independent_verification.json']
records=[]
for n in names+references:
 source=(root/'public_overrides'/n) if (root/'public_overrides'/n).is_file() else base/n;assert source.is_file(),n
 h=sha(source);assert n in index and h==index[n]['sha256'],('not identical to GitHub release',n)
 target=root/(('base_reference/'+n) if n in references else n);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
 records.append({'path':n,'sha256':h,'bytes':source.stat().st_size,'source':'verified against GitHub v1.0.0 manifest; existing Linux original'})
(root/'checks/source_reuse.json').write_text(json.dumps({'passed':True,'files':records},indent=2))
audit={'old_project':str(old),'old_complete_jobs':len(list((old/'runs').glob('*/complete.json'))),'old_model_sha256':sha(old/'models.py'),'new_base_model_sha256':sha(base/'models.py'),'raw_arrays_identical':sha(old/'data/arrays.npz')==sha(base/'data/arrays.npz'),'target_ids_identical':sha(old/'data/targets.csv')==sha(base/'data/targets.csv'),'old_generic_head':'94-55-1','current_generic_head':'87-57-1','old_model_reuse':False,'reason':'different current temperature/load information, head dimensions and preprocessing; no score-based decision','new_direct_jobs':100}
assert audit['old_complete_jobs']==830
(root/'checks/reuse_audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit))
