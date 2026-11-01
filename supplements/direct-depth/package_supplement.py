"""Only package after actual figure and scientific review have been recorded."""
import zipfile
from common import *

def main():
 freeze_check();assert json.loads((ROOT/'checks/independent_supplement_verification.json').read_text())['passed'];assert json.loads((ROOT/'checks/visual_scientific_review.json').read_text())['passed']
 output=ROOT/'delivery';output.mkdir(exist_ok=True);target=output/'eki_direct_supplement_reproduction.zip';assert not target.exists(),'Inspect existing delivery before any replacement'
 records=[]
 for path in sorted(ROOT.rglob('*')):
  relative=path.relative_to(ROOT)
  if not path.is_file() or any(x in relative.parts for x in ['.git','__pycache__','delivery']) or relative.name=='reproduction_manifest.json':continue
  records.append({'path':relative.as_posix(),'bytes':path.stat().st_size,'sha256':digest(path)})
 write(ROOT/'reproduction_manifest.json',{'time':stamp(),'files':records,'base_release':'https://github.com/Zerojy1/Turning-generic-neural-learners-into-pavement-experts/releases/tag/v1.0.0','contents':'100 new direct model fits plus matched original predictions and frozen numerical inputs; full original EKI weights remain in base release'})
 with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for r in records:z.write(ROOT/r['path'],r['path'])
  z.write(ROOT/'reproduction_manifest.json','reproduction_manifest.json')
 with zipfile.ZipFile(target) as z:assert z.testzip() is None
 write(output/'package_receipt.json',{'time':stamp(),'file':target.name,'sha256':digest(target),'bytes':target.stat().st_size,'files':len(records)+1,'Windows_verification':'pending'})
 review=output/'eki_direct_supplement_results.zip'
 with zipfile.ZipFile(review,'w',zipfile.ZIP_DEFLATED) as z:
  for r in records:
   if r['path'].startswith(('figures/','results/tables/','checks/')) and 'smoke' not in r['path'] and 'preflight_v1' not in r['path'] or r['path'] in ['REPORT_ZH.md','RESULT_INDEX.md','DESIGN.md','training_freeze.json','analysis_freeze.json','protocol.json']:z.write(ROOT/r['path'],r['path'])
 print(json.dumps(json.loads((output/'package_receipt.json').read_text())))
if __name__=='__main__':main()
