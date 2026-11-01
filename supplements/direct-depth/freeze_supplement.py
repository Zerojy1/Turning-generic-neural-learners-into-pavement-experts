import subprocess
from common import *

def main():
 assert not (ROOT/'training_freeze.json').exists() and not (ROOT/'runs').exists();assert json.loads((ROOT/'checks/preflight_complete.json').read_text())['passed']
 files=sorted([p.relative_to(ROOT).as_posix() for p in ROOT.glob('*.py')]+['DESIGN.md','protocol.json','run_matrix.csv','slot_mapping.json','target_manifest.csv','execution_addendum.md','github_source_receipt.json','base_release_provenance.json','base_FILE_INDEX.csv']+[p.relative_to(ROOT).as_posix() for p in (ROOT/'data').rglob('*') if p.is_file()]+['checks/calendar_manifest.json','checks/origin_manifest.json','checks/calendar_summary.json','checks/source_reuse.json','checks/reuse_audit.json','checks/unit_checks.json','checks/benchmark.json','checks/preflight_complete.json'])
 (ROOT/'.gitattributes').write_text('* -text\n');(ROOT/'.gitignore').write_text('__pycache__/\n*.pyc\nruns/\nqueues/\nlogs/\nevaluation/\nresults/\nfigures/\nbase_reference/\nchecks/smoke/\n*.zip\n')
 subprocess.run(['git','init'],cwd=ROOT,check=True);subprocess.run(['git','config','user.name','Research Automation'],cwd=ROOT,check=True);subprocess.run(['git','config','user.email','local-research@localhost'],cwd=ROOT,check=True);subprocess.run(['git','add','--',*files,'.gitignore','.gitattributes'],cwd=ROOT,check=True);subprocess.run(['git','commit','-m','Freeze matched direct-depth baseline supplement and all numerical analyses'],cwd=ROOT,check=True);commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
 manifest={'commit':commit,'time':stamp(),'files':{n:digest(ROOT/n) for n in files},'new_jobs':100,'prior_EKI_results_already_known':True,'new_direct_outer_results_seen':False,'base_public_commit':P['supplement']['base_public_commit']};write(ROOT/'training_freeze.json',manifest);write(ROOT/'analysis_freeze.json',manifest);print('FROZEN',commit)
if __name__=='__main__':main()
