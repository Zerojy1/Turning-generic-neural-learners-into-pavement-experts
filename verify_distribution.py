"""Verify the current-study distribution without loading model binaries."""
from pathlib import Path
import argparse,hashlib,json

def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['git','full'],default='full');a=p.parse_args()
 root=Path(__file__).resolve().parent;manifest=json.loads((root/'FILE_MANIFEST.json').read_text(encoding='utf-8'));count=0
 for r in manifest['files']:
  if a.mode=='git' and r['storage']=='release':continue
  f=root/r['path'];assert f.is_file(),r['path'];assert f.stat().st_size==r['bytes'],r['path'];h=hashlib.sha256()
  with f.open('rb') as stream:
   for b in iter(lambda:stream.read(4194304),b''):h.update(b)
  assert h.hexdigest()==r['sha256'],r['path'];count+=1
 print(json.dumps({'passed':True,'mode':a.mode,'verified_files':count}))
if __name__=='__main__':main()
