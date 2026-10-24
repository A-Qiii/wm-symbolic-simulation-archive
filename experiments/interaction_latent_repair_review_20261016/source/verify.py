"""Read-only integrity and artifact checks; no model or environment execution."""
from audit import *
import subprocess
def main():
 old=subprocess.run([sys.executable,str(OLD/'source/verify.py')],capture_output=True,text=True,check=True)
 f=read(E/'configs/review_delivery_manifest.json');total=0
 for name,h in f['sha256'].items():
  assert sha(name)==h,name;total+=Path(name).stat().st_size
 for file in ['proposal_freeze.json','matched_source_freeze.json']:
  for name,h in read(E/'configs'/file)['sha256'].items():assert sha(name)==h,name
 for r in read(E/'results/media_manifest.json'):
  assert sha(r['video'])==r['sha256']
  for name,h in r['source_endpoint_sha256'].items():assert sha(name)==h,name
 assert read(E/'results/probe_audit.json')['training_updates']==0
 assert all(not r['collected']for r in read(E/'configs/proposed_D2_roots.json'))
 print(json.dumps({'passed':True,'review_files':len(f['sha256']),'review_bytes':total,'old_sealed_verifier':old.stdout.strip(),'new_training_or_physical_execution':False,'verification_is_read_only':True},ensure_ascii=False))
if __name__=='__main__':main()
