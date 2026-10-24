"""Seal the completed bounded pilot, without simulation or training."""
from pathlib import Path
import hashlib,json,datetime,re,shutil,subprocess
P=Path('/home/robot/research/projects/wm_symbolic')
E=P/'experiments/official_state_resimulation_20261018'
R=P/'reports/official_state_resimulation_20261018'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def main():
 manifest=E/'configs/delivery_manifest.json';assert not manifest.exists()
 shutil.copy2('/tmp/official_state_resimulation_20261018.log',E/'logs/run.log')
 shutil.copy2(__file__,E/'source/seal.py')
 log=(E/'logs/run.log').read_text();assert 'RESIMULATION_COMPLETE'in log
 v=json.loads((E/'results/verification.json').read_text());assert v['passed']
 b=json.loads((E/'results/run_budget.json').read_text());assert b['completed_initializations']==128 and b['known_completed_physics_ticks']==1024 and b['retry_count']==0 and b['stop']is None
 frozen=json.loads((E/'configs/frozen.json').read_text())
 for p,h in frozen['sha256'].items():assert sha(p)==h,p
 proc=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],check=True,capture_output=True,text=True)
 gpu=subprocess.run(['nvidia-smi','--query-gpu=name,memory.used,memory.total','--format=csv,noheader'],check=True,capture_output=True,text=True)
 dump(E/'results/completion_audit.json',{
  'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
  'bounded_scope_completed':True,'formal_training_or_adoption':False,
  'initializations':128,'source_episodes':4,'physics_ticks':1024,'retries':0,
  'frozen_source_hashes_still_match':True,'independent_verification_passed':True,
  'prior_1271_file_pilot_seal_verified_by_independent_verifier':True,
  'figures_visually_reviewed':['episode_fact_coverage.png','original_new_examples.png','pose_and_image_drift.png'],
  'review_findings':'Chinese font, axes, legends, counts, original/new image identities and explanatory captions visible; all three figures inspected with view_image.',
  'gpu_compute_processes_at_completion':proc.stdout.strip(),'gpu_status_at_completion':gpu.stdout.strip(),
  'limitations':['Only four source episodes','Pink Holding positive only two adjacent frames from one episode','Reliability window spans only16.7ms; total33.3ms','Old official images have no new labels','State candidates are not fixed-executor skill transitions','Original full project acceptance remains incomplete'],
  'no_unresolved_technical_failure_in_this_bounded_run':True})
 # The manifest is the only expected link target created after this check.
 links=[]
 for doc in R.glob('*.md'):
  for raw in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',doc.read_text()):
   if raw.startswith(('http:','https:','#')):continue
   target=(doc.parent/raw).resolve();assert target.exists()or target==manifest,(doc,raw);links.append(str(target))
 files=sorted(p for root in[E,R]for p in root.rglob('*')if p.is_file()and p!=manifest)
 payload={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':len(files),'bytes':sum(p.stat().st_size for p in files),'report_markdown_links_checked':len(links),'sha256':{str(p):sha(p)for p in files}}
 dump(manifest,payload)
 for p,h in payload['sha256'].items():assert sha(p)==h,p
 print(json.dumps({k:v for k,v in payload.items()if k!='sha256'},ensure_ascii=False));print('SEALED',manifest)
if __name__=='__main__':main()
