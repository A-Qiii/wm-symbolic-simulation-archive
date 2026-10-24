"""Finalize verified bounded expansion, preserving earlier sealed experiments."""
from pathlib import Path
import hashlib,json,shutil,datetime,re,subprocess
P=Path('/home/robot/research/projects/wm_symbolic');E=P/'experiments/official_AB_state_expansion_20261018';R=P/'reports/official_AB_state_expansion_20261018'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def main():
 manifest=E/'configs/delivery_manifest.json';assert not manifest.exists()
 v=read(E/'results/verification.json');b=read(E/'results/run_budget.json');assert v['passed']and b['completed_initializations']==256 and b['known_completed_physics_ticks']==24576 and b['retry_count']==0 and b['stop']is None
 for name in ['metadata','download','run','verify']:
  source=Path('/tmp')/f'official_expansion_{name}_20261018.log';assert source.exists();shutil.copy2(source,E/'logs'/f'{name}.log')
 assert 'EXPANSION_RESIMULATION_COMPLETE'in(E/'logs/run.log').read_text()
 assert 'EXPANSION_VERIFIED'in(E/'logs/verify.log').read_text()
 for manifest_path in [E/'configs/frozen_selection.json',E/'configs/frozen_execution.json']:
  for p,h in read(manifest_path)['sha256'].items():assert sha(p)==h,p
 shutil.copy2(__file__,E/'source'/Path(__file__).name)
 proc=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],check=True,capture_output=True,text=True)
 gpu=subprocess.run(['nvidia-smi','--query-gpu=name,memory.used,memory.total','--format=csv,noheader'],check=True,capture_output=True,text=True)
 figures=sorted(p.name for p in(R/'figures').glob('*.png'));assert len(figures)==5
 dump(E/'results/completion_audit.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'bounded_expansion_completed':True,'source_episodes':12,'new_source_frames':240,'old_source_retests':16,'simulation_initializations':256,'physics_ticks':24576,'retry_count':0,'training_updates':0,'model_forward_calls':0,'cumulative_network_body_bytes':read(E/'results/network_ledger.json')['body_bytes_read'],'max_network_body_bytes':512*1024**2,'independent_verification_passed':True,'figures_visually_reviewed':figures,'visual_review':'All five figures inspected using view_image; labels, axes, units, original/new image identities and explanations readable.','gpu_compute_processes_at_completion':proc.stdout.strip(),'gpu_status_at_completion':gpu.stdout.strip(),'previous_seals_unchanged_confirmed_by_verifier':True,'formal_adoption_or_training':False,'remaining_scope':'Parent decides whether controlled reader training is appropriate; no further collection or parameter tuning in this run.'})
 links=[]
 for doc in R.glob('*.md'):
  for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',doc.read_text()):
   if target.startswith(('http:','https:','#')):continue
   path=(doc.parent/target).resolve();assert path.exists()or path==manifest,(doc,target);links.append(str(path))
 files=sorted(p for root in[E,R]for p in root.rglob('*')if p.is_file()and p!=manifest)
 data={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':len(files),'bytes':sum(p.stat().st_size for p in files),'report_links_checked':len(links),'sha256':{str(p):sha(p)for p in files}};dump(manifest,data)
 for p,h in data['sha256'].items():assert sha(p)==h,p
 print(json.dumps({k:v for k,v in data.items()if k!='sha256'},ensure_ascii=False));print('EXPANSION_SEALED',manifest)
if __name__=='__main__':main()
