from analysis import *
import os,subprocess,re,ast
def main():
 seal=E/'configs/delivery_manifest.json';assert not seal.exists()
 assert read(E/'results/verification.json')['passed']
 for p,h in read(E/'configs/input_manifest.json')['sha256'].items():assert sha(p)==h,p
 original=read(D/'configs/delivery_manifest.json')
 for p,h in original['sha256'].items():assert sha(p)==h,p
 # Make the saved source package importable using its local analysis.py name.
 for name in ['verify.py','report.py']:
  p=E/'source'/name;s=p.read_text().replace('from analysis import *','from analysis import *');ast.parse(s);p.write_text(s)
 source=Path(__file__).read_text().replace('from analysis import *','from analysis import *');(E/'source/finalize.py').write_text(source)
 assets=read(E/'results/report_assets.json');assert len(assets['figures'])==3
 save(E/'results/visual_inspection.json',{'utc':now(),'inspected':True,'method':'All three PNGs viewed; checked 0–11 common color scale, legible counts, group and seed labels, branch/action definitions, failure-root denominator disclosure.','figures':{n:sha(R/n)for n in assets['figures']},'remaining_issues':[]})
 active=[]
 for proc in Path('/proc').iterdir():
  if not proc.name.isdigit()or int(proc.name)==os.getpid():continue
  try:
   args=(proc/'cmdline').read_bytes().split(b'\0');exe=Path(args[0].decode()).name if args[0]else''
   if exe.startswith('python')and any(str(root).encode()in a for a in args for root in [D/'source',E/'source']):active.append(int(proc.name))
  except(FileNotFoundError,PermissionError,ProcessLookupError):pass
 assert not active,active
 gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],text=True,capture_output=True,check=True)
 save(E/'results/completion_audit.json',{'utc':now(),'task':'descriptive appendix from sealed probabilities only','model_forward_calls':0,'training_updates':0,'collection_calls':0,'threshold_changes':0,'new_repairs':0,'original_manifest_sha256':sha(D/'configs/delivery_manifest.json'),'original_sealed_files_verified_unchanged':len(original['sha256']),'active_experiment_processes':active,'GPU_compute_processes':gpu.stdout.strip().splitlines(),'round_stopped':True,'automatic_next_experiment':False,'remaining_execution_blockers':[]})
 for link in re.findall(r'\]\(([^)]+)\)',(R/'APPENDIX.md').read_text()):
  target=(R/link).resolve()
  if target==seal.resolve():continue
  assert target.exists(),link
 files=sorted(p for root in [E,R]for p in root.rglob('*')if p.is_file()and p!=seal and '__pycache__'not in p.parts)
 save(seal,{'utc':now(),'purpose':'separate read-only D12 error localization appendix; original delivery seal preserved','files':len(files),'bytes':sum(p.stat().st_size for p in files),'sha256':{str(p):sha(p)for p in files}})
 for p,h in read(seal)['sha256'].items():assert sha(p)==h,p
 print('LOCALIZATION_SEALED',len(files),'files',len(list(R.glob('*.csv'))),'CSV;3 figures;GPU',gpu.stdout.strip()or'none')
if __name__=='__main__':main()
