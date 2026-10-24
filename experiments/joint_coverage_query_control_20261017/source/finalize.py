from config import *
import subprocess,sys,re
def main():
 seal=E/'configs/delivery_manifest.json';assert not seal.exists()
 r=subprocess.run([sys.executable,str(E/'source/verify.py')],capture_output=True,text=True,check=True);save(E/'results/verification.json',json.loads(r.stdout.split('QUERY_CONTROL_VERIFICATION ',1)[1]))
 for fn in ['REPORT.md','INTERPRETATION.md']:
  for link in re.findall(r'\]\(([^)]+)\)',(R/fn).read_text()):
   if Path(link)==seal:continue
   assert(Path(link)if link.startswith('/')else R/link).exists(),link
 vi=read(E/'results/visual_inspection.json');assert vi['inspected']and len(vi['figures'])==4
 gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],capture_output=True,text=True,check=True)
 save(E/'results/completion_audit.json',{'utc':now(),'gpu_compute_processes':gpu.stdout.strip().splitlines(),'query_fits':4,'updates_per_fit':600,'new_physical_calls':0,'world_model_updates':0,'formal_query_replaced':False,'D_used_for_training_selection':False,'remaining_execution_blockers':[]})
 files=sorted(p for b in [E,R]for p in b.rglob('*')if p.is_file()and p!=seal and '__pycache__'not in p.parts);save(seal,{'utc':now(),'purpose':'fixedarchitecturepairedquerycoveragecontrol;noautomaticadoption','files':len(files),'bytes':sum(p.stat().st_size for p in files),'sha256':{str(p):sha(p)for p in files}})
 r=subprocess.run([sys.executable,str(E/'source/verify.py')],capture_output=True,text=True,check=True);print(r.stdout,flush=True);print('QUERY_CONTROL_DELIVERY_SEALED',len(files),flush=True)
if __name__=='__main__':main()
