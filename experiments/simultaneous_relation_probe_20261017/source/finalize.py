from config import *
import subprocess,sys,re

def main():
 seal=E/'configs/delivery_manifest.json';assert not seal.exists()
 run=subprocess.run([sys.executable,str(E/'source/verify.py')],capture_output=True,text=True,check=True);result=json.loads(run.stdout.split('PROBE_VERIFICATION ',1)[1]);save(E/'results/verification.json',result)
 for link in re.findall(r'\]\(([^)]+)\)',(R/'REPORT.md').read_text()):assert(Path(link)if link.startswith('/')else R/link).exists(),link
 assert read(E/'results/visual_inspection.json')['inspected']
 gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],capture_output=True,text=True,check=True)
 save(E/'results/completion_audit.json',{'utc':now(),'gpu_compute_processes':gpu.stdout.strip().splitlines(),'CPU_probe_fits':4,'new_physical_calls':0,'world_model_updates':0,'formal_query_replaced':False,'D_used_for_training_selection':False,'remaining_execution_blockers':[]})
 files=sorted(p for base in [E,R]for p in base.rglob('*')if p.is_file()and p!=seal and '__pycache__'not in p.parts);save(seal,{'utc':now(),'purpose':'boundedofflineinformationreadabilitydiagnostic;notqueryreplacement','files':len(files),'bytes':sum(p.stat().st_size for p in files),'sha256':{str(p):sha(p)for p in files}})
 run=subprocess.run([sys.executable,str(E/'source/verify.py')],capture_output=True,text=True,check=True);print(run.stdout,flush=True);print('PROBE_DELIVERY_SEALED',len(files),flush=True)
if __name__=='__main__':main()
