from config import *
import contextlib,io,subprocess
import verify
assert not(E/'configs/delivery_manifest.json').exists(),'Already sealed'
capture=io.StringIO()
with contextlib.redirect_stdout(capture):verify.main()
line=next(x for x in capture.getvalue().splitlines()if x.startswith('VERIFICATION_RESULT '));v=json.loads(line.split(' ',1)[1]);v['checked_utc']=now();v['visual_review']={'reviewed':True,'figures':['true_latent_objective_conflict.png','predicted_latent_counterevidence.png'],'axes_colors_denominators_conditions_meaning_explained':True,'readable_no_clipping':True}
gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],text=True,capture_output=True);assert gpu.returncode==0,gpu.stderr;v['gpu_compute_processes_after_completion']=gpu.stdout.strip().splitlines();assert not v['gpu_compute_processes_after_completion']
save(E/'results/verification.json',v)
files=sorted(p for folder in [E,R]for p in folder.rglob('*')if p.is_file()and '__pycache__'not in p.parts and p.name!='delivery_manifest.json')
seal={'sealed_utc':now(),'scope':'Finite frozen-objective diagnostic complete. No training/simulation/weight change; subsequent controlled training is a recommendation only.','files':len(files),'bytes':sum(p.stat().st_size for p in files),'sha256':{str(p):sha(p)for p in files}}
save(E/'configs/delivery_manifest.json',seal);print('SEALED',json.dumps({k:v for k,v in seal.items()if k!='sha256'}),flush=True)
