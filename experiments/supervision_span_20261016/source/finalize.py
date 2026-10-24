from config import *
import io,contextlib,subprocess
import verify

assert not(E/'configs/delivery_manifest.json').exists(),'Already sealed'
buf=io.StringIO()
with contextlib.redirect_stdout(buf):verify.main()
line=next(x for x in buf.getvalue().splitlines()if x.startswith('VERIFICATION_RESULT '))
check=json.loads(line.split(' ',1)[1])
check['checked_utc']=now()
check['visual_review']={'files':['C_step_and_prefix.png','C_selection_curves.png'],'reviewed':True,'legends_axes_conditions_meaning_present':True,'text_and_labels_readable':True}
gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],text=True,capture_output=True)
assert gpu.returncode==0,gpu.stderr
check['gpu_compute_processes_after_completion']=gpu.stdout.strip().splitlines()
assert not check['gpu_compute_processes_after_completion'],'GPU compute still running'
save(E/'results/verification.json',check)
files=sorted(p for folder in [E,R]for p in folder.rglob('*')if p.is_file()and '__pycache__'not in p.parts and p.name!='delivery_manifest.json')
manifest={'sealed_utc':now(),'scope':'Finite supervision-span comparison complete; gate failed; no newD; no additional tuning','files':len(files),'bytes':sum(p.stat().st_size for p in files),'sha256':{str(p):sha(p)for p in files}}
save(E/'configs/delivery_manifest.json',manifest)
print('SEALED',json.dumps({k:v for k,v in manifest.items()if k!='sha256'}),flush=True)
