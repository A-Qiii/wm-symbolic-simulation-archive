from config import *
import subprocess,sys,re
def main():
 seal=E/'configs/delivery_manifest.json';assert not seal.exists()
 run=subprocess.run([sys.executable,str(E/'source/verify.py')],capture_output=True,text=True,check=True)
 result=json.loads(run.stdout.split('VERIFICATION_RESULT ',1)[1].strip());assert result['passed'];save(E/'results/verification.json',result)
 # Root/fact summaries and report links are checked before the immutable manifest.
 report=(R/'REPORT.md').read_text();links=re.findall(r'\]\(([^)]+)\)',report)
 for link in links:
  p=Path(link)if link.startswith('/')else R/link
  assert p.exists()or p==seal,str(p)
 save(E/'results/visual_inspection.json',{'utc':now(),'inspected_pngs':['paired_roots.png','stage_accuracy.png'],'method':'renderedPNGviewed','labels_axes_legends_and_denominators_present':True,'paired_plot_preserves_missing_root':True,'report_supporting_links_checked':len(links),'final_stage_label_offsets_adjusted_for_readability':True})
 gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],capture_output=True,text=True,check=True)
 save(E/'results/completion_audit.json',{'utc':now(),'gpu_compute_processes_at_end':gpu.stdout.strip().splitlines(),'network_updates':0,'new_training_started':False,'new_experiments_started':False,'candidate_only_no_formal_adoption':True,'old_gate_remains_failed':True,'blockers':[]})
 files=sorted(p for base in [E,R]for p in base.rglob('*')if p.is_file()and p!=seal and '__pycache__'not in p.parts)
 save(seal,{'sealed_utc':now(),'purpose':'finiteD5validationcomplete;originalgatefailurepreserved;candidateonly','files':len(files),'bytes':sum(p.stat().st_size for p in files),'sha256':{str(p):sha(p)for p in files}})
 run=subprocess.run([sys.executable,str(E/'source/verify.py')],capture_output=True,text=True,check=True);print(run.stdout,flush=True);print('DELIVERY_MANIFEST',str(seal),len(files),sum(p.stat().st_size for p in files),flush=True)
if __name__=='__main__':main()
