from common import *
import shutil,re,subprocess

def main():
 dest=E/'configs/delivery_manifest.json';assert not dest.exists();assert read(E/'results/verification.json')['passed']
 attempts=read(E/'results/training_attempts.json');assert len(attempts)==4 and all(r['status']=='completed'for r in attempts)
 for name in ['source_references','training_frozen','final_heads_frozen']:
  for p,h in read(E/f'configs/{name}.json')['sha256'].items():assert sha(p)==h,p
 for p,h in read(BASE/'configs/delivery_manifest.json')['sha256'].items():assert sha(p)==h,p
 for stage in ['prepare','diagnose','freeze','train','evaluate','verify','report']:
  p=Path(f'/tmp/edge_control_{stage}_20261018.log');assert p.exists();shutil.copy2(p,E/'logs'/p.name)
 shutil.copy2(__file__,E/'source/seal.py');links=[]
 for md in R.glob('*.md'):
  for target in re.findall(r'\]\(([^)]+)\)',md.read_text()):
   if target.startswith(('https:','http:','#')):continue
   p=(md.parent/target).resolve();assert p.exists()or p==dest,(md,target);links.append(str(p))
 supcon=EX/'readout_supcon_ablation_20261018';assert not(supcon/'results/training_attempts.json').exists();assert not list(supcon.rglob('*update0600.pt'))
 gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],capture_output=True,text=True,check=True)
 save(E/'results/completion_audit.json',{'utc':now(),'new_completed_fits':4,'updates_each':600,'reference_retraining':False,'figures_inspected':['matched_control_changes.png','C_false_positive_decomposition.png','ABCD_fixed_ROI_review.png'],'original_experiment_manifest_unchanged':True,'C_In_severe_regression_preserved':True,'formal_model_replacement':False,'SupCon_training':False,'full_world_model_claim':False,'context_ROIs_not_generic_instance_graph':True,'new_data_or_C_tuning':False,'GPU_processes':gpu.stdout.strip().splitlines(),'report_links_checked':len(links)})
 files=sorted(p for root in[E,R]for p in root.rglob('*')if p.is_file()and p!=dest)
 mf={'utc':now(),'files':len(files),'bytes':sum(p.stat().st_size for p in files),'sha256':{str(p):sha(p)for p in files}};save(dest,mf)
 for p,h in read(dest)['sha256'].items():assert sha(p)==h,p
 print('MATCHED_ZERO_CONTROL_SEALED',len(files),mf['bytes'],sha(dest),flush=True)
if __name__=='__main__':main()
