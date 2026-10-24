from common import *
import shutil,re,subprocess
def main():
 dest=E/'configs/delivery_manifest.json';assert not dest.exists();assert read(E/'results/verification.json')['passed']
 assert len(read(E/'results/training_attempts.json'))==8
 for name in['source_references','pre_encoding_frozen','training_frozen','final_heads_frozen']:
  for p,h in read(E/f'configs/{name}.json')['sha256'].items():assert sha(p)==h,p
 for stage in['prepare','encode','tests','freeze','train','evaluate','verify','ambiguity','report']:
  p=Path(f'/tmp/node_edge_{stage}_20261018.log');assert p.exists();shutil.copy2(p,E/'logs'/p.name)
 shutil.copy2(__file__,E/'source/seal.py');links=[]
 for md in R.glob('*.md'):
  for target in re.findall(r'\]\(([^)]+)\)',md.read_text()):
   if target.startswith(('https:','http:','#')):continue
   p=(md.parent/target).resolve();assert p.exists()or p==dest,(md,target);links.append(str(p))
 gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],capture_output=True,text=True,check=True)
 save(E/'results/completion_audit.json',{'utc':now(),'completed_fits':8,'updates_each':600,'visual_figures_inspected':['visual_inputs_and_architecture.png','performance_changes.png','error_examples.png'],'C_In_severe_regression_preserved':True,'formal_model_replacement':False,'SupCon_training':False,'full_world_model_claim':False,'context_ROIs_not_generic_instance_graph':True,'GPU_processes':gpu.stdout.strip().splitlines(),'report_links_checked':len(links)})
 files=sorted(p for root in[E,R]for p in root.rglob('*')if p.is_file()and p!=dest)
 mf={'utc':now(),'files':len(files),'bytes':sum(p.stat().st_size for p in files),'report_links_checked':len(links),'sha256':{str(p):sha(p)for p in files}};save(dest,mf)
 for p,h in read(dest)['sha256'].items():assert sha(p)==h,p
 print('VISUAL_NODE_EDGE_SEALED',len(files),mf['bytes'],sha(dest),flush=True)
if __name__=='__main__':main()
