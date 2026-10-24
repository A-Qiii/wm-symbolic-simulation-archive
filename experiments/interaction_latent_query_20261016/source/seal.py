from common import *
from verify import sha as streamsha
def main():
 assert read(E/'results/delivery_audit.json')['passed'];assert read(E/'results/runtime_bridge_preflight.json')['passed'];assert len(read(E/'results/closed_loop.json'))==18
 m={};total=0
 for base in [E,R]:
  for p in sorted(base.rglob('*')):
   if p.is_file()and'__pycache__'not in p.parts and p!=E/'configs/delivery_manifest.json'and p!=E/'logs/seal.log':m[str(p)]=streamsha(p);total+=p.stat().st_size
 # Verify linked original inputs and generic feature caches without changing them.
 linked={}
 for r in read(E/'data/public/states.json'):
  for p,h in[(r['source'],r['source_sha256'])]+([(r['feature_source'],r['feature_source_sha256'])]if'feature_source'in r else[]):assert streamsha(p)==h,p;linked[p]=h
 for p,h in read(E/'configs/final_before_D.json')['sha256'].items():assert streamsha(p)==h,p;linked[p]=h
 save(E/'configs/delivery_manifest.json',{'created_utc':now(),'experiment_and_report_files':len(m),'linked_original_files':len(linked),'bytes':total,'sha256':m|linked,'D_tuning':False,'active_training_or_collection':False});print('SEALED',len(m),'files',len(linked),'linked',total,'bytes',flush=True)
if __name__=='__main__':main()
