from common import *
import collector,fcntl
def main():
 lock=(P/'runs/gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);B=E/'data/D_raw'
 for p,h in read(E/'configs/final_before_D.json')['sha256'].items():assert sha(p)==h,p
 m=read(B/'manifest.json')
 for root in read(B/'roots_manifest.json'):
  for case in collector.CASES if root['kind']=='fixtures'else['chain']:
   eid=root['root_id']+'__'+case
   if any(r['episode_id']==eid for r in m['episodes']):continue
   r=collector.episode(root,case);m['episodes'].append(r);save(B/'manifest.json',m);print({k:v for k,v in r.items()if k!='final_labels'},flush=True)
   if r['error']:raise RuntimeError('Preserved collection failure, no replacement')
 print('D_REFERENCE_COMPLETE',len(m['episodes']),flush=True)
if __name__=='__main__':main()
