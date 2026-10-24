from common import *
import collector_open,fcntl
def main():
 lock=(P/'runs/gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);B=E/'data/D_raw';m=read(B/'manifest.json')
 for root in read(B/'roots_manifest.json'):
  if root['kind']!='chain'or root['drawer_reset_m']<.18:continue
  case='chain_open_continuation';eid=root['root_id']+'__'+case
  if any(x['episode_id']==eid for x in m['episodes']):continue
  r=collector_open.episode(root,case);r['continuation_is_new_reset_branch_not_contiguous_with_refusal']=True;m['episodes'].append(r);save(B/'manifest.json',m);print({k:v for k,v in r.items()if k!='final_labels'},flush=True)
  if r['error']:raise RuntimeError('Preserved infrastructure failure')
 print('OPEN_BRANCHES_COMPLETE',flush=True)
if __name__=='__main__':main()
