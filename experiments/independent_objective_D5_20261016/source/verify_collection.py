from config import *
import sys,gzip,time
import imageio.v2 as imageio
sys.path.insert(0,str(REP/'source'))
from labels_v2_1 import evaluate
def main():
 hashes=0
 for fn in ['before_collection_frozen.json','raw_D5_hashes.json']:
  for p,h in read(E/'configs'/fn)['sha256'].items():assert sha(p)==h,p;hashes+=1
 old=read(ABL/'configs/delivery_manifest.json')['sha256']
 for p,h in old.items():assert sha(p)==h,p
 for p,h in read(E/'configs/root_exclusion.json')['prior_roots_file_sha256'].items():assert sha(p)==h,p
 roots=read(E/'configs/roots.json');assert len(roots)==len({r['root_id']for r in roots})==len({r['seed']for r in roots})==30;assert all(sum(r['target']==c for r in roots)==10 for c in COLORS);assert all(list(r['branches'])==['normal']for r in roots)
 eps=read(E/'results/collection_D5.json');stamp=datetime.datetime.fromisoformat(read(E/'configs/before_collection_frozen.json')['frozen_utc']).timestamp();frames=pointsN=callsN=startedN=0;media=[];content={};nativefail=[]
 for ep in eps:
  folder=Path(ep['folder']);assert(folder/'_private/provenance.json').stat().st_mtime>=stamp;assert ep['error']is None;pts=read(folder/'_private/states.json');bf={p['frame']:p for p in pts};calls=read(folder/'_private/calls.json');inv=read(folder/'_private/invocations.json');assert len(inv)==ep['started_calls']and len(calls)==ep['actual_calls'];startedN+=len(inv)
  for p in pts:
   assert evaluate(p['physical'])==p['labels'];pointsN+=1
   with np.load(p['public_path'])as z:
    assert set(z.files)=={'rgb_static','rgb_gripper','robot_obs'};assert z['robot_obs'].shape==(15,);h=hashlib.sha256(b''.join(z[k].tobytes()for k in sorted(z.files))).hexdigest();content.setdefault(h,set()).add(ep['root_id'])
  for j,c in enumerate(calls):
   assert c['slot']==j and c['action']==action(c['skill']);assert c['primitive_end']-c['start_frame']==c['native']['controls']<=900;assert c['end_frame']-c['primitive_end']==96;assert c['native']['attempts']<=2
   if j:assert calls[j-1]['after_frame']==c['before_frame']and calls[j-1]['status']=='success'
   if c['status']!='success':nativefail.append({'root_id':ep['root_id'],'slot':j,'skill':c['skill'],'status':c['status'],'native':c['native']})
   callsN+=1
  count=0
  with gzip.open(folder/'_private/physical_control_trace.jsonl.gz','rt')as f:
   for line in f:
    r=json.loads(line);assert r['frame']==count and r['ticks']==count*8
    if count in bf:
     assert r['physical']==bf[count]['physical']
     with np.load(bf[count]['public_path'])as z:assert np.array_equal(z['robot_obs'],r['robot_obs'])
    count+=1
  assert count==ep['frames'];frames+=count;reader=imageio.get_reader(folder/'continuous.mp4');meta=reader.get_meta_data();nf=reader.count_frames();reader.close();assert nf==count and meta['fps']==30;media.append({'root_id':ep['root_id'],'path':str(folder/'continuous.mp4'),'frames':nf,'fps':30})
 assert startedN<=120 and all(len(v)==1 for v in content.values());assert sha(ABL/'results/gate.json')==read(E/'configs/before_collection_frozen.json')['sha256'][str(ABL/'results/gate.json')];assert not read(ABL/'results/gate.json')['passed']
 result={'passed':True,'checked_utc':now(),'frozen_hash_entries':hashes,'previous_ablation_files_verified':len(old),'root_count':len(eps),'declared_roots':30,'started_skill_calls':startedN,'observed_calls':callsN,'point_labels_recomputed':pointsN,'trace_frames_checked':frames,'videos_checked':len(media),'exact_observation_overlap_across_new_roots':0,'external_retries':0,'replacement_roots':0,'original_ablation_gate_still_failed':True,'physical_failures':nativefail,'models_frozen_before_collection':True}
 if not(E/'configs/delivery_manifest.json').exists():save(E/'results/collection_verification.json',result);save(E/'results/media_verification.json',media)
 print('COLLECTION_VERIFIED',json.dumps({k:v for k,v in result.items()if k!='physical_failures'}),flush=True)
if __name__=='__main__':main()
