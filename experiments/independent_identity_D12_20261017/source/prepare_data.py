from config import *
import collections
def main():
 verify_frozen();episodes=read(E/'results/collection.json');assert len(episodes)==36 and all(ep['error']is None for ep in episodes)
 states=[];pairs=[];native={};legacy={};outcomes={};hashes={};slots=[]
 for ep in episodes:
  folder=Path(ep['folder']);points=read(folder/'_private/states.json');frame={}
  for p in points:
   i=len(states);frame[p['frame']]=i;states.append({'id':f'n{i:05d}',**{k:ep[k]for k in ['root_id','environment','split','target','other_target','branch']},'frame':p['frame'],'path':p['public_path'],'tags':p['tags'],'phase':p['phase']});native[str(i)]=p['labels'];legacy[str(i)]=p['labels_legacy']
  for call in ep['slots']:
   slots.append({k:ep[k]for k in ['root_id','environment','target','other_target','branch']}|{k:call[k]for k in ['slot','skill','status']})
   if 'after_frame'not in call:continue
   pid=f't{len(pairs):05d}';pairs.append({'id':pid,**{k:ep[k]for k in ['root_id','environment','split','target','other_target','branch']},'slot':call['slot'],'before':frame[call['before_frame']],'after':frame[call['after_frame']],'skill':call['skill'],'action':call['action']});outcomes[pid]=call
  for p in folder.rglob('*'):
   if p.is_file():hashes[str(p)]=sha(p)
 d=E/'data/combo';save(d/'public/states.json',states);save(d/'public/pairs.json',pairs);save(d/'_private/labels.json',native);save(d/'_private/legacy_labels.json',legacy);save(d/'_private/outcomes.json',outcomes);save(d/'_private/slots.json',slots)
 summary={'roots':12,'attempted_branches':len(episodes),'observed_transitions':len(pairs),'planned_slots':144,'states':len(states),'full9_valid_states':sum(all(x['v'])for x in native.values()),'slot_status_counts':dict(collections.Counter(r['status']for r in slots)),'native_legacy_y_or_v_changed_states':sum(native[k]['y']!=legacy[k]['y']or native[k]['v']!=legacy[k]['v']for k in native),'raw_bytes':sum(Path(p).stat().st_size for p in hashes),'elapsed_collection_seconds':sum(ep['elapsed_seconds']for ep in episodes)}
 save(E/'results/coverage.json',summary);save(E/'configs/raw_hashes.json',{'sha256':hashes});ix={r[k]for r in pairs[:50]for k in ['before','after']};save(E/'results/first50_storage.json',{'observed_transitions':len(pairs[:50]),'unique_boundary_files':len(ix),'boundary_bytes':sum(Path(states[i]['path']).stat().st_size for i in ix),'whole_collection_bytes':summary['raw_bytes'],'note':'Whole collection includes continuous video/physical trace; not attributed to first50.'});print('DATA_PREPARED',json.dumps(summary),flush=True)
if __name__=='__main__':main()
