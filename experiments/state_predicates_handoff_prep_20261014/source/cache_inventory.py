from pathlib import Path
from collections import Counter
import json,hashlib
P=Path('/home/robot/research/projects/wm_symbolic');E=P/'experiments/state_predicates_handoff_prep_20261014'
def read(p):return json.loads(Path(p).read_text())
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 assert not(E/'data/macro_transition_history_inventory_v2.json').exists();lookup={}
 for p in [E/'data/metadata_long_history_v1.json',P/'experiments/state_predicates_held_down_20261014/data/metadata.json',P/'experiments/state_predicates_countermotion_20261014/data/metadata.json',P/'experiments/state_predicates_interface_acceptance_20261014/data/metadata.json']:
  for r in read(p):lookup[tuple(r['public_frames'])]={'feature_path':r['feature_path'],'sha256':sha(r['feature_path']),'metadata_source':str(p),'sample_id':r.get('unique_window_id',r['sample_id'])}
 rows=read(E/'data/macro_transition_history_inventory.json');summ=[]
 for r in rows:
  for side in ['before','after']:
   q=r[side];match=lookup.get(tuple(q['long_public_paths']))if q['complete_2p1s']else None;q['existing_long_feature']=match;q['cache_status']='missing_public_input'if not q['complete_2p1s']else('public_and_cache_ready'if match else'public_ready_needs_cache');q['new_DINO_cache_required_for_long']=None if not q['complete_2p1s']else not bool(match)
 for part,source in sorted({(r['partition'],r['source'])for r in rows}):
  rr=[r for r in rows if r['partition']==part and r['source']==source];summ.append({'partition':part,'source':source,'macro_transitions':len(rr),'before':dict(Counter(r['before']['cache_status']for r in rr)),'after':dict(Counter(r['after']['cache_status']for r in rr)),'both_long_cached':sum(all(r[s]['cache_status']=='public_and_cache_ready'for s in ['before','after'])for r in rr)})
 save(E/'data/macro_transition_history_inventory_v2.json',rows);save(E/'results/long_cache_inventory.json',{'counts':summ,'match_rule':'Exact ordered public-file path match to saved2.1s state-window metadata; no new DINO inference. Only full matches counted. No predicted output read.','v1_to_v2_change':'Separate raw long-history availability from existing long DINO cache; retain v1 inventory for traceability.'});print(json.dumps(summ,ensure_ascii=False))
if __name__=='__main__':main()
