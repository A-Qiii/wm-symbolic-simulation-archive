"""Offline provenance exporter. This script alone reads old evaluator metadata.
The training reader accepts only the separate public exports, never this module.
"""
from common import *
from collections import Counter,defaultdict
import shutil
def main():
 assert not(E/'data/public/states.json').exists()
 for d in ['source','configs','data/public/observations','data/public/features','data/_private','models','results','logs']:(E/d).mkdir(parents=True,exist_ok=True)
 states=[];pairs=[];labels={};roots={};seen={};sourcehash={};evaluation=[];originals={}
 def source(p):sourcehash[str(p)]=sha(p);return read(p)
 def addroot(r,dataset):
  rid=r['root_id'];env=r['environment'];assert env in'ABC'
  roots[rid]={'root_id':rid,'environment':env,'split':'train'if env in'AB'else'dev','dataset':dataset,'seed':r.get('seed'),'layout':r.get('layout')}
 def state(path,rid,cache=None):
  with np.load(path)as z:raw={k:z[k].copy()for k in ['rgb_static','rgb_gripper','robot_obs']}
  raw['robot_obs']=raw['robot_obs'].astype(np.float32)
  h=hashlib.sha256(b''.join(raw[k].tobytes()for k in sorted(raw))).hexdigest();key=(rid,h)
  if key in seen:return seen[key]
  idx=len(states);sid=f's{idx:05d}';out=E/'data/public/observations'/f'{sid}.npz';np.savez_compressed(out,**raw)
  row={'id':sid,'index':idx,'root_id':rid,'environment':roots[rid]['environment'],'split':roots[rid]['split'],'source':str(path),'source_sha256':sha(path),'content_sha256':h,'path':str(out),'sha256':sha(out)}
  if cache is not None:
   with np.load(cache)as z:f={k:z[k][-1].copy()for k in ['local_tokens','geometry','global_grid','robot_history']}
   assert np.allclose(f['robot_history'],raw['robot_obs']);fp=E/'data/public/features'/f'{sid}.npz';np.savez_compressed(fp,**f);row.update(feature_path=str(fp),feature_source=str(cache),feature_source_sha256=sha(cache))
  states.append(row);seen[key]=idx;originals[str(path)]=idx;return idx
 def pair(rid,eid,step,skill,b,a,target,hold,status,dataset):
  pairs.append({'id':f'p{len(pairs):05d}','root_id':rid,'episode_id':eid,'step':step,'skill':skill,'action':action(skill,target,hold),'before':b,'after':a,'split':roots[rid]['split'],'dataset':dataset})
  evaluation.append({'pair_id':pairs[-1]['id'],'status':status,'normal_executable':status=='success'})
 core=P/'data/core1200_v1';cr=source(core/'roots_manifest.json')
 for r in cr:
  if r['environment']in'ABC':addroot(r,'core')
 target_by_episode={}
 for r in source(core/'data_manifest.json')['records']:
  if r['environment']not in'ABC':continue
  rid=r['root_id'];eid=r['episode_id'];sk=r['skill'];target=target_by_episode.get(eid)
  if family(sk)=='lift':target=COLORS.index(sk.split('_')[1]);target_by_episode[eid]=target
  # A first-place request has no object parameter in the old public contract.
  # It remains unbound, not retrospectively labelled with the held object's GT.
  b=state(r['obs_before'],rid);a=state(r['obs_after'],rid)
  native=read(Path(r['obs_before']).parent/'result_private.json')
  pair(rid,eid,r['step'],sk,b,a,target,0,native['status'],'core')
 lp=P/'data/predicate_long_pairs_20261014';m=source(lp/'data_manifest.json')
 for r in source(lp/'roots_manifest.json'):addroot(r,'longpair')
 wi={}
 for w in m['windows']:
  wi[w['window_id']]=state(w['public_frames'][-1],w['root_id'],P/'experiments/state_predicates_rgb_registry_v1_20261014/data/features'/(w['window_id']+'.npz'))
 private={r['pair_id']:r for r in m['private_pairs']}
 for r in m['records']:
  target=next(j for j,c in enumerate(COLORS)if'_'+c+'_'in r['root_id']);pr=private[r['pair_id']]
  pair(r['root_id'],r['root_id'],r['macro_index'],r['skill'],wi[r['before_window_id']],wi[r['after_window_id']],target,96,pr['primitive_result']['status']if'primitive_result'in pr else pr.get('primitive',{}).get('status','success'),'longpair')
 sb=P/'data/visual_state_semantics_v2_ABC_20261015'
 for r in source(sb/'roots_manifest.json'):addroot(r,'semantics')
 windows=source(BASE/'data/windows.json');oldwid={};windowmeta=[]
 for w in windows:
  i=state(w['public_frames'][-1],w['root_id'],BASE/'data/features'/(w['window_id']+'.npz'));oldwid[w['window_id']]=i
  t=w['targets_v2_1'];labels[str(i)]={'y':t['y'],'v':t['v']}
  windowmeta.append({'index':i,'root_id':w['root_id'],'split':w['split'],'condition':w['condition'],'anchor':w['anchor'],'roles':w['roles'],'window_id':w['window_id']})
 for r in source(BASE/'data/calls.json'):
  b=oldwid[r['episode_id']+f"__t{r['start']:05d}"];a=oldwid[r['episode_id']+f"__t{r['end']:05d}"]
  pair(r['root_id'],r['episode_id'],r['step'],r['skill'],b,a,r['object'],96,r['native']['status'],'semantics')
 # Public-label budgeting uses fixed root and collection-role strata, not labels.
 selected=[];devsel=[]
 for rid in sorted({w['root_id']for w in windowmeta}):
  ww=[w for w in windowmeta if w['root_id']==rid];rng=np.random.default_rng(16022026)
  if'chain'in rid:
   chosen=[w for w in ww if'before_action'in w['roles']or'after_action'in w['roles']]
   chosen+=sorted([w for w in ww if w not in chosen],key=lambda w:w['anchor'])[:3]
  else:
   chosen=[]
   for c in ['table_open','single_inside','edge_straddle','outside_front','stacked','separated','empty_mid','empty_closed']:
    candidates=[w for w in ww if w['condition']==c]
    # Alternate pre/post states for closing fixtures without inspecting truth.
    after=[w for w in candidates if'after_action'in w['roles']]
    want_after=c in['table_open','single_inside','stacked','separated']
    chosen.append((after if want_after and after else [w for w in candidates if w['anchor']==127])[0])
  assert len(chosen)==8
  (selected if roots[rid]['split']=='train'else devsel).extend(w['index']for w in chosen)
 # All trajectory links are actual equal endpoints within one episode only.
 chains=[]
 for p in pairs:
  chain=[p['id']];cur=p
  for _ in range(2):
   nxt=[q for q in pairs if q['episode_id']==cur['episode_id']and q['step']==cur['step']+1 and q['before']==cur['after']]
   if not nxt:break
   assert len(nxt)==1;cur=nxt[0];chain.append(cur['id'])
  chains.append(chain)
 hgroups=defaultdict(set)
 for s in states:hgroups[s['content_sha256']].add(s['split'])
 assert all(len(v)==1 for v in hgroups.values()),'cross-split identical public states'
 layouts=[(rid,np.array([r['layout']['block_'+c]for c in COLORS]))for rid,r in roots.items()if isinstance(r['layout'],dict)]
 overlap=[]
 for j,(a,x)in enumerate(layouts):
  for b,y in layouts[j+1:]:
   if roots[a]['split']!=roots[b]['split']and np.all(np.linalg.norm(x[:,:3]-y[:,:3],axis=1)<.012):overlap.append([a,b])
 assert not overlap,overlap
 save(E/'data/public/states.json',states);save(E/'data/public/pairs.json',pairs);save(E/'data/public/chains.json',chains);save(E/'data/public/roots.json',list(roots.values()))
 save(E/'data/_private/labels.json',labels);save(E/'data/_private/pair_outcomes.json',evaluation);save(E/'data/_private/window_metadata.json',windowmeta)
 save(E/'configs/semantic_budget.json',{'train_state_indices':selected,'development_selection_state_indices':devsel,'train_states':len(selected),'dev_selection_states':len(devsel),'max_binary_answers_train':len(selected)*9,'max_binary_answers_dev_selection':len(devsel)*9,'all_three_arms_identical':True,'selection_uses_labels':False,'train_true_observations_only':True,'additional_predicted_state_label_calibration':0})
 audit={'created_utc':now(),'source_manifests_sha256':sourcehash,'states':len(states),'cached_generic_current_states':sum('feature_path'in s for s in states),'roots_by_split':dict(Counter(r['split']for r in roots.values())),'pairs_by_split':dict(Counter(p['split']for p in pairs)),'roots_by_dataset':dict(Counter(r['dataset']for r in roots.values())),'pairs_by_dataset':dict(Counter(p['dataset']for p in pairs)),'chains_length3_by_split':dict(Counter(p['split']for p,c in zip(pairs,chains)if len(c)==3)),'cross_split_identical_states':0,'cross_split_near_initial_layouts':overlap,'old_D_loaded_into_new_dataset':False,'original_data_modified':False,'public_arrays':['rgb_static','rgb_gripper','robot_obs'],'first_place_without_declared_object_argument':sum(family(p['skill'])=='place'and not any(p['action'][4:7])for p in pairs),'posthold_contract_recorded_in_action':True}
 save(E/'results/data_audit.json',audit);print(json.dumps(audit),flush=True)
if __name__=='__main__':main()
