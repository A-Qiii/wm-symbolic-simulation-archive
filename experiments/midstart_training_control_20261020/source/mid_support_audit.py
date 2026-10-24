from common import *
def main():
 guard();train=read(O/'cache/train_metadata.json');dev=read(O/'cache/development_metadata.json');z=dict(np.load(O/'cache/train_seed0.npz'));cat=z['cat'];audit=[];aligned=0
 for eidx,e in enumerate(train):
  labs=[read(r['labels_path'])for r in e['rows']];truths=[labs[0]['before']]+[l['after']for l in labs]
  for t,truth in enumerate(truths):
   assert np.array_equal(cat[eidx,t],truth['categorical']),(e['episode_id'],t)
   assert len(e['features'][t]['public_inputs'])==4
   for path in e['features'][t]['public_inputs']:assert 'private'not in path and Path(path).exists()
   aligned+=1
 for seed in [0,1]:
  s=dict(np.load(S/f'configs/schedule_seed{seed}.npz'));ids=s['ids'].reshape(-1);starts=s['starts'].reshape(-1)
  for arm in ARMS:
   actions=collections.Counter();states=collections.Counter();origins=collections.Counter();unique=set();st=np.zeros_like(starts)if arm=='full'else starts
   for eidx,start in zip(ids,st):
    e=train[eidx];unique.add((int(eidx),int(start)));origins[int(start)]+=1
    for t in range(start,e['length']):actions[e['rows'][t]['skill']]+=1
    for t in range(start,e['length']+1):
     for c in cat[eidx,t,:3]:states[str(int(c))]+=1
   audit.append({'arm':arm,'seed':seed,'actions':dict(actions),'object_categorical_supervision':dict(states),'unique_initial_origins':len(unique),'unique_initial_any_In':sum(bool((cat[i,t,:3]==3).any())for i,t in unique),'unique_initial_any_Held':sum(bool((cat[i,t,:3]==6).any())for i,t in unique),'unique_training_episodes':len(set(i for i,t in unique)),'sampled_origins':dict(origins)})
 for seed in [0,1]:
  a=next(r for r in audit if r['seed']==seed and r['arm']=='history_mix');b=next(r for r in audit if r['seed']==seed and r['arm']=='cold_mix');assert all(a[k]==b[k]for k in a if k!='arm')
 trainroots=set(e['root_id']for e in train);devroots=set(e['root_id']for e in dev);trainfam=set(e['family_id']for e in train);devfam=set(e['family_id']for e in dev);assert not trainroots&devroots and not trainfam&devfam
 save(S/'results/support_and_leakage_audit.json',{'utc':now(),'passed':True,'all_cached_labels_align_original_boundary':aligned,'matching_crop_action_and_state_exposure':True,'train_roots':len(trainroots),'train_families':len(trainfam),'dev_roots':len(devroots),'dev_families':len(devfam),'overlap_roots':[],'overlap_families':[],'audit':audit,'no_future_rule':'All initialization features are original causal four-control-frame windows at the selected boundary; warm prefix consumes only earlier boundary observations/actions. Sampling code selects starts from episode length, never from labels.','known_metadata_limit':'Underlying feature records may retain split=train from older cache production. Authoritative episode split is original O/train_metadata vs O/development_metadata, root/family disjointness verified here.','new_training_data':0});print('SUPPORT_AND_LEAKAGE_PASS',aligned,flush=True)
if __name__=='__main__':main()
