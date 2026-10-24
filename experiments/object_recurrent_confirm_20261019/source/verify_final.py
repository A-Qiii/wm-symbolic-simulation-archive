from common import *
from data import metadata,ns,VisibleRegistry
def main():
 guard();verify();budget();roots=read(S/'data/roots_manifest.json');families=read(S/'data/families_manifest.json');excluded=read(S/'configs/AB_exclusion_inventory.json')['entries'];nearest=[]
 for i,f in enumerate(families):
  a=np.array([f['layout']['block_'+c][:3]for c in COLORS]);best=10.;bestsource='';old=excluded+[{'environment':g['environment'],'poses':[g['layout']['block_'+c]for c in COLORS],'source':g['family_id']}for g in families[:i]]
  for r in old:
   if r['environment']!=f['environment']:continue
   b=np.array(r['poses'])[:,:3];d=min(float(np.linalg.norm(a-b[list(order)],axis=1).max())for order in itertools.permutations(range(3)))
   if d<best:best=d;bestsource=r['source']
  assert best>.020;nearest.append({'family':f['family_id'],'nearest_minimax_permutation_distance_m':best,'nearest_source':bestsource})
 fs=read(S/'configs/models_frozen.json');assert all(sha(p)==h for p,h in fs['sha256'].items());assert read(O/'results/analysis.json')['training_fit_gate']is False;rootfreeze=read(S/'configs/roots_frozen.json');assert all(sha(p)==h for p,h in rootfreeze['sha256'].items());assert read(S/'results/collection_complete.json')['total_transitions']==576
 for p in (S/'results').glob('features_confirm_seed*.json'):
  z=read(p);assert sha(S/f"cache/{p.stem.replace('features_','')}.npz")==z['sha256']
 route=collections.Counter();public_hashes={}
 for e in metadata('confirm'):
  for item in e['features']:
   f=dict(np.load(item['path']));frames=[dict(np.load(p))for p in item['public_inputs']];weights,slots,chosen,candidates=ns['public_routes'](frames,f['visible_registry_ids'].tolist(),f['valid']);h=item['step'];row=e['rows'][max(h-1,0)];private=read(Path(row['labels_path']).parent/'endpoint_info_private.json')['before'if h==0 else'after'];route['boundaries']+=1
   for path in item['public_inputs']:public_hashes[path]=sha(path)
   for j,c in enumerate(COLORS):
    uid=private['scene_info']['movable_objects']['block_'+c]['uid'];visible=any(np.sum(VisibleRegistry.grouped(fr['seg_'+cam])==uid)>=2 for fr in frames for cam in ['static','gripper']);route['objects']+=1;route['visible']+=visible;route['missing']+=not bool(weights[j].any());route['wrong']+=int(chosen[j]>=0 and chosen[j]!=uid);route['unresolved_visible']+=int(visible and not weights[j].any())
 assert route['wrong']==0;prefault=[]
 for p in sorted((S/'results').glob('predictions_*.npz')):
  z=dict(np.load(p));assert np.isfinite(z['prob']).all();diff=0.
  for rid in sorted(set(z['root_id'])):
   ids=np.where(z['root_id']==rid)[0];ix={(int(z['branch'][i]),str(z['mode'][i]),int(z['step'][i])):i for i in ids}
   for m,t in [('current',0),('current',1),('one_step',1),('one_step',2)]:diff=max(diff,float(np.abs(z['prob'][ix[1,m,t]]-z['prob'][ix[2,m,t]]).max()))
  assert diff<1e-6;prefault.append({'file':p.name,'max_difference':diff})
 oldread=read(O/'MANIFEST.json');assert all(sha(O/r['path'])==r['sha256']for r in oldread['files']);save(S/'results/final_verification.json',{'utc':now(),'passed':True,'new_families':8,'historical_unique_geometries_excluded':len(excluded),'minimum_novelty_distance_m':min(x['nearest_minimax_permutation_distance_m']for x in nearest),'novelty':nearest,'models_unchanged':4,'all_previous_run_files_unchanged':len(oldread['files']),'old_training_fit_gate_unchanged_false':True,'prefault':prefault,'route':dict(route),'public_inputs_sha256':public_hashes,'training_updates':0,'threshold_changes':0,'CD_rollouts_and_scoring':0,'uploads':0,'GT_usage':'Only offline labels, scores, and post-inference physicalUID route audit; never model feature/action-binding input.'});print('VERIFICATION_PASSED',dict(route),flush=True)
if __name__=='__main__':main()
