from common import *
from data import metadata,ns,VisibleRegistry
import torch
def main():
 guard();verify();start=time.time();models=read(S/'configs/models_frozen.json');fits=[read(p)for p in sorted((S/'results').glob('fit_*.json'))];assert len(fits)==4 and all(x['updates']==1500 for x in fits)
 for seed in [0,1]:assert len({x['initial_interface_sha256']for x in fits if x['seed']==seed})==1
 for p,h in models['sha256'].items():assert sha(p)==h
 for p in (S/'results').glob('features_*.json'):
  r=read(p);name=p.stem.replace('features_','');assert sha(S/f'cache/{name}.npz')==r['sha256']
 original=read(S/'configs/cache_audit.json');allhash={**original['source_sha256'],**original['cache_sha256']};assert all(sha(p)==h for p,h in allhash.items())
 routecounts=collections.Counter();hist_hashes={}
 for e in metadata('historical_evaluation'):
  for item in e['features']:
   f=dict(np.load(item['path']));frames=[dict(np.load(p))for p in item['public_inputs']];weights,slots,chosen,candidates=ns['public_routes'](frames,f['visible_registry_ids'].tolist(),f['valid']);h=item['step'];row=e['rows'][max(h-1,0)];private=read(Path(row['labels_path']).parent/'endpoint_info_private.json')['before'if h==0 else'after'];hist_hashes[item['path']]=sha(item['path']);routecounts['boundaries']+=1
   for j,c in enumerate(COLORS):
    uid=private['scene_info']['movable_objects']['block_'+c]['uid'];visible=any(np.sum(VisibleRegistry.grouped(fr['seg_'+cam])==uid)>=2 for fr in frames for cam in ['static','gripper']);routecounts['objects']+=1;routecounts['visible']+=visible;routecounts['missing']+=not bool(weights[j].any());routecounts['wrong']+=int(chosen[j]>=0 and chosen[j]!=uid);routecounts['unresolved_visible']+=int(visible and not weights[j].any())
 assert routecounts['wrong']==0 and routecounts['unresolved_visible']==0
 prefault=[]
 for path in sorted((S/'results').glob('predictions_*.npz')):
  z=dict(np.load(path));maxdiff=0.;pairs=0
  for rid in sorted(set(z['root_id'][z['kind']=='role'])):
   ids=np.where(z['root_id']==rid)[0];ix={(int(z['branch'][i]),str(z['mode'][i]),int(z['step'][i])):i for i in ids}
   for m,t in [('current',0),('current',1),('one_step',1),('one_step',2)]:
    diff=float(np.abs(z['prob'][ix[1,m,t]]-z['prob'][ix[2,m,t]]).max());maxdiff=max(maxdiff,diff)
   pairs+=1
  assert maxdiff<1e-6
  assert np.isfinite(z['prob']).all();prefault.append({'file':path.name,'role_pairs':pairs,'max_prefault_probability_difference':maxdiff})
 result={'utc':now(),'passed':True,'fixed_fits':4,'total_updates':6000,'all_four_frozen_before_45min':datetime.datetime.fromisoformat(models['utc']).timestamp()<read(S/'configs/launch.json')['training_deadline_epoch'],'train_dev_cache_sources_unchanged':len(allhash),'historical_route_audit':dict(routecounts),'historical_original_cache_sha256':hist_hashes,'prefault_checks':prefault,'model_and_shared_feature_hashes_unchanged':True,'shared_interface_initialization_identical_within_seed':True,'no_new_collection':True,'CD_samples':0,'uploads':0,'seconds':time.time()-start,'GT_usage':'Historical physical UID read only after frozen model inference for independent route audit; never passed to encoder, recurrence or action binding.'};save(S/'results/final_verification.json',result);print({k:v for k,v in result.items()if k not in ['historical_original_cache_sha256','prefault_checks']},flush=True)
if __name__=='__main__':main()
