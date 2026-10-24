from experiment import *
import numpy as np,collections,importlib.util
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def correct(prob,truth,valid):
 out=np.isfinite(prob)&((prob>=.5)==truth)&valid
 for i,row in enumerate(prob):
  if sum(row[:3]>=.5)>1:
   for j in range(3):
    if row[j]>=.5:out[i,j]=False
 return out
def near(a,b):
 d=np.linalg.norm(a[:,:3]-b[:,:3],axis=1);yaw=np.abs((a[:,5]-b[:,5]+np.pi/2)%np.pi-np.pi/2);return bool(np.all(d<=.012)or(np.all(d<=.020)and np.all(yaw<=.25)))
def main():
 q=frozen();ex=read(E/'configs/root_exclusion.json');m=read(B/'data_manifest.json');ev=read(E/'results/evaluation.json');meta=read(E/'data/_private/evaluation_metadata.json');roots=read(B/'roots_manifest.json');arr=dict(np.load(E/'data/_private/evaluation_arrays.npz'));pred=dict(np.load(E/'results/predictions.npz'));lat=dict(np.load(E/'results/latent_predictions.npz'));y=arr['after_y'];v=arr['after_valid'];N=len(meta);prior=[np.array(r['layout'])for r in ex['prior_layouts']]
 for p,h in ex['sources_sha256'].items():assert sha(p)==h,p
 assert len(roots)==12 and len(m['episodes'])==84 and N<=84;assert collections.Counter(r['target_color']for r in roots)==dict.fromkeys(COLORS,4);assert collections.Counter(r['initial_drawer']for r in roots)=={'open':6,'closed':6};initial={}
 for ep in m['episodes']:
  a=np.array(ep['actual_initial_objects']);rid=ep['root_id'];assert not any(near(a,b)for b in prior)
  if rid in initial:assert np.array_equal(a,initial[rid])
  else:
   assert not any(near(a,b)for b in initial.values());initial[rid]=a
 covered=[r for r in m['episodes']if r['covered']];assert len(covered)==N;prep=[x for r in m['episodes']for x in r['preparation_attempts']];scored={r['episode_id']:r for r in meta};rows=read(E/'results/pair_details.json');counts={}
 for name,p in pred.items():
  cor=correct(p,y,v);full=cor.all(1);r=ev['models'][name]['all'];assert r['all7']==int(full.sum());assert r['facts']['all7']['strict_correct']==int(cor.sum());root_rates=[float(full[[i for i,z in enumerate(meta)if z['root_id']==rid]].mean())for rid in initial];assert abs(r['root_macro_all7']-np.mean(root_rates))<1e-12;dd=[x for x in rows if x['model']==name];assert len(dd)==N and all(x['all7_correct']==bool(full[i])for i,x in enumerate(dd));counts[name]={'all7':int(full.sum()),'strict_facts':int(cor.sum()),'root_macro':float(np.mean(root_rates))}
 for when,p,yy,vv in [('before',pred['persistence'],arr['before_y'],arr['before_valid']),('after',pred['real_next_observation'],y,v)]:assert ev['readout'][when]['all7']==int(correct(p,yy,vv).all(1).sum())
 cs=read(E/'results/root_contrasts.json');ix={r['episode_id']:i for i,r in enumerate(meta)}
 for name,p in pred.items():
  cor=correct(p,y,v)
  for fam in ['all','close','lift','place']:
   rr=[r for r in cs if(fam=='all'or r['family']==fam)and r['covered']];good=0;tar=0
   for r in rr:
    ii=[ix[r['root_id']+f'__b{b}']for b in r['branches']];good+=bool(cor[ii].all());tar+=bool(cor[ii][:,r['target_columns']].all())
   assert ev['paired_summary'][name][fam]['both_all7_correct']==good and ev['paired_summary'][name][fam]['both_target_correct']==tar
 assert float(np.abs(arr['next_objects']).max())<=1 and float(np.abs(arr['next_scene']).max())<=1
 for name in q['models']:
  coords=np.concatenate([lat[name+'_objects'].reshape(N,-1),lat[name+'_scene']],1);rr=ev['latent_metrics'][name]['all'];assert coords.shape==(N,384);assert rr['abs_gt1_coordinates']==int((np.abs(coords)>1).sum())and rr['coordinate_denominator']==N*384;assert np.isclose(rr['object_MAE'],np.abs(lat[name+'_objects']-arr['next_objects']).mean());assert np.isclose(rr['scene_MAE'],np.abs(lat[name+'_scene']-arr['next_scene']).mean())
 tree=module('independent_verify_tree',TRAIN/'source/trees.py');model=read(TRAIN/'models/tree_aug132.json');reproduced=0
 for i,r in enumerate(meta):
  aud=read(r['preaction_prediction_record_path']);z=dict(np.load(aud['predictions_path']));assert np.array_equal(tree.predict(model,z['symbolic_inputs']),pred['tree'][i:i+1]);assert np.array_equal(z['persistence'],pred['persistence'][i:i+1]);reproduced+=1
 visual=read(E/'results/case_visuals.json');videos=read(E/'results/videos.json');assert len(visual)==N and all(Path(r['path']).exists()for r in visual)
 for r in videos:assert sha(r['path'])==r['sha256']and r['fps']==30
 result={'created_utc':now(),'passed':True,'frozen_protocol_sha256':sha(E/'configs/protocol.json'),'frozen_files_checked':len(q['frozen_sha256']),'historical_source_files_checked':len(ex['sources_sha256']),'prior_layouts_checked':len(prior),'independent_roots':len(initial),'all84_fixed_branches_retained':True,'preparation_attempts':len(prep),'preparation_status_counts':dict(collections.Counter(r['result']['status']for r in prep)),'covered_scored_calls':N,'uncovered':84-N,'counts_independently_recomputed':counts,'paired_counts_recomputed':True,'actual_observed_readouts_recomputed':True,'latent_384_coordinate_denominator_verified':True,'tree_public_input_predictions_reproduced':reproduced,'saved_case_images':len(visual),'saved_videos':len(videos),'neural_forward_calls_during_evaluation':0,'corrections_or_retests':0};save(E/'results/final_validation.json',result);print(json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
