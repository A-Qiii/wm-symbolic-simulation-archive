from common import *
import collections
def content(path):
 with np.load(path)as z:
  data={k:z[k].copy()for k in ['rgb_static','rgb_gripper','robot_obs']}
 h=hashlib.sha256()
 for k,v in sorted(data.items()):h.update(k.encode());h.update(str(v.shape).encode());h.update(str(v.dtype).encode());h.update(v.tobytes())
 return h.hexdigest(),data
def main():
 for p in[E/'source',E/'configs',E/'data',E/'results',R/'images',R/'assets']:p.mkdir(parents=True,exist_ok=True)
 protocol={'created_utc':now(),'purpose':'Sealed diagnostic visualization only, no model training, new simulation, model selection or threshold tuning.','model_seed':0,'representation':'Actual continuous interaction encoder output:3 object tokens+scene token,4x64 flattened256. Frozen v1 interaction seed0; identical non-transition parameters in repaired model. AE seed0 separate contextual comparison. Not query probabilities or raw DINO.','main_candidates':'Only actual before/after indices in recorded skill pairs with existing9-column ground-truth labels. No moving interior windows. Ground truth only used for colour/hover and post-hoc diagnostics, never fitted to representation/PCA/tSNE or sample selection.','sampling':{'per_root_max':8,'order':'SHA256 of tsne20261016:<batch>:<state index>, independent of labels and embedding','exact_dedup':'hash raw RGB and robot arrays; one actual representative, original IDs retained','near_dedup':'within same root, both-camera mean absolute RGB difference/255 <=0.005 and max absolute difference of15 proprio entries <=0.005; keep first hash-ordered actual state; no averaging','root_cap_after_dedup':True},'main_preprocess':{'latent_dimensions':256,'standardization':'mean/std fitted to selected AB training states only, std floor1e-6, separately for each encoder','PCA_dimensions':50,'PCA_fit':'all selected unlabeled points, full SVD; only diagnostic transductive coordinates','labels_used':False},'tsne':{'n_components':2,'perplexity':30,'random_state':42,'init':'pca','learning_rate':'auto','early_exaggeration':12,'max_iter':1500,'method':'barnes_hut','angle':.5,'metric':'euclidean','n_jobs':1},'stability':'One prespecified alternative perplexity15, same seed42 and all preprocessing/points; keep both, never select prettier one. Compare10-neighbour overlap.','joint':'D2 same selected pairs: actual successor, sealed control seed0 predicted successor, sealed matched seed0 predicted successor jointly fit once. Reuse main interaction AB scaling; PCA50 fitted unlabeled union.','joint_sampling':'Each of12 roots:held_closed,held_open,empty_open for even root ordinal else empty_closed; reference slot ordinal modulo planned reference length, omit unavailable endpoint and record omission. No outcome-driven replacement.','paired_lines':'Six predetermined pair IDs:held_closed on red_0/blue_0/pink_0 and held_open on red_1/blue_1/pink_1, two prediction-to-actual lines each.','environment':'A/B/C/D are actual CALVIN physical scene domains. D2 uses scene D, distinct used-test sampling batch only. Models share robot/cameras; appearances and scene geometry can differ.','outputs':['main shared-coordinate semantic/domain PNG','same-state cross-domain PNG','AE separate projection','shared successor joint PNG','stability PNG','offline Plotly HTML with click-to-RGB and truth/query text'],'no_installs':True}
 save(E/'configs/protocol.json',protocol);candidates=[];sources={}
 for batch,base in BASES.items():
  states=read(base/'public/states.json');pairs=read(base/'public/pairs.json');labels=read(base/'_private/labels.json');used=sorted({r[k]for r in pairs for k in['before','after']if str(r[k])in labels})
  for f in[base/'public/states.json',base/'public/pairs.json',base/'_private/labels.json',base/'public/inputs.npz']:sources[str(f)]=sha(f)
  roles=collections.defaultdict(list)
  for r in pairs:
   for k in['before','after']:roles[r[k]].append({'pair_id':r['id'],'role':k,'skill':r['skill']})
  for i in used:
   s=states[i];environment=s.get('environment')or('D'if batch in['D','D2']else s['root_id'][0]);assert environment in'ABCD';candidates.append({'key':f'{batch}:{i}','batch':batch,'state_index':i,'root_id':s['root_id'],'environment':environment,'split':s['split'],'sampling_group':'AB训练'if s['split']=='train'else'C开发'if s['split']=='dev'else'D已用测试'if batch=='D'else'D2已用测试','path':s['path'],'source_path':s.get('source',s['path']),'truth':labels[str(i)],'roles':roles[i],'order':hashlib.sha256(f'tsne20261016:{batch}:{i}'.encode()).hexdigest()})
 selected=[];decisions=[];hashes={};cached={}
 for root in sorted({r['root_id']for r in candidates}):
  kept=[]
  for row in sorted([r for r in candidates if r['root_id']==root],key=lambda r:r['order']):
   h,arr=content(row['path']);row['content_sha256']=h;match=hashes.get(h);reason='exact_content_duplicate'if match else None
   if match is None:
    for prev in kept:
     a=cached[prev['key']];rgb=float(np.mean([np.abs(arr[k].astype(float)-a[k].astype(float)).mean()/255 for k in['rgb_static','rgb_gripper']]));prop=float(np.max(np.abs(arr['robot_obs']-a['robot_obs'])))
     if rgb<=.005 and prop<=.005:match=prev;reason='near_input_duplicate';break
   if match is not None:decisions.append({'key':row['key'],'root_id':root,'reason':reason,'representative':match['key']});continue
   if len(kept)>=8:decisions.append({'key':row['key'],'root_id':root,'reason':'per_root_cap8'});continue
   kept.append(row);selected.append(row);hashes[h]=row;cached[row['key']]=arr
 selected.sort(key=lambda r:(r['batch'],r['root_id'],r['state_index']))
 for j,r in enumerate(selected):r['point_id']=f'S{j+1:03d}'
 save(E/'data/main_states.json',selected);save(E/'data/sampling_decisions.json',decisions);save(E/'data/all_boundary_candidates.json',candidates)
 # Joint selection references declared branches/slots only, before looking at projection or losses.
 roots=read(REPAIR/'configs/D2_roots.json');episodes=read(REPAIR/'results/D2_collection.json');pairs=read(REPAIR/'data/D2/public/pairs.json');pairby={ (r['root_id'],r['before'],r['after']):r for r in pairs};states=read(REPAIR/'data/D2/public/states.json');pathindex={r['path']:i for i,r in enumerate(states)};joint=[];omitted=[]
 for ordinal,root in enumerate(roots):
  requests=[('held_closed',0),('held_open',0),('empty_open'if ordinal%2==0 else'empty_closed',0),('reference',ordinal%len(root['normal_reference_skills']))]
  for condition,slot in requests:
   ep=next(r for r in episodes if r['root_id']==root['root_id']and r['condition']==condition);task=ep['tasks'][slot]
   if not task.get('after'):omitted.append({'root_id':root['root_id'],'condition':condition,'slot':slot,'reason':task['status']});continue
   b=pathindex[task['before']['public_path']];a=pathindex[task['after']['public_path']];r=pairby[root['root_id'],b,a];joint.append({'pair_id':r['id'],'root_id':r['root_id'],'condition':condition,'skill':r['skill'],'status':task['status'],'before_index':b,'after_index':a,'before_path':task['before']['public_path'],'after_path':task['after']['public_path'],'true_before':task['before']['labels'],'true_after':task['after']['labels'],'target':root['target'],'sampling_group':'D2已用测试','environment':'D'})
 save(E/'data/joint_pairs.json',joint);save(E/'data/joint_omissions.json',omitted);save(E/'results/sampling.json',{'candidate_boundaries':len(candidates),'selected':len(selected),'roots':len({r['root_id']for r in selected}),'root_max':max(collections.Counter(r['root_id']for r in selected).values()),'by_batch':dict(collections.Counter(r['batch']for r in selected)),'by_environment':dict(collections.Counter(r['environment']for r in selected)),'by_split':dict(collections.Counter(r['sampling_group']for r in selected)),'excluded':dict(collections.Counter(r['reason']for r in decisions)),'joint_pairs':len(joint),'joint_roots':len({r['root_id']for r in joint}),'joint_omissions':omitted,'GT_used_for_sampling':False})
 for arm in['canonical_control','canonical_matched']:
  f=REPAIR/'results'/f'D2_{arm}_seed0.npz';sources[str(f)]=sha(f)
 save(E/'configs/sources.json',{'sha256':sources});save(E/'configs/selection_frozen.json',{'created_utc':now(),'sha256':{str(p):sha(p)for p in[E/'configs/protocol.json',E/'data/main_states.json',E/'data/joint_pairs.json']}});print('SELECTION_FROZEN',read(E/'results/sampling.json'),flush=True)
if __name__=='__main__':main()
