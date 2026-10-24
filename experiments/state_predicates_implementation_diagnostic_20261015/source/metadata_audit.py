from audit_common import *
from collections import Counter
def normalize(f,stats):
 vm=np.array(stats['vision_mean'],np.float32);vs=np.array(stats['vision_std'],np.float32);rm=np.array(stats['robot_mean'],np.float32);rs=np.array(stats['robot_std'],np.float32);g=f['geometry'];vis=g[:,:,:,0,4].astype(bool);tok=(f['local_tokens'].astype(np.float32)-vm)/vs;ob=np.zeros((4,3,2,390),np.float32)
 for j in range(3):
  if not vis[:,:,j].any():continue
  ob[:,j,:,:384]=tok[:,:,j].mean(2)*vis[:,:,j,None];ob[:,j,:,384:389]=g[:,:,j,0,:5];ob[:,j,:,389]=g[:,:,j,:,5].mean(2)
 grid=(f['global_grid'].astype(np.float32)-vm)/vs;return {'objects':ob.reshape(4,3,780),'context':grid.reshape(4,2,2,2,2,2,384).mean((3,5)).reshape(4,3072),'robot':(f['robot_history']-rm)/rs}
def main():
 q=frozen();meta=read(TRAIN/'data/_private/evaluation_metadata.json');z=dict(np.load(TRAIN/'data/public/inputs.npz'));t=dict(np.load(TRAIN/'data/_private/targets.npz'));assert len(meta)==174;dm=read(DB/'data_manifest.json');dp={r['pair_id']:r for r in dm['private_pairs']};rows=[];verified=0
 for dataset,rr in [('ABC',meta),('D',dm['records'])]:
  for i,r in enumerate(rr):
   if dataset=='ABC':a=z['declared_actions'][i];before=t['before_y'][i];valid=t['before_valid'][i];family=r['family'];condition=r['condition'];src=r['source'];group='AB'if r['split']=='train'else'C';status=r['primitive_status']
   else:
    pr=dp[r['pair_id']];audit=read(r['preaction_prediction_record_path']);assert sha(audit['predictions_path'])==audit['predictions_sha256'];a=np.load(audit['predictions_path'])['declared_actions'][0];before=np.array(pr['before_targets']['y']);valid=np.array(pr['before_targets']['valid']);family=pr['family'];condition=pr['condition'];src='independent_D';group='D';status=pr['primitive']['status'];decl=read(DB/'episodes'/r['episode_id']/'public/diagnostic_declaration.json');assert action(decl).tolist()==a.tolist()
   expected=action(r);assert np.array_equal(a,expected);verified+=1;held=[COLORS[j]for j in range(3)if before[j]and valid[j]];target=r['declared_arguments'].get('color');heldname=held[0]if len(held)==1 else'empty'if not held else'multiple';rows.append({'pair_id':r['pair_id'],'root_id':r['root_id'],'environment':r['environment'],'group':group,'source':src,'family':family,'condition':condition,'status_saved':status,'held_colour_truth_scoring_only':heldname,'target_colour_declared':target,'drawer_open_truth_scoring_only':bool(before[6]),'before_all7_valid':bool(np.all(valid)),'held_other':bool(family=='lift'and len(held)==1 and held[0]!=target),'action_vector_saved':a.tolist(),'action_vector_matches_declaration':True,'target_slot':COLORS.index(target)if target else None})
 csvsave(E/'results/all_binding_rows.csv',rows);save(E/'results/all_binding_rows.json',rows);tables=[]
 for group in ['AB','C','D']:
  for drawer in ['all','closed','open']:
   selected=[r for r in rows if r['group']==group and r['family']=='lift'and(drawer=='all'or r['drawer_open_truth_scoring_only']==(drawer=='open'))];table={h:{target:sum(r['held_colour_truth_scoring_only']==h and r['target_colour_declared']==target for r in selected)for target in COLORS}for h in ['empty']+COLORS};tables.append({'group':group,'drawer':drawer,'lift_calls':len(selected),'independent_roots':len(set(r['root_id']for r in selected)),'rows_held_columns_target':table})
 save(E/'results/combination_tables.json',tables);core=[r for r in rows if r['held_other']];csvsave(E/'results/held_other_combinations.csv',core);pairs={g:sorted(set((r['held_colour_truth_scoring_only'],r['target_colour_declared'])for r in core if r['group']==g))for g in ['AB','C','D']};assert pairs['AB']==pairs['D'] and set(pairs['C']).issubset(set(pairs['AB']));alloff={(h,target)for h in COLORS for target in COLORS if h!=target};counts={g:dict(Counter(r['held_colour_truth_scoring_only']+'->'+r['target_colour_declared']for r in core if r['group']==g))for g in ['AB','C','D']};drawer_counts={g:dict(Counter('open'if r['drawer_open_truth_scoring_only']else'closed'for r in core if r['group']==g))for g in ['AB','C','D']}
 # Prove cached training targets/current state are copies of the same reader outputs, no latent renormalization.
 ox=dict(np.load(OLD/'data/public/transition_inputs.npz'));ot=dict(np.load(OLD/'data/_private/training_targets_and_audit.npz'));nx=dict(np.load(ABC/'results/latents.npz'));na=np.load(ABC/'data/public/inputs_full.npz')['declared_action_features'];nt=dict(np.load(ABC/'data/_private/supervision_full.npz'));assert [r['source']for r in meta]==['old']*120+['new']*54
 source_map={'current_objects':(z['current_objects'],ox['current_objects'],nx['current_objects']),'current_scene':(z['current_scene'],ox['current_scene'],nx['current_scene']),'declared_actions':(z['declared_actions'],ox['declared_actions'],na),'next_objects':(t['next_objects'],ot['next_objects'],nx['actual_next_objects']),'next_scene':(t['next_scene'],ot['next_scene'],nx['actual_next_scene'])}
 for key in ['before_y','before_valid','after_y','after_valid']:source_map[key]=(t[key],ot[key],nt[key])
 for key,(merged,old,new)in source_map.items():assert np.array_equal(merged,np.concatenate([old,new])),key
 stats=read(CAND/'normalization.json');normchecks=[]
 for group,wm,inputs in [('old_ABC',read(RGB/'data/public/windows_v1.json'),dict(np.load(RGB/'data/public/paired_inputs_v1.npz'))),('new_ABC',read(ABC/'data/public/windows_full.json'),dict(np.load(ABC/'data/public/inputs_full.npz')))]:
  for i,w in enumerate(wm):
   fp=w.get('raw_RGB_feature_path',w.get('raw_features_path'));rec=normalize(dict(np.load(fp)),stats);assert all(np.array_equal(rec[k],inputs[k][i])for k in rec);normchecks.append({'group':group,'window_id':w['window_id'],'raw_path':fp,'all_normalized_arrays_byte_equal':True})
 for r in dm['records']:
  audit=read(r['preaction_prediction_record_path']);rec=normalize(dict(np.load(audit['raw_features_path'])),stats);saved=dict(np.load(audit['normalized_path']));assert all(np.array_equal(rec[k],saved[k])for k in rec);normchecks.append({'group':'D_saved_before_only_no_inference','window_id':r['before_window_id'],'raw_path':audit['raw_features_path'],'all_normalized_arrays_byte_equal':True})
 registry=module('pure_frozen_colour_registry',RGB/'source/registry.py');swatch=np.zeros((16,48,3),np.uint8)
 for j,rgb in enumerate([(255,0,0),(0,0,255),(255,0,255)]):swatch[:,j*16:(j+1)*16]=rgb
 masks,_=registry.extract(swatch)
 for j in range(3):expected=np.zeros((16,48),bool);expected[:,j*16:(j+1)*16]=True;assert np.array_equal(masks[j],expected)
 save(E/'results/normalization_checks.json',normchecks);result={'created_utc':now(),'passed':True,'all258_saved_action_vectors_exactly_match_declaration':verified==258,'slot_order_everywhere':COLORS,'action_order':['open','lift','place','close','target_red','target_blue','target_pink','budget/900','posthold/96'],'held_other_AB_and_D_combinations_identical_C_subset':True,'mapping_held_to_target':pairs,'held_other_counts':counts,'held_other_drawer_counts':drawer_counts,'uncovered_reverse_combinations_in_all_groups':sorted(alloff-set(map(tuple,pairs['AB']))),'training_merge_arrays_exact':list(source_map),'normalization_windows_byte_equal':len(normchecks),'normalization_hash':sha(CAND/'normalization.json'),'RGB_registry_swatch_slot_check_passed':True,'D_new_model_or_encoder_forward_calls':0,'D_predictions_or_scores_modified':False};save(E/'results/binding_summary.json',result);print(json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
