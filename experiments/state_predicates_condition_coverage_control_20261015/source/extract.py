from training_core import *
from matching import match
def main():
 q=frozen();assert read(E/'results/data_validation.json')['passed'];m=read(B/'data_manifest.json');roots=read(B/'roots_manifest.json');pubs={p['episode_id']:p for p in m['records']};matches=[]
 for root in roots:
  if not all(b['episode_id']in pubs for b in root['branches']):continue
  frames=[read(B/'episodes'/b['episode_id']/'_private'/(pubs[b['episode_id']]['before_window_id']+'.json'))['frames']for b in root['branches']];matches.append(match(root,*frames,q))
 gate=len(matches)==6 and all(x['strictly_matched']for x in matches)and len(m['records'])==12 and all(all(p['before_targets']['valid'])and all(p['after_targets']['valid'])for p in m['private_pairs']);save(E/'results/training_collection_gate.json',{'passed':bool(gate),'matched_pairs':matches,'completed_calls':len(m['records']),'exact138_design_available':bool(gate),'no_failed_or_readout_wrong_samples_removed':True});assert gate,'Exact paired138 design incomplete; retain all attempts, no substitutes or reduced training experiment.'
 assert not(E/'configs/dataset_frozen.json').exists();z={k:[v]for k,v in dict(np.load(TRAIN/'data/public/inputs.npz')).items()};t={k:[v]for k,v in dict(np.load(TRAIN/'data/_private/targets.npz')).items()};meta=read(TRAIN/'data/_private/evaluation_metadata.json');basepub=read(TRAIN/'data/public/pairs.json');oldpred=dict(np.load(TRAIN/'results/predictions.npz'));oldlat=dict(np.load(TRAIN/'results/latent_predictions.npz'));background={};blat={}
 for seed in [0,1,2]:
  name=f'original132_seed{seed}';key=f'aug132_seed{seed}_final1500';background[name]=[oldpred[key]]
  for kind in ['objects','scene']:blat[name+'_'+kind]=[oldlat[key+'_'+kind]]
 for row in meta:row['evaluation_group']='AB_original132'if row['split']=='train'else'C_old'if row['source']=='old'else'C_state'
 for cohort,dm in [('C_drawer',read(CB/'data_manifest.json')),('AB_coverage',m)]:
  private={p['pair_id']:p for p in dm['private_pairs']};rmap={r['root_id']:r for r in read((CB if cohort=='C_drawer'else B)/'roots_manifest.json')}
  # Root+branch order is invariant to counterbalanced physical execution order.
  for pub in sorted(dm['records'],key=lambda p:(list(rmap).index(p['root_id']),int(p['episode_id'].rsplit('__b',1)[1]))):
   pr=private[pub['pair_id']];root=rmap[pub['root_id']];branch=root['branches'][int(pub['episode_id'].rsplit('__b',1)[1])];a=read(pub['preaction_prediction_record_path']);af=read(pub['after_observation_record_path']);bo=dict(np.load(a['observed_path']));ao=dict(np.load(af['observed_path']));pp=dict(np.load(a['predictions_path']));assert all(pr['before_targets']['valid'])and all(pr['after_targets']['valid']);cur=bo['probability'];act=pp['declared_actions']
   vals={'current_objects':bo['current_objects'],'current_scene':bo['current_scene'],'declared_actions':act,'observed_current_probabilities':cur,'symbolic_inputs':symbol_inputs(cur,act)}
   for k,v in vals.items():z[k].append(v)
   for side in ['before','after']:
    t[side+'_y'].append(np.array([pr[side+'_targets']['y']],np.float32));t[side+'_valid'].append(np.array([pr[side+'_targets']['valid']],bool))
   t['next_objects'].append(ao['current_objects']);t['next_scene'].append(ao['current_scene']);t['observed_next_probabilities'].append(ao['probability']);meta.append(pub|{'source':'drawer_factor'if cohort=='C_drawer'else'coverage','evaluation_group':cohort,'planned_colour_evaluation_only':root['target_color'],'other_colour_evaluation_only':root['other_color'],'drawer_condition':branch['expected_drawer'],'primitive_status':pr['primitive']['status'],'primitive_reason':pr['primitive'].get('reason'),'family':pr['family'],'condition':pr['condition'],'contrast_role':pr['contrast_role']});basepub.append({k:pub[k]for k in ['pair_id','root_id','environment','split','dev_role','skill','declared_arguments','declared_primitive_budget','declared_posthold_controls','before_window_id','after_window_id']}|{'source':cohort})
   for seed,pname in [(0,'candidate_seed0'),(1,'parallel_seed1'),(2,'parallel_seed2')]:
    name=f'original132_seed{seed}';background[name].append(pp[pname])
    for kind in ['objects','scene']:blat[name+'_'+kind].append(pp[pname+'_'+kind])
 z={k:np.concatenate(v)for k,v in z.items()};t={k:np.concatenate(v)for k,v in t.items()};background={k:np.concatenate(v)for k,v in background.items()};blat={k:np.concatenate(v)for k,v in blat.items()};assert len(meta)==198 and len(set(r['pair_id']for r in meta))==198;assert len(set(r['root_id']for r in meta))==60;assert all(len({r['split']for r in meta if r['root_id']==rid})==1 for rid in set(r['root_id']for r in meta));common=[i for i,r in enumerate(meta)if r['evaluation_group']=='AB_original132'];assert len(common)==132;cohorts={}
 for name,drawer in [('closed138','closed'),('open138','open')]:
  add=[next(i for i,r in enumerate(meta)if r['root_id']==root['root_id']and r.get('drawer_condition')==drawer)for root in roots];cohorts[name]=common+add;assert len(cohorts[name])==138
 for a,b in zip(cohorts['closed138'][132:],cohorts['open138'][132:]):assert meta[a]['root_id']==meta[b]['root_id']and np.array_equal(z['declared_actions'][a],z['declared_actions'][b])
 assert {g:sum(r['evaluation_group']==g for r in meta)for g in ['C_old','C_state','C_drawer']}=={'C_old':24,'C_state':18,'C_drawer':12};assert all(meta[i]['split']=='train'for ids in cohorts.values()for i in ids)
 files=[E/'data/training/public/inputs.npz',E/'data/training/_private/targets.npz',E/'data/training/public/pairs.json',E/'data/training/_private/metadata.json',E/'data/training/public/cohorts.json',E/'results/original132_cached_predictions.npz',E/'results/original132_cached_latents.npz'];np.savez_compressed(files[0],**z);np.savez_compressed(files[1],**t);save(files[2],basepub);save(files[3],meta);save(files[4],cohorts);np.savez_compressed(files[5],**background);np.savez_compressed(files[6],**blat);bad=~(known(z['observed_current_probabilities'])&((z['observed_current_probabilities']>=.5)==t['before_y'])&t['before_valid']).all(1);afterbad=~(known(t['observed_next_probabilities'])&((t['observed_next_probabilities']>=.5)==t['after_y'])&t['after_valid']).all(1)
 d={'created_utc':now(),'passed':True,'rows':198,'old_AB':132,'new_AB':12,'dev':54,'cohorts':{k:len(v)for k,v in cohorts.items()},'shared_first132_identical':True,'new_root_indices_pairwise_correspond':True,'readout_wrong_rows_retained':[{'pair_id':r['pair_id'],'group':r['evaluation_group'],'before_wrong':bool(bad[i]),'after_wrong':bool(afterbad[i])}for i,r in enumerate(meta)if bad[i]or afterbad[i]],'C_optimizer_rows':0,'files_sha256':{str(f):sha(f)for f in files},'protocol_sha256':sha(E/'configs/protocol.json'),'old132_predictions_reused_no_new_reference_model_forward':True};save(E/'configs/dataset_frozen.json',d)
 for f in files+[E/'configs/dataset_frozen.json']:f.chmod(0o444)
 print(json.dumps(d,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
