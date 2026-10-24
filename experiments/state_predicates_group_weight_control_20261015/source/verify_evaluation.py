from common_control import *
def main():
 q=frozen();ev=read(E/'results/evaluation.json');sel=read(E/'configs/model_selection_frozen.json');audit=read(E/'results/training_audit.json');assert audit['passed']and audit['total_new_optimizer_updates']==3000;assert datetime.datetime.fromisoformat(ev['created_utc'])>datetime.datetime.fromisoformat(sel['created_utc']);pi=read(E/'results/prediction_input_audit.json');assert pi['D_forward_calls']==0 and len(pi['models'])==6
 for r in pi['models']:assert r['evaluation_rows']==196 and r['input_keys']==['current_objects','current_scene','declared_actions']and r['D_rows']==0
 t=np.load(PREV/'data/training/_private/targets.npz');meta=read(PREV/'data/training/_private/metadata.json');pred=np.load(E/'results/predictions.npz');lat=np.load(E/'results/latent_predictions.npz');old=np.load(PREV/'results/predictions.npz');assert not any(r['environment']=='D'for r in meta)
 y=t['after_y'].astype(bool);valid=t['after_valid'];dev=np.array([r['split']=='dev'for r in meta]);pairs=read(E/'results/root_contrasts.json');checked=[]
 for name in [f'{a}_seed{s}'for a in ['start','uniform','equal_groups']for s in [0,1,2]]:
  p=pred[name];assert p.shape==(196,7);binary=p>=.5;conflict=binary[:,:3].sum(1)>1;correct=(binary==y)&valid&np.isfinite(p);correct[:,:3]&=~(conflict[:,None]&binary[:,:3]);full=correct.all(1)
  if name.startswith('start'):assert np.array_equal(p,old['open137_seed'+name[-1]])
  for group in ['C_old','C_state','C_drawer','C_combined']:
   mask=dev if group=='C_combined'else np.array([r['evaluation_group']==group for r in meta]);assert int(full[mask].sum())==ev['metrics'][name][group]['all']['all7'];a=np.concatenate([lat[name+'_objects'][mask].reshape(-1,288),lat[name+'_scene'][mask]],1);b=np.concatenate([t['next_objects'][mask].reshape(-1,288),t['next_scene'][mask]],1);assert np.isclose(np.abs(a-b).mean(),ev['latent_metrics'][name][group]['all384_MAE'],atol=1e-7)
  for group in ['C_state','C_drawer']:
   rs=[r for r in pairs if r['group']==group];n=sum(bool(full[r['ids']].all())for r in rs);assert n==ev['paired_summary'][name][group]['both_all7']
  checked.append(name)
 assert len([r for r in meta if r['evaluation_group']=='C_old'])==24 and len([r for r in meta if r['evaluation_group']=='C_state'])==18 and len([r for r in meta if r['evaluation_group']=='C_drawer'])==12
 save(E/'results/evaluation_audit.json',{'created_utc':now(),'passed':True,'models_checked':checked,'C_counts_and_pair_counts_recomputed_from_saved_probabilities':True,'latent_MAE_recomputed':True,'starting_outputs_match_existing_cache':True,'evaluation_after_all_training_complete':True,'only_current_public_inputs_used_by_predictor':True,'C_rows_in_training_or_selection':0,'D_forward_rows':0,'new_model_forwards_in_this_audit':0,'protocol_sha256':sha(E/'configs/protocol.json')});print('EVALUATION_AUDIT_PASSED',flush=True)
if __name__=='__main__':main()
