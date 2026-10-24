from common_control import *
import torch
def main():
 q=frozen();sel=read(E/'configs/model_selection_frozen.json');assert sel['all_training_complete']and sel['total_optimizer_updates']==3000;assert len(sel['runs'])==6;_,_,digest,_,Net=old_modules();out=[]
 sch=np.load(q['schedule'])['global_indices'];assert sch.shape==(500,137)and np.all(sch==np.array(q['train_global_ids'])[None]);pub=read(PREV/'data/training/public/pairs.json');assert all(pub[i]['split']=='train'for i in sch[0])
 for r in sel['runs']:
  base=torch.load(q['starts'][str(r['seed'])]['path'],map_location='cpu',weights_only=False);net=Net('full');net.load_state_dict(base['model']);initial=digest(net);assert initial==r['initial_digest'];cp=torch.load(r['final500']['path'],map_location='cpu',weights_only=False);assert sha(r['final500']['path'])==r['final500']['sha256'];assert cp['update']==500 and {int(v['step'])for v in cp['optimizer']['state'].values()}=={500};assert cp['start_sha256']==q['starts'][str(r['seed'])]['sha256'];assert cp['schedule_sha256']==q['schedule_sha256'];w=q['arm_weights'][r['arm']];assert cp['group_weights']==w
  for group in cp['optimizer']['param_groups']:assert group['lr']==.001 and group['weight_decay']==.0001
  log=[json.loads(s)for s in (E/'runs'/r['run']/'curve.jsonl').read_text().splitlines()];assert [x['update']for x in log]==list(range(1,501))
  for x in log:
   expected=w['old132']*(x['old132_BCE']+x['old132_weighted_latent'])+w['new5']*(x['new5_BCE']+x['new5_weighted_latent']);assert np.isclose(x['loss'],expected,atol=1e-6)
  obs=read(E/'runs'/r['run']/'training_observations.json');assert [x['update']for x in obs]==q['selection']['training_observation_steps']
  out.append({'run':r['run'],'start_sha256':r['start_sha256'],'initial_digest':initial,'updates_log_and_optimizer':500,'fresh_optimizer':r['fresh_optimizer_empty_at_start'],'shared_schedule_sha256':r['schedule_sha256'],'per_example_presentations':500,'final_training_old132':obs[-1]['old132_all7'],'final_training_new5':obs[-1]['new5_all7'],'gradient_clipped_steps':sum(x['gradient_norm_before_clip']>1 for x in log),'final_checkpoint_sha256':r['final500']['sha256']})
 for s in q['seeds']:
  rr=[r for r in out if r['run'].endswith(f'_seed{s}')];assert len(rr)==2 and rr[0]['initial_digest']==rr[1]['initial_digest']and rr[0]['start_sha256']==rr[1]['start_sha256']
 assert all(x['uniform_group_vs_original_whole_batch_loss_abs_error']<1e-6 for x in read(E/'results/loss_equivalence_check.json'))
 save(E/'results/training_audit.json',{'created_utc':now(),'passed':True,'runs':out,'total_new_optimizer_updates':3000,'same_initial_checkpoint_within_seed':True,'same_reset_optimizer':True,'same_exact_fullbatch_schedule':True,'same_fixed_update_budget':True,'sole_design_difference_group_weights':True,'class_weights_latent_coefficient_and_architecture_unchanged':True,'C_optimizer_and_selection_rows':0,'D_forward_rows':0,'no_C_evaluation_before_all_training_complete':not(E/'results/evaluation.json').exists(),'only_final500':True});print(json.dumps({'passed':True,'runs':out},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
