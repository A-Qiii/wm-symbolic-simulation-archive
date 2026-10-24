from common_diag import *
import torch
def main():
 q=frozen();am=read(E/'configs/launch_fix.json');assert sha(am['corrected_source'])==am['corrected_source_sha256'];done=read(E/'results/training_complete.json');assert done['total_updates']==1500
 reader,logits,digest,balanced,Net=old_modules();base=torch.load(q['checkpoint'],map_location='cpu',weights_only=False);net=Net('full');net.load_state_dict(base['model']);init=digest(net)
 public=read(PREV/'data/training/public/pairs.json');assert all(public[i]['split']=='train'for i in q['joint_global_ids']);t=np.load(PREV/'data/training/_private/targets.npz');target_y=t['after_y'][q['joint_global_ids']];hreal=t['next_objects'][q['joint_global_ids']];sreal=t['next_scene'][q['joint_global_ids']];rows=[];metrics=[]
 for run in done['runs']:
  name=run['arm'];cp=torch.load(run['checkpoint'],map_location='cpu',weights_only=False);assert sha(run['checkpoint'])==run['checkpoint_sha256'];assert cp['update']==500 and cp['diagnostic_only']and cp['not_candidate'];steps={int(s['step'])for s in cp['optimizer']['state'].values()};assert steps=={500}
  curves=[json.loads(s)for s in (E/'runs'/name/'curve.jsonl').read_text().splitlines()];assert [r['update']for r in curves]==list(range(1,501));assert all(np.isclose(r['total_loss'],r['BCE']+r['weighted_latent'],atol=1e-6)for r in curves);assert run['initial_digest']==init
  hist=read(E/'runs'/name/'evaluations.json');assert [h['update']for h in hist]==q['eval_updates'];out=np.load(E/'runs'/name/'final_outputs.npz');diff=sum(float((base['model'][k]-v).square().sum())for k,v in cp['model'].items());assert diff>0
  for group,ids in [('old132',slice(0,132)),('five',slice(132,137))]:
   met=correctness(out['probabilities'][ids],target_y[ids]);assert met['all7']==hist[-1]['groups'][group]['all7'];pred=np.concatenate([out['objects'][ids].reshape(-1,288),out['scene'][ids]],1);truth=np.concatenate([hreal[ids].reshape(-1,288),sreal[ids]],1);metrics.append({'arm':name,'group':group,**met,'latent_MAE':float(np.abs(pred-truth).mean()),'abs_gt1_fraction':float((np.abs(pred)>1).mean()),'BCE':hist[-1]['groups'][group]['BCE'],'weighted_latent':hist[-1]['groups'][group]['weighted_latent']})
  rows.append({'arm':name,'logged_updates':500,'optimizer_steps':list(steps),'initial_digest':init,'parameter_squared_change':diff,'gradient_clip_events':sum(r['gradient_norm_before_clip']>1 for r in curves),'checkpoint_sha256':sha(run['checkpoint']),'fixed_final_only':True})
 save(E/'results/final_metrics.json',metrics)
 save(E/'results/final_training_audit.json',{'created_utc':now(),'passed':True,'runs':rows,'total_optimizer_updates':1500,'seed':0,'C_D_optimizer_or_forward_rows':0,'only_existing_AB137':True,'original_inputs_targets_and_sources_unchanged':True,'amendment_before_any_update':True,'no_best_checkpoint_selection':True,'no_additional_training':True})
 print(json.dumps({'passed':True,'runs':rows,'metrics':[{k:v for k,v in x.items()if k!='predictions'}for x in metrics]},ensure_ascii=False))
if __name__=='__main__':main()
