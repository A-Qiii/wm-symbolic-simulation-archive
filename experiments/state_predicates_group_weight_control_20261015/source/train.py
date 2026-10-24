from common_control import *
import torch,fcntl,time,platform,os
from torch.nn import functional as F
def main():
 q=frozen();assert not(E/'configs/model_selection_frozen.json').exists();schedule=np.load(q['schedule'])['global_indices'];assert schedule.shape==(500,137)and np.all(schedule==np.array(q['train_global_ids'])[None]);ids=schedule[0];z=np.load(PREV/'data/training/public/inputs.npz');t=np.load(PREV/'data/training/_private/targets.npz');pub=read(PREV/'data/training/public/pairs.json');assert all(pub[i]['split']=='train'for i in ids)
 lock=(P/'runs/gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(4);torch.set_float32_matmul_precision('high');torch.backends.cudnn.benchmark=False
 reader,logits,digest,balanced,Net=old_modules();m=reader();rd=digest(m);w=torch.tensor(q['class_weights'],device='cuda');d={k:torch.as_tensor(z[k][ids],device='cuda')for k in ['current_objects','current_scene','declared_actions']};d.update({k:torch.as_tensor(t[k][ids],device='cuda')for k in ['next_objects','next_scene','after_y','after_valid']});assert d['after_valid'].all()
 save(E/'configs/runtime.json',{'created_utc':now(),'python':platform.python_version(),'torch':torch.__version__,'cuda':torch.version.cuda,'GPU':torch.cuda.get_device_name(),'pid':os.getpid(),'serial_lock':True})
 def part(h,s,ll,ii):
  b=balanced(ll[ii],d['after_y'][ii],d['after_valid'][ii],w);l=.025*(F.smooth_l1_loss(h[ii],d['next_objects'][ii])+F.smooth_l1_loss(s[ii],d['next_scene'][ii]));return b,l
 def observe(net,step):
  with torch.no_grad():
   h,s=net(d['current_objects'],d['current_scene'],d['declared_actions']);ll=logits(m,h,s);p=ll.sigmoid().cpu().numpy();y=d['after_y'].cpu().numpy();good=(known(p)&((p>=.5)==y)).all(1);out={'update':step,'old132_all7':int(good[:132].sum()),'new5_all7':int(good[132:].sum()),'old132_BCE':float(part(h,s,ll,slice(0,132))[0]),'new5_BCE':float(part(h,s,ll,slice(132,137))[0]),'new5_probabilities':p[132:].tolist()};return out
 results=[];equivalence=[]
 for seed in q['seeds']:
  cp=q['starts'][str(seed)];base=torch.load(cp['path'],map_location='cuda',weights_only=False);assert sha(cp['path'])==cp['sha256'];initial=None
  for arm in q['arms']:
   rid=f'{arm}_seed{seed}';run=E/'runs'/rid;run.mkdir();torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);net=Net('full').cuda();net.load_state_dict(base['model']);di=digest(net)
   if initial is None:initial=di
   assert initial==di;opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.0001);assert len(opt.state)==0;weights=q['arm_weights'][arm];wo,wn=weights['old132'],weights['new5'];history=[observe(net,0)];t0=time.perf_counter();log=(run/'curve.jsonl').open('w')
   # Check original loss equivalence before first optimizer step, using the same group implementation in both arms.
   h,s=net(d['current_objects'],d['current_scene'],d['declared_actions']);ll=logits(m,h,s);b0,l0=part(h,s,ll,slice(0,132));b1,l1=part(h,s,ll,slice(132,137));bb,llat=part(h,s,ll,slice(None));err=float(((132*(b0+l0)+5*(b1+l1))/137-(bb+llat)).detach().abs());assert err<1e-6;equivalence.append({'run':rid,'uniform_group_vs_original_whole_batch_loss_abs_error':err})
   for step,ix in enumerate(schedule,1):
    assert np.array_equal(ix,ids);opt.zero_grad(set_to_none=True);h,s=net(d['current_objects'],d['current_scene'],d['declared_actions']);ll=logits(m,h,s);b0,l0=part(h,s,ll,slice(0,132));b1,l1=part(h,s,ll,slice(132,137));b=wo*b0+wn*b1;l=wo*l0+wn*l1;loss=b+l;assert torch.isfinite(loss);loss.backward();gn=torch.nn.utils.clip_grad_norm_(net.parameters(),1);assert torch.isfinite(gn);opt.step();log.write(json.dumps({'update':step,'old132_BCE':float(b0.detach()),'new5_BCE':float(b1.detach()),'old132_weighted_latent':float(l0.detach()),'new5_weighted_latent':float(l1.detach()),'weighted_BCE':float(b.detach()),'weighted_latent':float(l.detach()),'loss':float(loss.detach()),'gradient_norm_before_clip':float(gn),'seconds':time.perf_counter()-t0})+'\n')
    if step in q['selection']['training_observation_steps']:
     o=observe(net,step);history.append(o);save(run/'training_observations.json',history);print(json.dumps({'run':rid,**{k:v for k,v in o.items()if k!='new5_probabilities'}}),flush=True)
   log.close();assert digest(m)==rd and all(p.grad is None for p in m.parameters());fp=E/'models'/f'{rid}_final500.pt';torch.save({'model':net.state_dict(),'optimizer':opt.state_dict(),'seed':seed,'arm':arm,'update':500,'start_sha256':cp['sha256'],'protocol_sha256':sha(E/'configs/protocol.json'),'schedule_sha256':q['schedule_sha256'],'group_weights':weights},fp)
   r={'run':rid,'seed':seed,'arm':arm,'updates':500,'prior_weight_updates':1500,'trajectory_weight_updates':2000,'initial_digest':initial,'start_sha256':cp['sha256'],'fresh_optimizer_empty_at_start':True,'schedule_sha256':q['schedule_sha256'],'examples_per_update':137,'presentations':68500,'presentations_per_example':500,'reader_digest':rd,'C_optimizer_and_selection_rows':0,'final500':{'path':str(fp),'sha256':sha(fp)},'seconds':time.perf_counter()-t0};save(run/'summary.json',r);results.append(r);save(E/'results/training_progress.json',{'completed_runs':len(results),'optimizer_updates':500*len(results)});del net,opt;torch.cuda.empty_cache()
 assert len(results)==6;save(E/'results/loss_equivalence_check.json',equivalence);sel={'created_utc':now(),'all_training_complete':True,'total_optimizer_updates':3000,'runs':results,'reader_digest':rd,'fixed_final500_only':True,'representatives':['uniform_seed0','equal_groups_seed0'],'C_used_for_selection':False};save(E/'configs/model_selection_frozen.json',sel);(E/'configs/model_selection_frozen.json').chmod(0o444);print('TRAINING_3000_COMPLETED',flush=True)
if __name__=='__main__':main()
