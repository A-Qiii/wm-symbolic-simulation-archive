from common_diag import *
import torch,fcntl,time,os,platform
from torch.nn import functional as F

def main():
 q,d,_,_=data();assert read(E/'results/audit_complete.json')['training_can_start'];assert not(E/'results/training_complete.json').exists()
 lock=(P/'runs/gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(4);torch.set_float32_matmul_precision('high');torch.backends.cudnn.benchmark=False
 reader,logits,digest,balanced,Net=old_modules();m=reader();rd=digest(m);w=torch.tensor(q['class_weights'],device='cuda');base=torch.load(q['checkpoint'],map_location='cuda',weights_only=False)['model'];summaries=[];initial=None
 save(E/'configs/runtime.json',{'created_utc':now(),'torch':torch.__version__,'python':platform.python_version(),'cuda':torch.version.cuda,'device':torch.cuda.get_device_name(),'pid':os.getpid(),'serial_gpu_lock':True})
 def loss_parts(h,s,ll,ids):
  bce=balanced(ll[ids],d['after_y'][ids],d['after_valid'][ids],w);lat=.025*(F.smooth_l1_loss(h[ids],d['next_objects'][ids])+F.smooth_l1_loss(s[ids],d['next_scene'][ids]));return bce,lat
 def evaluate(net,update):
  with torch.no_grad():
   h,s=net(d['current_objects'],d['current_scene'],d['declared_actions']);ll=logits(m,h,s);p=ll.sigmoid().cpu().numpy();out={'update':update,'groups':{}}
   for name,ids in [('old132',slice(0,132)),('five',slice(132,137))]:
    b,l=loss_parts(h,s,ll,ids);out['groups'][name]=correctness(p[ids],d['after_y'][ids].cpu().numpy())|{'BCE':float(b),'weighted_latent':float(l),'total_loss':float(b+l),'probabilities':p[ids].tolist()}
   return out,h.cpu().numpy(),s.cpu().numpy(),p
 for arm in q['arms']:
  name=arm['name'];run=E/'runs'/name;run.mkdir();torch.manual_seed(0);torch.cuda.manual_seed_all(0);net=Net('full').cuda();net.load_state_dict(base);startdigest=digest(net)
  if initial is None:initial=startdigest
  assert initial==startdigest;opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.0001);history=[];ev,_,_,_=evaluate(net,0);history.append(ev);t0=time.perf_counter();log=(run/'curve.jsonl').open('w')
  for step in range(1,501):
   net.train();opt.zero_grad(set_to_none=True);h,s=net(d['current_objects'],d['current_scene'],d['declared_actions']);ll=logits(m,h,s)
   if name=='five_only':b,l=loss_parts(h,s,ll,slice(132,137))
   elif name=='joint_original':b,l=loss_parts(h,s,ll,slice(None))
   else:
    b0,l0=loss_parts(h,s,ll,slice(0,132));b1,l1=loss_parts(h,s,ll,slice(132,137));b=.5*(b0+b1);l=.5*(l0+l1)
   loss=b+l;assert torch.isfinite(loss);loss.backward();gn=torch.nn.utils.clip_grad_norm_(net.parameters(),1);assert torch.isfinite(gn);opt.step();log.write(json.dumps({'update':step,'BCE':float(b.detach()),'weighted_latent':float(l.detach()),'total_loss':float(loss.detach()),'gradient_norm_before_clip':float(gn),'seconds':time.perf_counter()-t0})+'\n')
   if step in q['eval_updates']:
    ev,_,_,_=evaluate(net,step);history.append(ev);save(run/'evaluations.json',history);print(json.dumps({'arm':name,'update':step,'five_all7':ev['groups']['five']['all7'],'old132_all7':ev['groups']['old132']['all7'],'five_BCE':ev['groups']['five']['BCE']},ensure_ascii=False),flush=True)
  log.close();ev,h,s,p=evaluate(net,500);assert digest(m)==rd and all(p.grad is None for p in m.parameters());fp=E/'models'/f'{name}_diagnostic_only_final500.pt';torch.save({'model':net.state_dict(),'optimizer':opt.state_dict(),'update':500,'seed':0,'arm':name,'diagnostic_only':True,'not_candidate':True,'protocol_sha256':sha(E/'configs/protocol.json')},fp);np.savez_compressed(run/'final_outputs.npz',objects=h,scene=s,probabilities=p)
  out={'arm':name,'updates':500,'seed':0,'initial_digest':initial,'start_checkpoint_sha256':q['checkpoint_sha256'],'checkpoint':str(fp),'checkpoint_sha256':sha(fp),'final_five':ev['groups']['five']['all7'],'final_old132':ev['groups']['old132']['all7'],'five_presentations_each':500,'old132_presentations_each':0 if name=='five_only'else500,'effective_gradient_rows_each_update':5 if name=='five_only'else137,'not_candidate':True,'all_updates_logged':True,'seconds':time.perf_counter()-t0};save(run/'summary.json',out);summaries.append(out);save(E/'results/training_progress.json',{'completed_arms':len(summaries),'updates':500*len(summaries)});del net,opt;torch.cuda.empty_cache()
 assert len(summaries)==3;save(E/'results/training_complete.json',{'created_utc':now(),'passed':True,'runs':summaries,'total_updates':1500,'seed':0,'C_D_forward_calls':0,'new_collection':0,'best_checkpoint_selection':False,'no_adaptive_extension':True,'diagnostic_only':True});print('DIAGNOSTIC_TRAINING_COMPLETED',flush=True)
if __name__=='__main__':main()
