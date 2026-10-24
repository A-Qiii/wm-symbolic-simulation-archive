from config import *
import torch,sys,fcntl,time
sys.path.insert(0,str(REP/'source'))
from runtime import load,tensor_hash
from model import Query,feature_error

def main():
 protocol=read(E/'configs/protocol.json')
 for p,h in read(E/'configs/frozen.json')['sha256'].items():assert sha(p)==h,p
 def guard(event,args):
  if event=='open'and isinstance(args[0],(str,bytes))and '/_private/'in str(args[0]):raise RuntimeError('Latent diagnostic cannot read GT')
 sys.addaudithook(guard);lock=(P/'runs/gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(4);torch.manual_seed(0);start=time.monotonic()
 data=dict(np.load(DIAG/'data/public/inputs.npz'));z=torch.tensor(data['z'],device='cuda');o=torch.tensor(data['objects'],device='cuda');s=torch.tensor(data['scene'],device='cuda');samples=read(E/'configs/samples.json');b=torch.tensor([r['before']for r in samples],device='cuda');a=torch.tensor([r['after']for r in samples],device='cuda');act=torch.tensor([r['action']for r in samples],device='cuda');target=z[a];m,_,stats=load(0,'canonical_matched');fixed=tensor_hash({k:v for k,v in m.state_dict().items()if not k.startswith('transition.')});N=len(samples)
 cases=[];hist=[];grads=[];positions=[];cpmeta=[];cparr=[];qhash={}
 def logits(u,q):return q((u-stats['mean'])/stats['std'])
 def terms(u):
  f=feature_error(m.decode(u),o[a],s[a],(o[b],s[b]));zz=(u-target).square().mean((1,2));return f,zz
 def preds():
  with torch.no_grad():
   one=m.transition(z[b],act);cur=z[b[::4]];out=[]
   for j in range(4):cur=m.transition(cur,act[j::4]);out.append(cur)
   return one,torch.stack(out,dim=1).reshape(N,4,64)
 def probe(seed,name,origin,initial,q):
  u=initial.detach().clone().requires_grad_(True);f,zz=terms(u);gf=torch.autograd.grad(f.mean(1).sum(),u,retain_graph=True)[0];gz=.5*(u-target)/256
  gg=torch.stack([torch.autograd.grad((f[:,j]/4).sum(),u,retain_graph=True)[0]for j in range(4)],1)
  _,direction=torch.autograd.functional.jvp(lambda v:logits(v,q),u,-gf.detach(),create_graph=False)
  grads.append({'gF':gf.detach().cpu().numpy(),'g025Z':gz.detach().cpu().numpy(),'gF_groups':gg.detach().cpu().numpy(),'query_logit_direction_minus_gF':direction.detach().cpu().numpy()})
  hx=[];ends=[]
  for objective in ['mixed','latent_only']:
   u=initial.detach().clone();ss=[]
   for it in range(protocol['optimization']['steps']+1):
    u.requires_grad_(True);ff,zerr=terms(u);loss=ff.mean(1)+.25*zerr if objective=='mixed'else .25*zerr
    assert torch.isfinite(loss).all()and torch.isfinite(u).all(),(name,origin,objective,it)
    if it in protocol['optimization']['snapshots']:
     with torch.no_grad():
      ss.append(torch.cat([ff,zerr[:,None],logits(u,q),(u-initial).square().mean((1,2))[:,None],((u-target)/stats['std']).square().mean((1,2))[:,None]],1).cpu().numpy())
    if it<protocol['optimization']['steps']:
     g=torch.autograd.grad(loss.sum(),u)[0];u=(u-protocol['optimization']['learning_rate']*g).detach()
   hx.append(np.stack(ss,1));ends.append(u.detach().cpu().numpy())
  hist.append(np.stack(hx,1));positions.append({'initial':initial.detach().cpu().numpy(),'target':target.cpu().numpy(),'end':np.stack(ends,1)})
  cases.extend({'case_id':len(cases)+i,'sample_id':i,'seed':seed,'model':name,'origin':origin}for i in range(N))
  assert all(p.grad is None and not p.requires_grad for p in m.parameters())and all(p.grad is None and not p.requires_grad for p in q.parameters())
  print('LATENT_PROBE',name,origin,N,flush=True)
 for seed in [0,1]:
  q=Query().cuda();q.load_state_dict(torch.load(protocol['query_heads'][str(seed)]['path'],weights_only=True,map_location='cuda')['state']);q.eval().requires_grad_(False);qhash[seed]=tensor_hash(q.state_dict());probe(seed,f'actual_seed{seed}','actual',target,q)
  for arm in ['span3','span4']:
   name=f'{arm}_seed{seed}';m.load_state_dict(torch.load(protocol['selected_models'][name]['path'],weights_only=True,map_location='cuda')['state']);assert tensor_hash({k:v for k,v in m.state_dict().items()if not k.startswith('transition.')})==fixed
   one,rec=preds();probe(seed,name,'one_step',one,q);probe(seed,name,'recursive',rec,q)
   for update in [0,250,500,750,1000]:
    cp=SPAN/'models'/name/f'update{update:04d}.pt';m.load_state_dict(torch.load(cp,weights_only=True,map_location='cuda')['state']);one,rec=preds()
    for origin,u in [('one_step',one),('recursive',rec)]:
     with torch.no_grad():ff,zz=terms(u);cparr.append(torch.cat([ff,zz[:,None],logits(u,q)],1).cpu().numpy())
     cpmeta.extend({'record_id':len(cpmeta)+i,'sample_id':i,'seed':seed,'model':name,'origin':origin,'update':update}for i in range(N))
    print('FROZEN_CHECKPOINT',name,update,flush=True)
  assert tensor_hash(q.state_dict())==qhash[seed]
 assert len(cases)==2400 and len(cpmeta)==9600
 # Extending a generator changes len during iteration; assign canonical IDs once.
 for i,r in enumerate(cases):r['case_id']=i
 for i,r in enumerate(cpmeta):r['record_id']=i
 np.savez_compressed(E/'data/latent_probes.npz',history=np.concatenate(hist),**{k:np.concatenate([g[k]for g in grads])for k in grads[0]},**{k:np.concatenate([p[k]for p in positions])for k in positions[0]})
 np.savez_compressed(E/'data/checkpoints.npz',values=np.concatenate(cparr))
 save(E/'data/cases.json',cases);save(E/'data/checkpoint_cases.json',cpmeta)
 norm=dict(np.load(REP/'data/public/normalization.npz'));desc=lambda x:{'min':float(np.min(x)),'median':float(np.median(x)),'max':float(np.max(x))}
 scale={'feature_dimensions':[3*256,3*12,1024,15],'feature_group_coefficients':[.25]*4,'target_change_weights_range':[1,5],'latent_dimensions':256,'latent_MSE_coefficient':.25,'per_latent_squared_error_coefficient':.25/256,'latent_raw_RMS':float(target.square().mean().sqrt()),'latent_per_coordinate_std':desc(target.std(0).cpu().numpy()),'query_normalization_std':desc(stats['std'].cpu().numpy()),'query_normalization_mean':desc(stats['mean'].cpu().numpy()),'feature_normalization_std':{k:desc(norm[k])for k in ['vision_std','pca_scale','geometry_std','robot_std']},'query_norm_used_in_loss':False,'feature_standardization_fit':'priorABonly;frozen','loss_normalization':'F already uses feature preprocessing, per-group means; Z is raw latent and is not query-standardized','latent_only_analytic_shrink_per_step':1-4*.5/256,'latent_only_analytic_distance_ratio_step32':(1-4*.5/256)**32}
 save(E/'results/scale_audit.json',scale)
 torch.cuda.synchronize()
 save(E/'results/run_audit.json',{'finished_utc':now(),'network_updates':0,'simulation_calls':0,'private_truth_reads':0,'latent_optimizer_steps':32,'cases':len(cases),'checkpoint_records':len(cpmeta),'encoder_decoder_hash':fixed,'query_hashes':qhash,'all_model_parameters_frozen':True,'seconds':time.monotonic()-start,'history_columns':['F_obj_visual','F_obj_geometry','F_scene_visual','F_robot','Z',*FACTS,'MSE_from_start','query_normalized_MSE_from_target']})
 print('DIAGNOSTIC_DONE',len(cases),len(cpmeta),flush=True)
if __name__=='__main__':main()
