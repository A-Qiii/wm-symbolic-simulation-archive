from config import *
from losses import *
import fcntl,time,argparse
from runtime import load,tensor_hash
from model import Query
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=['mixed','latent'],required=True);ap.add_argument('--seed',type=int,choices=[0,1],required=True);args=ap.parse_args()
 frozen=read(E/'configs/training_frozen.json')
 for p,h in frozen['sha256'].items():assert sha(p)==h,p
 protocol=read(E/'configs/protocol.json');qsel=protocol['query_heads'][str(args.seed)];chains=read(E/'configs/chains.json');alpha=frozen['alpha'];sc=dict(np.load(E/'configs'/f'schedule_seed{args.seed}.npz'))
 allowed={str(p)for p in[PACK/'inputs.npz',PACK/'pairs.json',Path(qsel['path']),REP/'data/public/normalization.npz',REP/'models/canonical_matched_seed0/selected.json',REP/'models/canonical_matched_seed0/update1000.pt',REP/'models/query_seed0.pt']}
 def guard(event,a):
  if event=='open'and isinstance(a[0],(str,bytes)):
   p=Path(a[0]).resolve()
   if p.is_relative_to(P/'experiments')and p.suffix in['.json','.npz','.pt']and str(p)not in allowed and not p.is_relative_to(E/'models')and not p.is_relative_to(E/'results'):raise RuntimeError('Trainer forbidden input '+str(p))
 sys.addaudithook(guard);lock=(P/'runs/gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(4);torch.manual_seed(args.seed);torch.backends.cuda.matmul.allow_tf32=True
 rows,data=load_data();o,s,z,b,a,act=data;m,_,_=load(0,'canonical_matched');q=Query().cuda();q.load_state_dict(torch.load(qsel['path'],weights_only=True,map_location='cuda')['state']);q.eval().requires_grad_(False);m.transition.requires_grad_(True);params=list(m.transition.parameters());init=tensor_hash(m.state_dict());fhash=lambda:tensor_hash({k:v for k,v in m.state_dict().items()if not k.startswith('transition.')});fixed=fhash();qh=tensor_hash(q.state_dict());opt=torch.optim.AdamW(params,lr=1e-4,weight_decay=1e-4);name=f'{args.arm}_seed{args.seed}';folder=E/'models'/name;folder.mkdir();logs=[];batches=[];best=float('inf');chosen=None;started=time.monotonic()
 seqs={'new_AB':list(chains['AB'].values()),'old_AB':list(chains['old_AB'].values()),'C_select':list(chains['C_selection'].values())};seqs={k:torch.tensor(v,device='cuda')for k,v in seqs.items()};dev=[i for i,r in enumerate(rows)if r['split']in['dev','dev_select']];counts=np.bincount(sc['single'].ravel(),minlength=len(rows));trainix=np.flatnonzero(counts);freq=torch.tensor(counts[trainix]/counts.sum(),device='cuda',dtype=torch.float32)
 def checkpoint(update):
  nonlocal best,chosen
  diag={}
  with torch.no_grad():
   for group,seq in seqs.items():
    cur=z[b[seq[:,0]]];fe=[];ze=[]
    for depth in range(4):
     ix=seq[:,depth];cur=m.transition(cur,act[ix]);fe.append(float(feature_error(m.decode(cur),o[a[ix]],s[a[ix]],(o[b[ix]],s[b[ix]])).mean()));ze.append(float((cur-z[a[ix]]).square().mean()))
    diag[group]={'roots':len(seq),'recursive_feature':fe,'recursive_rawZ':ze,'common_multistep_F':.25*fe[1]+.5*fe[2]+.25*fe[3],'common_multistep_Z':.25*ze[1]+.5*ze[2]+.25*ze[3]}
   pr=m.transition(z[b[dev]],act[dev]);fe=feature_error(m.decode(pr),o[a[dev]],s[a[dev]],(o[b[dev]],s[b[dev]])).mean(1).cpu().numpy();ze=(pr-z[a[dev]]).square().mean((1,2)).cpu().numpy();dv={root:[float(fe[[j for j,i in enumerate(dev)if rows[i]['root_id']==root]].mean()),float(ze[[j for j,i in enumerate(dev)if rows[i]['root_id']==root]].mean())]for root in sorted({rows[i]['root_id']for i in dev})};old=[v for k,v in dv.items()if not k.startswith('C_intermediate_v3_')];new=[v for k,v in dv.items()if k.startswith('C_intermediate_v3_')];assert len(old)==37 and len(new)==6;single=(np.mean(old,0)+np.mean(new,0))*.5
   pr=m.transition(z[b[trainix]],act[trainix]);tf=float((feature_error(m.decode(pr),o[a[trainix]],s[a[trainix]],(o[b[trainix]],s[b[trainix]])).mean(1)*freq).sum());tz=float(((pr-z[a[trainix]]).square().mean((1,2))*freq).sum())
  score=diag['C_select']['common_multistep_Z'];assert fhash()==fixed and tensor_hash(q.state_dict())==qh;path=folder/f'update{update:04d}.pt';torch.save({'state':m.state_dict(),'update':update,'C_score':score,'frozen_hash':fixed},path);logs.append({'update':update,'score':score,'same_metric_AB_C':diag,'C_single_feature':float(single[0]),'C_single_rawZ':float(single[1]),'AB_single_scheduleweighted_feature':tf,'AB_single_scheduleweighted_rawZ':tz});print(name,update,'CmultiZ',round(score,6),'CmultiF',round(diag['C_select']['common_multistep_F'],6),flush=True)
  if score<best:best=score;chosen=path
 checkpoint(0)
 for u in range(1000):
  terms=batch_terms(m,data,sc,u,need_feature=args.arm=='mixed');loss=mixed_loss(terms)if args.arm=='mixed'else alpha*rawZ(terms);opt.zero_grad();assert torch.isfinite(loss);loss.backward();assert all(p.grad is None for n,p in m.named_parameters()if not n.startswith('transition.'));assert all(p.grad is None for p in q.parameters());gn=float(torch.nn.utils.clip_grad_norm_(params,5));opt.step();batches.append({'update':u+1,'rawF_weighted':sum(w*float(f.detach())for w,f,zz in terms),'rawZ_weighted':sum(w*float(zz.detach())for w,f,zz in terms),'scaled_loss':float(loss.detach()),'gradient_norm_before_clip':gn,'contains_multi':len(terms)>1,'pre_update_batch':True})
  if(u+1)%250==0:checkpoint(u+1)
 torch.cuda.synchronize();save(folder/'selected.json',{'path':str(chosen),'sha256':sha(chosen),'C_score':best});save(E/'results'/f'train_{name}.json',{'updates':1000,'single_draws':64000,'multi_starts':12000,'transition_state_evaluations':100000,'alpha':alpha,'initial_model_hash':init,'frozen_encoder_decoder_hash':fixed,'final_frozen_hash':fhash(),'query_hash':qh,'final_query_hash':tensor_hash(q.state_dict()),'query_path':qsel['path'],'log':logs,'batch_log':batches,'only_transition_optimized':True,'validation_or_test_access':False,'elapsed_seconds':time.monotonic()-started});print('TRAIN_DONE',name,flush=True)
if __name__=='__main__':main()
