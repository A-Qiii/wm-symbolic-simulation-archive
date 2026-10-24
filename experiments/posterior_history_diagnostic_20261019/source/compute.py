from common import *
import torch,ast,fcntl,signal
from wm_symbolic.world_model import WorldModel
from wm_symbolic.dataset import Episodes,static_arguments
from wm_symbolic.vision_cache import FrozenVision,VisibleRegistry
EXTRA=[];RNG_AUDIT=[];RUN_SEED=None;RUN_SPLIT=None
def extract(path,names,ns):
 tree=ast.parse(path.read_text());nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))and n.name in names];assert len(nodes)==len(names);exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns)
def state_hash(m):
 h=hashlib.sha256()
 for k,v in sorted(m.state_dict().items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def rng_hash(x):return hashlib.sha256(x.cpu().numpy().tobytes()).hexdigest()
def history_rng(m,enc,actions):
 rng=[torch.cuda.get_rng_state()];root,_=m.initial(enc[:,0]);posts=[root];priors=[]
 for t in range(actions.shape[1]):
  prior,_=m.imagine(posts[-1],actions[:,t]);priors.append(prior);rng.append(torch.cuda.get_rng_state());post,_=m.observe(prior,enc[:,t+1]);posts.append(post)
 return posts,rng,priors
def extra(m,data,enc,actions,posts,n,sample,ids,rngs,cp_rngs,free_rngs,priors):
 global EXTRA
 before=torch.cuda.get_rng_state();resets=[]
 with torch.random.fork_rng(devices=[0]),torch.autocast('cuda',dtype=torch.bfloat16):
  for t in range(5):
   torch.cuda.set_rng_state(rngs[t]);initial,_=m.initial(enc[:,t].repeat_interleave(sample,0));resets.append(initial);RNG_AUDIT.append({'seed':RUN_SEED,'split':RUN_SPLIT,'batch_first':ids[0],'boundary':t,'posterior_rng_sha256':rng_hash(rngs[t]),'same_observation_feature':True,'same_random_numbers':True})
  assert torch.equal(posts[0].h,resets[0].h)and torch.equal(posts[0].z,resets[0].z)
  def append(condition,mode,origin,target,state,event=None):
   prob=PROB(m,state,n,sample)
   for j,ei in enumerate(ids):
    ep=data.episodes[ei];labels=ep['labels'];lab=labels[0]['before']if target==0 else labels[target-1]['after'];statuses=['success'if a['success']else'failed'if a['control_samples_including_start']>1 else'refused'for a in labels];EXTRA.append({'seed':RUN_SEED,'split':RUN_SPLIT,'root_id':ep['root_id'],'episode_id':ep['episode_id'],'condition':condition,'mode':mode,'origin':origin,'target':target,'horizon':target-origin,'prob':prob[j],'cat':lab['categorical'],'cat_valid':lab['categorical_valid'],'target_previous_status':statuses[target-1]if target else'root','origin_previous_status':statuses[origin-1]if origin else'root','after_first_failure':any(s=='failed'for s in statuses[:origin]),'success_prob':float(event[j])if event is not None else np.nan})
  for t in range(1,5):append('prior','prior_current',t,t,priors[t-1])
  for name,states in [('continuous',posts),('reset',resets)]:
   for t in range(5):append(name,'current',t,t,states[t])
   for t in range(4):
    ca=actions[:,t].repeat_interleave(sample,0);torch.cuda.set_rng_state(cp_rngs[t]);after,_=m.imagine(states[t],ca);event=m.outcome_logits(states[t],ca,after).float().sigmoid().reshape(n,sample,4).mean(1).cpu().numpy()[:,0];append(name,'one_step',t,t+1,after,event)
   for origin in range(4):
    state=states[origin]
    for target in range(origin+1,5):
     if origin==0:torch.cuda.set_rng_state(free_rngs[target-1])
     else:torch.manual_seed(91831+100000*RUN_SEED+10000*SPLITS.index(RUN_SPLIT)+100*ids[0]+10*origin+target)
     state,_=m.imagine(state,actions[:,target-1].repeat_interleave(sample,0));append(name,'free_root'if origin==0 else'free_suffix',origin,target,state)
 assert torch.equal(torch.cuda.get_rng_state(),before),'Extra comparison changed original RNG stream'
def pixels_and_E():
 torch.use_deterministic_algorithms(False);torch.set_float32_matmul_precision('highest');idx=read(R/'data/features/index.json')['items'];items=sorted([x for x in idx if x['episode_id']=='B_coverage_gap_04_b0'],key=lambda x:x['step']);encoder=FrozenVision(P,reserve_gpu=False);reg=VisibleRegistry();pixel=[]
 for item in items:
  new=encoder.encode(item['public_inputs'],reg);old=dict(np.load(item['path']));assert set(new)==set(old)
  for k in old:assert np.array_equal(new[k],old[k]),('fresh image feature mismatch',item['step'],k)
  pixel.append({'episode_id':item['episode_id'],'boundary':item['step'],'all_cache_arrays_exact':True,'keys':list(old),'source_frames':item['public_inputs']})
 del encoder;torch.cuda.empty_cache();torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('high');ns={'P':P,'S':R,'B':B,'read':read,'Episodes':Episodes,'Path':Path};extract(R/'source/compute.py',['Data'],ns);data=ns['Data']('gap');ei=next(i for i,e in enumerate(data.episodes)if e['episode_id']=='B_coverage_gap_04_b0');ids=list(range(ei//4*4,min(ei//4*4+4,len(data.episodes))));spatial=[]
 for seed in [0,1]:
  cp=B/f'cells/roots40/models/joint_seed{seed}/final6000.pt';m=WorldModel(skill_count=6,event_count=3).cuda();m.load_state_dict(torch.load(cp,map_location='cuda',weights_only=True)['model']);m.eval().requires_grad_(False);before=state_hash(m);batch=data.batch(ids,'cuda');ref=np.load(R/f'data/features_seed{seed}_gap.npz');lookup={(str(e),int(t)):j for j,(e,t)in enumerate(zip(ref['episode_id'],ref['step']))}
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):enc=m.encode(batch['tokens'],batch['geometry'],batch['visible'],batch['robot'])
  actual=enc[:,:5].float().cpu().numpy();expected=np.array([[ref['global'][lookup[data.episodes[i]['episode_id'],t]]for t in range(5)]for i in ids]);assert np.array_equal(actual,expected);assert state_hash(m)==before;spatial.append({'seed':seed,'episodes':[data.episodes[i]['episode_id']for i in ids],'boundaries_exact':len(ids)*5,'max_difference':float(abs(actual-expected).max())});del m,batch;torch.cuda.empty_cache()
 save(S/'results/fresh_pixel_alignment.json',{'utc':now(),'passed':True,'fresh_DINO_windows':len(pixel),'fresh_spatial_batches':2,'pixel_to_cache':pixel,'cache_to_global_E':spatial,'no_training':True})
def main():
 global RUN_SEED,RUN_SPLIT,PROB
 sys.addaudithook(guard);assert not(S/'results/compute_complete.json').exists();lock=(P/'runs/gpu.lock').open('r');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);signal.signal(signal.SIGALRM,lambda *_:(_ for _ in()).throw(TimeoutError('600 second frozen budget')));signal.alarm(600);begin=time.monotonic();verify_inputs();torch.set_num_threads(4);pixels_and_E();ns=globals().copy();ns['E']=S;source=(H/'source/evaluate_with_h2.py').read_text();source=source.replace("enc=m.encode(b['tokens'],b['geometry'],b['visible'],b['robot'])","enc=data.cached_encoded(ids,'cuda')");source=source.replace("posts,_,_,_=m.history(enc.repeat_interleave(sample,0),actions.repeat_interleave(sample,0));free=posts[0]","posts,rngs,priors=history_rng(m,enc.repeat_interleave(sample,0),actions.repeat_interleave(sample,0));free=posts[0];cp_rngs=[];free_rngs=[]");source=source.replace("rng=torch.cuda.get_rng_state();wp,_=m.imagine(posts[t],wa)","rng=torch.cuda.get_rng_state();cp_rngs.append(rng);wp,_=m.imagine(posts[t],wa)");source=source.replace("free,_=m.imagine(free,ca)","free_rngs.append(torch.cuda.get_rng_state());free,_=m.imagine(free,ca)");needle="   if off%40==0:";assert source.count(needle)==1;source=source.replace(needle,"   extra(m,data,enc,actions,posts,n,sample,ids,rngs,cp_rngs,free_rngs,priors)\n"+needle);tree=ast.parse(source);nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name in ['probabilities','savearrays','assess']];exec(compile(ast.Module(body=nodes,type_ignores=[]),'<paired evaluator>','exec'),ns);(S/'source/paired_evaluator.py').write_text(ast.unparse(ast.Module(body=nodes,type_ignores=[]))+'\n');PROB=ns['probabilities'];data_ns=globals().copy();extract(G/'source/compute.py',['CachedData'],data_ns);Data=data_ns['CachedData'];parity=[];weights={}
 for seed in [0,1]:
  RUN_SEED=seed;cp=B/f'cells/roots40/models/joint_seed{seed}/final6000.pt';m=WorldModel(skill_count=6,event_count=3).cuda();m.load_state_dict(torch.load(cp,map_location='cuda',weights_only=True)['model']);m.eval().requires_grad_(False);before=state_hash(m)
  for split in SPLITS:
   RUN_SPLIT=split;ns['assess'](m,Data(split,seed),'joint',seed);a=np.load(S/f'data/predictions_joint_seed{seed}_{split}.npz');b=np.load(G/f'data/predictions_joint_seed{seed}_{split}.npz');checks={k:np.array_equal(a[k],b[k],equal_nan=True)if b[k].dtype.kind in 'fc'else np.array_equal(a[k],b[k])for k in b.files};assert all(checks.values()),checks;parity.append({'seed':seed,'split':split,'rows':len(a['prob']),'all_arrays_exact':True,'checks':checks,'max_probability_difference':float(abs(a['prob']-b['prob']).max())})
  assert state_hash(m)==before;weights[str(cp)]=sha(cp);del m;torch.cuda.empty_cache()
 np.savez_compressed(S/'data/paired_predictions.npz',**{k:np.array([r[k]for r in EXTRA])for k in EXTRA[0]});save(S/'results/rng_pairs.json',RNG_AUDIT);save(S/'results/parity.json',parity);verify_inputs();save(S/'results/compute_complete.json',{'utc':now(),'seconds':time.monotonic()-begin,'peak_cuda_GiB':torch.cuda.max_memory_allocated()/1024**3,'prediction_rows':len(EXTRA),'posterior_rng_pairs':len(RNG_AUDIT),'original_rows_exact':sum(r['rows']for r in parity),'model_weights_unchanged':weights,'DINO_windows':5,'spatial_batches':2,'training_updates':0,'collection':0,'C_D_samples':0,'uploads':0});print('COMPUTE_COMPLETE',len(EXTRA),time.monotonic()-begin,flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:save(S/'results/FAILED.json',{'utc':now(),'error':repr(e),'no_automatic_tuning':True});raise
