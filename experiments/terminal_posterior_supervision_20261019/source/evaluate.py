from common import *
import torch,ast,fcntl,signal
from wm_symbolic.world_model import WorldModel
from wm_symbolic.dataset import Episodes,static_arguments
from fixed_dataset import CoreEpisodes
EXTRA=[];INTERNAL=[];RUN={};OLD=None;PROB=None
class CoverageData(Episodes):
 def __init__(self,split):
  self.project=P;self.split=split;self.event_count=3;self.skills=SKILLS;self.stats=read(CELL/'configs/normalization.json');rows=[r for r in read(R/'data/data_manifest.json')['records']if r['split']==split];self.cache={(x['episode_id'],x['step']):x['path']for x in read(R/'data/features/index.json')['items']if x['split']==split};by=collections.defaultdict(list);self.episodes=[]
  for r in rows:by[r['episode_id']].append(r)
  for eid,rr in sorted(by.items()):
   rr.sort(key=lambda r:r['step']);self.episodes.append({'episode_id':eid,'root_id':rr[0]['root_id'],'rows':rr,'labels':[read(r['labels_path'])for r in rr]})
  self.root_ids=sorted({r['root_id']for r in rows});self.transitions=len(rows);self.matched=list(range(len(self.episodes)))
def extra(m,data,enc,actions,posts,priors,qs,n,sample,ids,cp_rngs,dp,dq):
 before=torch.cuda.get_rng_state()
 with torch.random.fork_rng(devices=[0]),torch.autocast('cuda',dtype=torch.bfloat16):
  def append(mode,origin,target,state,head=None):
   prob=PROB(m if head is None else head,state,n,sample)
   for j,ei in enumerate(ids):
    ep=data.episodes[ei];lab=ep['labels'][0]['before']if target==0 else ep['labels'][target-1]['after'];prev=ep['labels'][origin-1]if origin else None;status='root'if prev is None else'success'if prev['success']else'failed'if prev['control_samples_including_start']>1 else'refused';EXTRA.append({**RUN,'root_id':ep['root_id'],'episode_id':ep['episode_id'],'mode':mode,'origin':origin,'target':target,'horizon':target-origin,'prob':prob[j],'cat':lab['categorical'],'cat_valid':lab['categorical_valid'],'origin_status':status})
  for t in range(1,5):append('prior_current',t,t,priors[t-1])
  for t in range(5):append('original_head_current',t,t,posts[t],OLD)
  for t in range(4):
   torch.cuda.set_rng_state(cp_rngs[t]);state,_=m.imagine(posts[t],actions[:,t].repeat_interleave(sample,0));append('original_head_one',t,t+1,state,OLD)
  for origin in range(1,4):
   state=posts[origin]
   for target in range(origin+1,5):
    torch.manual_seed(91831+100000*RUN['seed']+10000*SPLITS.index(RUN['split'])+100*ids[0]+10*origin+target);state,_=m.imagine(state,actions[:,target-1].repeat_interleave(sample,0));append('free_suffix',origin,target,state);append('original_head_suffix',origin,target,state,OLD)
  for t in range(5):
   h=dp[t].h.float().cpu().numpy();q=dq[t].float().cpu().numpy();e=enc[:,t].float().cpu().numpy()
   for j,ei in enumerate(ids):
    ep=data.episodes[ei];INTERNAL.append({**RUN,'root_id':ep['root_id'],'episode_id':ep['episode_id'],'boundary':t,'h':h[j],'q':q[j],'E':e[j]})
 assert torch.equal(torch.cuda.get_rng_state(),before)
def main():
 global RUN,OLD,PROB
 guard('eval');assert not(S/'results/evaluation_complete.json').exists();models=read(S/'configs/models_frozen.json');verify_inputs()
 for p,h in models['sha256'].items():assert sha(p)==h,p
 save(S/'configs/evaluation_frozen.json',{'utc':now(),'before_exploration_prediction':True,'files':{str(p):sha(p)for p in [S/'source/evaluate.py',S/'source/analyze.py']},'protocol_sha256':sha(S/'configs/protocol.json'),'model_selection_sha256':sha(S/'configs/models_frozen.json')})
 lock=(P/'runs/gpu.lock').open('r');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);start=time.monotonic();torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('high');ns=globals().copy();source=(H/'source/evaluate_with_h2.py').read_text();needle='posts,_,_,_=m.history(enc.repeat_interleave(sample,0),actions.repeat_interleave(sample,0));free=posts[0]';assert source.count(needle)==1;source=source.replace(needle,'posts,qs,priors,ps=m.history(enc.repeat_interleave(sample,0),actions.repeat_interleave(sample,0));free=posts[0];cp_rngs=[]');source=source.replace('rng=torch.cuda.get_rng_state();wp,_=m.imagine(posts[t],wa)','rng=torch.cuda.get_rng_state();cp_rngs.append(rng);wp,_=m.imagine(posts[t],wa)');source=source.replace('   if off%40==0:','   extra(m,data,enc,actions,posts,priors,qs,n,sample,ids,cp_rngs,dp,dq)\n   if off%40==0:');tree=ast.parse(source);nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name in ['probabilities','savearrays','assess']];exec(compile(ast.Module(body=nodes,type_ignores=[]),'<paired original evaluator>','exec'),ns);(S/'source/paired_evaluator.py').write_text(ast.unparse(ast.Module(body=nodes,type_ignores=[]))+'\n');PROB=ns['probabilities'];parity=[];runs=[];data={}
 for split in SPLITS:
  d=CoreEpisodes(P,split,read(CELL/'configs/normalization.json'))if split in ['train','validation']else CoverageData(split);cache=[d.get(i)for i in range(len(d.episodes))]
  def batch(ids,device,cache=cache):return {k:torch.stack([cache[int(i)][k]for i in ids]).to(device)for k in cache[0]}
  d.batch=batch;data[split]=d
 for seed in [0,1]:
  native=CELL/f'models/joint_seed{seed}/final6000.pt';OLD=WorldModel(skill_count=6,event_count=3).cuda();OLD.load_state_dict(torch.load(native,map_location='cuda',weights_only=True)['model']);OLD.eval().requires_grad_(False)
  variants=[('native',native)]
  for arm in ARMS:
   selected=next(r for r in models['selection']if(r['seed'],r['arm'])==(seed,arm));variants.append((arm,Path(selected['path'])));final=S/f'models/{arm}_seed{seed}_step750.pt'
   if final!=Path(selected['path']):variants.append((arm+'_final750',final))
  for variant,path in variants:
   m=WorldModel(skill_count=6,event_count=3).cuda();m.load_state_dict(torch.load(path,map_location='cuda',weights_only=True)['model']);m.eval().requires_grad_(False);before=statehash(m)
   for split in SPLITS:
    RUN={'seed':seed,'variant':variant,'split':split};runs.append(ns['assess'](m,data[split],variant,seed))
    if variant=='native':
     oldpath=CELL/f'data/predictions_joint_seed{seed}_{split}.npz'if split in ['train','validation']else G/f'data/predictions_joint_seed{seed}_{split}.npz';a=dict(np.load(S/f'data/predictions_native_seed{seed}_{split}.npz'));b=dict(np.load(oldpath));ix={(str(e),str(mode),int(t)):i for i,(e,mode,t)in enumerate(zip(a['episode_id'],a['mode'],a['step']))};idx=[ix[str(e),str(mode),int(t)]for e,mode,t in zip(b['episode_id'],b['mode'],b['step'])];checks={k:np.array_equal(a[k][idx],b[k],equal_nan=True)if b[k].dtype.kind in 'fc'else np.array_equal(a[k][idx],b[k])for k in b};assert all(checks.values()),checks;parity.append({'seed':seed,'split':split,'original_rows_exact':len(idx),'all_arrays_exact':True})
   assert statehash(m)==before;del m;torch.cuda.empty_cache()
  del OLD;torch.cuda.empty_cache()
 np.savez_compressed(S/'data/extra_predictions.npz',**{k:np.array([r[k]for r in EXTRA])for k in EXTRA[0]});np.savez_compressed(S/'data/internal_states.npz',**{k:np.array([r[k]for r in INTERNAL])for k in INTERNAL[0]});save(S/'results/baseline_parity.json',parity);verify_inputs()
 for p,h in models['sha256'].items():assert sha(p)==h
 save(S/'results/evaluation_complete.json',{'utc':now(),'seconds':time.monotonic()-start,'assessments':len(runs),'extra_rows':len(EXTRA),'internal_rows':len(INTERNAL),'original_rows_exact':sum(r['original_rows_exact']for r in parity),'models_unchanged':True,'peak_GiB':torch.cuda.max_memory_allocated()/1024**3,'no_fitting_collection_C_D_upload':True});print('EVALUATION_COMPLETE',time.monotonic()-start,flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:save(S/'results/EVAL_FAILED.json',{'utc':now(),'error':repr(e),'no_metric_driven_retuning':True});raise
