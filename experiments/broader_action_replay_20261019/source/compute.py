from common import *
import torch,ast,collections,fcntl,signal
from wm_symbolic.world_model import WorldModel
from wm_symbolic.dataset import static_arguments
def extract(path,names,namespace):
 tree=ast.parse(path.read_text());nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))and n.name in names];assert len(nodes)==len(names);exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
def state_hash(m):
 h=hashlib.sha256()
 for k,v in sorted(m.state_dict().items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
class CachedData:
 def __init__(self,split,seed,old=False):
  self.split=split;self.skills=SKILLS;root=O if old else R;self.features=dict(np.load(root/f'data/features_seed{seed}_{split}.npz'));self.index={(str(e),int(t)):i for i,(e,t)in enumerate(zip(self.features['episode_id'],self.features['step']))};allowed=set(self.features['root_id']);rows=[r for r in read(P/'data/core1200_v1/data_manifest.json'if old else R/'data/data_manifest.json')['records']if r['root_id']in allowed];by=collections.defaultdict(list)
  for r in rows:by[r['episode_id']].append(r)
  self.episodes=[]
  for eid,rr in sorted(by.items()):
   rr.sort(key=lambda r:r['step']);self.episodes.append({'episode_id':eid,'root_id':rr[0]['root_id'],'rows':rr,'labels':[read(r['labels_path'])for r in rr]})
 def batch(self,ids,device):
  rows=[self.episodes[i]['rows']+[self.episodes[i]['rows'][-1]]*2 for i in ids];return {'skill':torch.tensor([[SKILLS.index(r['skill'])for r in rr]for rr in rows],device=device),'arguments':torch.tensor([[static_arguments(r['skill'])for r in rr]for rr in rows],device=device),'budget':torch.tensor([[r['executor']['budget_steps']for r in rr]for rr in rows],device=device,dtype=torch.float32)}
 def cached_encoded(self,ids,device):
  a=np.array([[self.features['global'][self.index[self.episodes[i]['episode_id'],min(t,4)]]for t in range(7)]for i in ids]);return torch.tensor(a,device=device,dtype=torch.bfloat16)
def main():
 sys.addaudithook(guard);lock=(P/'runs/gpu.lock').open('r');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);signal.signal(signal.SIGALRM,lambda *_:(_ for _ in()).throw(TimeoutError('Frozen 600 second compute limit')));signal.alarm(600);begin=time.monotonic();verify_inputs();assert not(S/'results/compute_complete.json').exists();torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('high');parity=[];weights={};runs=[]
 ns=globals().copy();ns['E']=S;source=(H/'source/evaluate_with_h2.py').read_text();needle="enc=m.encode(b['tokens'],b['geometry'],b['visible'],b['robot'])";assert source.count(needle)==1;source=source.replace(needle,"enc=data.cached_encoded(ids,'cuda')");tree=ast.parse(source);nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name in ['probabilities','savearrays','assess']];exec(compile(ast.Module(body=nodes,type_ignores=[]),'<cached exact evaluator>','exec'),ns);(S/'source/cached_evaluator.py').write_text(ast.unparse(ast.Module(body=nodes,type_ignores=[]))+'\n')
 headns={'torch':torch,'nn':torch.nn};extract(O/'source/common.py',['Head','probabilities'],headns)
 for seed in [0,1]:
  cp=B/f'cells/roots40/models/joint_seed{seed}/final6000.pt';model=WorldModel(skill_count=6,event_count=3).cuda();model.load_state_dict(torch.load(cp,weights_only=True,map_location='cuda')['model']);model.eval().requires_grad_(False);before=state_hash(model)
  # Replay old historical test first, including RNG draws for wrong-action controls.
  runs.append(ns['assess'](model,CachedData('test',seed,old=True),'joint',seed));new=np.load(S/f'data/predictions_joint_seed{seed}_test.npz');old=np.load(H/f'data/predictions_joint_seed{seed}_test.npz');check={k:np.array_equal(new[k],old[k],equal_nan=True)if old[k].dtype.kind in 'fc'else np.array_equal(new[k],old[k])for k in old.files};assert all(check.values()),check;parity.append({'seed':seed,'model':'native_RSSM','arrays_exact':check,'rows':len(old['prob']),'max_probability_difference':float(abs(new['prob']-old['prob']).max())});save(S/'results/parity_progress.json',parity)
  for split in SPLITS:runs.append(ns['assess'](model,CachedData(split,seed),'joint',seed))
  assert state_hash(model)==before;weights[str(cp)]=sha(cp);del model;torch.cuda.empty_cache()
  for arm,root,filename in [('old_object',O,'head_object'),('coverage_object',R,'head_coverage')]:
   cp=root/f'models/{filename}_seed{seed}.pt';head=headns['Head']().cuda();head.load_state_dict(torch.load(cp,weights_only=True,map_location='cuda')['state']);head.eval().requires_grad_(False);before=state_hash(head)
   for split in (['test']if arm=='old_object'else[])+SPLITS:
    raw=dict(np.load((O if split=='test'else R)/f'data/features_seed{seed}_{split}.npz'));a={k:torch.tensor(raw[k],device='cuda')for k in ['object','global','availability']}
    with torch.inference_mode():prob=headns['probabilities'](head(a['object'],a['global'],a['availability'])).cpu().numpy()
    if split=='test':expected=O/f'data/predictions_object_seed{seed}_test.npz'
    elif split!='train':expected=R/f'data/predictions_{"control"if arm=="old_object"else"coverage"}_seed{seed}_{split}.npz'
    else:expected=None
    if expected:
     ref=np.load(expected);assert np.array_equal(prob,ref['prob']),str(expected);parity.append({'seed':seed,'model':arm,'split':split,'rows':len(prob),'max_probability_difference':0.})
    np.savez_compressed(S/f'data/predictions_{arm}_seed{seed}_{split}.npz',prob=prob,**{k:raw[k]for k in ['root_id','episode_id','step','cat']})
   assert state_hash(head)==before;weights[str(cp)]=sha(cp);del head;torch.cuda.empty_cache()
 # Reuse the same small dynamics weights under four explicitly different starting interfaces.
 smallns={'torch':torch,'nn':torch.nn};extract(D/'source/train.py',['Model','outputs','probabilities'],smallns);torch.set_float32_matmul_precision('highest');normal=np.load(D/'configs/physical_normalization.npz')
 for row in read(S/'configs/selected_models.json')['selection']:
  seed=row['seed'];arm=row['arm'];model=smallns['Model'](112).cuda();model.load_state_dict(torch.load(row['path'],weights_only=True,map_location='cuda')['state']);model.eval().requires_grad_(False);before=state_hash(model)
  for split in ['test']+SPLITS:
   d=dict(np.load((D if split=='test'else S)/f'data/dataset_{split}.npz'));physical=(d['physical']-normal['mean'])/normal['std'];physical=np.zeros_like(physical)if arm=='relations'else physical;variants=['truth']+(['rssm','old_object','coverage_object']if arm=='relations'and split!='test'else[])
   for variant in variants:
    x=d['initial_state'].copy()
    if variant!='truth':
     f=np.load(S/f'data/predictions_{"joint"if variant=="rssm"else variant}_seed{seed}_{split}.npz');ix={(str(e),int(t)):i for i,(e,t)in enumerate(zip(f['episode_id'],f['step']))if variant!='rssm'or f['mode'][i]=='current'};v=np.array([[rel_from_prob(f['prob'][ix[str(e),t]])for t in range(5)]for e in d['episode_id']]);x[...,:9]=hard(v);x[...,9:]=1;np.savez_compressed(S/f'data/current_{variant}_seed{seed}_{split}.npz',prob=v,episode_id=d['episode_id'],root_id=d['root_id'])
    xx=torch.tensor(np.concatenate([x,physical],-1),device='cuda',dtype=torch.float32);aa=torch.tensor(d['action'],device='cuda',dtype=torch.float32)
    with torch.inference_mode():one,free=smallns['outputs'](model,xx,aa);op=smallns['probabilities'](one);fp=smallns['probabilities'](free)
    if split=='test':
     ref=np.load(D/f'data/predictions_{arm}_truth_seed{seed}_test.npz');assert np.array_equal(op,ref['one_step'])and np.array_equal(fp,ref['free']);parity.append({'seed':seed,'model':arm,'split':'test','one_step_and_free_exact':True})
    else:np.savez_compressed(S/f'data/predictions_small_{arm}_{variant}_seed{seed}_{split}.npz',one_step=op,free=fp,episode_id=d['episode_id'],root_id=d['root_id'])
  assert state_hash(model)==before;weights[row['path']]=sha(row['path']);del model;torch.cuda.empty_cache()
 verify_inputs();save(S/'results/parity.json',parity);save(S/'results/compute_complete.json',{'utc':now(),'seconds':time.monotonic()-begin,'peak_cuda_GiB':torch.cuda.max_memory_allocated()/1024**3,'new_fits':0,'collection':0,'DINO_forwards':0,'spatial_encoder_forwards':0,'frozen_RSSM_assessments':len(runs),'all_previous_predictions_exact':True,'model_weights_unchanged':weights,'runs':runs,'C_D_samples':0,'uploads':0});print('COMPUTE_COMPLETE',time.monotonic()-begin,flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:save(S/'results/FAILED.json',{'utc':now(),'error':repr(e),'no_automatic_repair_or_tuning':True});raise
