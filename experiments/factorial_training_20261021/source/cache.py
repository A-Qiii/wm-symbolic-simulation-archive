from common import *
import ast,torch
from wm_symbolic.vision_cache import FrozenVision,VisibleRegistry,WEIGHTS
from wm_symbolic.world_model import WorldModel
from wm_symbolic.public_geometry import visible_world_points,colour_masks
# Reuse the original public geometry, RGB/mask routing and target packing, unmodified.
def function(path,name,ns):
 node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)and n.name==name)
 exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),ns)
function(OLD/'source/data.py','feature',globals())
function(P/'experiments/object_identity_readout_20261019/source/run.py','public_routes',globals())
function(O/'source/data.py','labels',globals())
def metadata(split):
 ids=set(read(D/'data/membership.json')['episodes'][split]);by=collections.defaultdict(list)
 for r in read(D/'data/data_manifest.json')['records']:
  if r['episode_id']in ids:by[r['episode_id']].append(r)
 eps=[]
 for eid,rr in sorted(by.items()):
  rr.sort(key=lambda r:r['step']);assert len(rr)==6
  eps.append(dict(episode_id=eid,root_id=rr[0]['root_id'],family_id=rr[0]['family_id'],split=split,kind='factorial',branch=rr[0]['branch_index'],branch_role=rr[0]['branch_role'],length=6,rows=rr))
 return eps
def augmented(data,raw,seed):
 norm=read(M/'configs/input_normalization.json')['values'][f'geometry_seed{seed}']
 for name,x in raw.items():
  n=norm[name];out=(x-np.array(n['mean'],np.float32))/np.array(n['std'],np.float32);width=8 if name=='objects'else 4
  for b in range(0,x.shape[-1],width):out[...,b:b+width]*=x[...,b+width-1:b+width]>0
  data[name]=np.concatenate([data[name],out.astype(np.float32)],-1)
 return data
def olddata(split,seed):
 return augmented(dict(np.load(O/f'cache/{split}_seed{seed}.npz')),dict(np.load(OLD/f'cache/{split}_geometry_raw.npz')),seed)
def load_arrays(split,seed):
 if split in ['old_train','old_development']:return olddata(split[4:],seed)
 if split=='combined':
  a=olddata('train',seed);b=dict(np.load(S/f'cache/train_seed{seed}.npz'));assert set(a)==set(b)
  return{k:np.concatenate([a[k],b[k]])for k in a}
 return dict(np.load(S/f'cache/{split}_seed{seed}.npz'))
def eps_for(split):
 if split in ['old_train','old_development']:return read(O/f'cache/{split[4:]}_metadata.json')
 if split=='combined':return read(O/'cache/train_metadata.json')+metadata('train')
 return metadata(split)
def run_cache(split):
 if split=='combination_test':assert(S/'configs/models_frozen.json').exists()
 dest=S/f'cache/{split}_seed1.npz'
 if dest.exists():
  assert read(S/f'results/cache_{split}.json')['passed'];return
 eps=metadata(split);vision=FrozenVision(P,reserve_gpu=False);encoders={};captures={};hashes={};hooks=[]
 norm=read(CELL/'configs/normalization.json');vm=np.array(norm['vision_mean'],np.float32);vs=np.array(norm['vision_std'],np.float32);rm=np.array(norm['robot_mean'],np.float32);rs=np.array(norm['robot_std'],np.float32)
 for seed in SEEDS:
  enc=WorldModel(skill_count=6,event_count=3);enc.load_state_dict(torch.load(CELL/f'models/joint_seed{seed}/latest.pt',map_location='cpu',weights_only=False)['model']);enc.cuda().eval().requires_grad_(False);encoders[seed]=enc;hashes[seed]=statehash(enc);captures[seed]={}
  hooks.append(enc.spatial.register_forward_hook(lambda m,i,o,seed=seed:captures[seed].update(x=o)))
 began=time.time();cdir=S/'cache/episodes';cdir.mkdir(exist_ok=True);new=0
 torch.use_deterministic_algorithms(False)
 for ei,e in enumerate(eps):
  budget();out=cdir/(e['episode_id']+'.npz')
  if out.exists():
   with np.load(out)as z:assert z['objects0'].shape==(7,3,528)and z['context1'].shape==(7,544)
   continue
  d=Path(e['rows'][0]['labels_path']).parent.parent
  windows=[[d/f'root_observation_{j}.npz'for j in range(4)]]+[[Path(r['labels_path']).parent/f'boundary_history_{j}.npz'for j in range(4)]for r in e['rows']]
  reg=VisibleRegistry();fs=[];weights=[];geo=[];provenance=[]
  for paths in windows:
   f=vision.encode(paths,reg);fs.append(f);frames=[]
   for p in paths:
    with np.load(p)as z:frames.append({k:z[k]for k in z.files})
    provenance.append(dict(path=str(p),sha256=sha(p)))
   ww,*_=public_routes(frames,f['visible_registry_ids'].tolist(),f['valid']);weights.append(ww);ob,ct,_=feature(frames[-1]);geo.append(dict(objects=ob,context=ct))
  raw={k:np.stack([g[k]for g in geo])for k in geo[0]};arrays={}
  with torch.inference_mode():
   tok=torch.tensor(np.stack([(f['tokens'].astype(np.float32)-vm)/vs for f in fs]),device='cuda')[None];geom=torch.tensor(np.stack([f['geometry']for f in fs]),device='cuda')[None];valid=torch.tensor(np.stack([f['valid']for f in fs]),device='cuda')[None];robot=torch.tensor(np.stack([(f['robot']-rm)/rs for f in fs]),device='cuda')[None];w=torch.tensor(np.stack(weights),device='cuda',dtype=torch.float32)[None]
   for seed,enc in encoders.items():
    g=enc.encode(tok,geom,valid,robot);sp=captures[seed].pop('x').reshape(1,7,513,256);v=w.sum(-1)/32.;pool=torch.einsum('bton,btnd->btod',w,sp)/w.sum(-1,keepdim=True).clamp_min(1);o=enc.observation(pool)*(v>0)[...,None]
    a=augmented(dict(objects=o[0].cpu().numpy(),context=g[0].cpu().numpy(),availability=v[0].cpu().numpy()),raw,seed)
    arrays.update({k+str(seed):val for k,val in a.items()})
  tmp=out.with_suffix('.partial.npz');np.savez_compressed(tmp,**arrays);tmp.replace(out);save(cdir/(e['episode_id']+'_provenance.json'),dict(public_files=provenance,registry_causal=True));new+=1
  if(ei+1)%10==0:
   save(S/'results/cache_progress.json',dict(utc=now(),split=split,episodes=ei+1,total=len(eps),windows=(ei+1)*7,seconds=time.time()-began));print('CACHE',split,ei+1,len(eps),round(time.time()-began,1),flush=True)
 for seed,enc in encoders.items():assert statehash(enc)==hashes[seed]
 for h in hooks:h.remove()
 del vision,encoders;torch.cuda.empty_cache();torch.use_deterministic_algorithms(True)
 ys=labels(eps)
 for seed in SEEDS:
  arr=collections.defaultdict(list)
  for e in eps:
   with np.load(cdir/(e['episode_id']+'.npz'))as z:
    for k in ['objects','context','availability']:arr[k].append(z[k+str(seed)])
  path=S/f'cache/{split}_seed{seed}.npz';tmp=path.with_suffix('.partial.npz');np.savez_compressed(tmp,**{k:np.stack(v)for k,v in arr.items()},**ys);tmp.replace(path)
 save(S/f'cache/{split}_metadata.json',eps);save(S/f'results/cache_{split}.json',dict(utc=now(),passed=True,episodes=len(eps),windows=7*len(eps),new_cached_episodes=new,seconds=time.time()-began,frozen_encoders_unchanged=True,only_public_model_inputs=True,original_normalization_retained=True,DINO_sha256=sha(WEIGHTS/'model.safetensors')))
 print('CACHE_COMPLETE',split,len(eps),flush=True)
