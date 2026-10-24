from common import *
import torch,fcntl,ast
from wm_symbolic.vision_cache import FrozenVision,VisibleRegistry,WEIGHTS
from wm_symbolic.world_model import WorldModel
from wm_symbolic.public_geometry import visible_world_points,colour_masks
node=next(n for n in ast.parse((OLD/'source/data.py').read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='feature');exec(compile(ast.Module(body=[node],type_ignores=[]),'unchanged public RGBD feature','exec'))
ns={'np':np,'COLORS':COLORS,'VisibleRegistry':VisibleRegistry}
for f,n in[(P/'src/wm_symbolic/public_geometry.py','colour_masks'),(P/'experiments/object_identity_readout_20261019/source/run.py','public_routes')]:
 node=next(x for x in ast.parse(f.read_text()).body if isinstance(x,ast.FunctionDef)and x.name==n);exec(compile(ast.Module(body=[node],type_ignores=[]),str(f),'exec'),ns)
def make_cache():
 encoder=FrozenVision(P,reserve_gpu=False);roots={r['root_id']:r for r in read(S/'data/roots_manifest.json')};by=collections.defaultdict(list)
 for r in read(S/'data/data_manifest.json')['records']:by[r['episode_id']].append(r)
 items=[];eps=[];geo=[];started=time.time()
 for eid,rr in sorted(by.items()):
  budget();rr.sort(key=lambda r:r['step']);d=Path(rr[0]['labels_path']).parent.parent;root=roots[rr[0]['root_id']];registry=VisibleRegistry();ee={'episode_id':eid,'root_id':root['root_id'],'family_id':root['family_id'],'template':root['template'],'branch':rr[0]['branch_index'],'held_color':root['held_color'],'target_color':root['matched_color'],'length':len(rr),'rows':rr,'boundaries':[],'probes':[]}
  windows=[{'type':'boundary','step':0,'public_inputs':[str(d/f'root_observation_{j}.npz')for j in range(4)]}]
  for t,r in enumerate(rr):
   sd=Path(r['labels_path']).parent;probes=read(sd/'held_probes_private.json')
   for q in probes['probes']:windows.append({'type':'probe','step':t+1,'action_index':t,'completion_control':q['completion_control'],'public_inputs':q['public_inputs'],'probe_labels_path':str(sd/'held_probes_private.json')})
   windows.append({'type':'boundary','step':t+1,'public_inputs':[str(sd/f'boundary_history_{j}.npz')for j in range(4)]})
  for item in windows:
   j=len(items);out=S/f'data/features/window_{j:04d}.npz';paths=[Path(x)for x in item['public_inputs']];np.savez(out,**encoder.encode(paths,registry));item.update(index=j,episode_id=eid,path=str(out));items.append(item)
   with np.load(paths[-1])as z:f={k:z[k]for k in z.files}
   ob,ct,_=feature(f);geo.append({'objects':ob,'context':ct})
   ee['boundaries'if item['type']=='boundary'else'probes'].append(j)
  eps.append(ee);save(S/'results/cache_progress.json',{'utc':now(),'episodes':len(eps),'windows':len(items),'seconds':time.time()-started});print('CACHE',len(eps),len(items),flush=True)
 del encoder;torch.cuda.empty_cache();save(S/'cache/items.json',items);save(S/'cache/episodes.json',eps);np.savez_compressed(S/'cache/geometry_raw.npz',objects=np.stack([g['objects']for g in geo]),context=np.stack([g['context']for g in geo]));save(S/'results/cache_complete.json',{'utc':now(),'episodes':len(eps),'windows':len(items),'probe_windows':sum(x['type']=='probe'for x in items),'seconds':time.time()-started,'DINO_weights_sha256':sha(WEIGHTS/'model.safetensors'),'future_frames_used':0})
def extract(seed):
 items=read(S/'cache/items.json');norm=read(CELL/'configs/normalization.json');vm=np.array(norm['vision_mean'],np.float32);vs=np.array(norm['vision_std'],np.float32);rm=np.array(norm['robot_mean'],np.float32);rs=np.array(norm['robot_std'],np.float32)
 enc=WorldModel(skill_count=6,event_count=3);enc.load_state_dict(torch.load(CELL/f'models/joint_seed{seed}/latest.pt',map_location='cpu',weights_only=False)['model']);enc.cuda().eval().requires_grad_(False);before=statehash(enc);captured={};hook=enc.spatial.register_forward_hook(lambda m,i,o:captured.update(x=o));arrays=[];started=time.time()
 with torch.inference_mode():
  for start in range(0,len(items),8):
   budget();part=items[start:start+8];fs=[dict(np.load(x['path']))for x in part];weights=[]
   for item,f in zip(part,fs):
    frames=[dict(np.load(p))for p in item['public_inputs']];ww,*_=ns['public_routes'](frames,f['visible_registry_ids'].tolist(),f['valid']);weights.append(ww)
   tok=torch.tensor(np.stack([(f['tokens'].astype(np.float32)-vm)/vs for f in fs]),device='cuda')[None];geom=torch.tensor(np.stack([f['geometry']for f in fs]),device='cuda')[None];valid=torch.tensor(np.stack([f['valid']for f in fs]),device='cuda')[None];robot=torch.tensor(np.stack([(f['robot']-rm)/rs for f in fs]),device='cuda')[None]
   g=enc.encode(tok,geom,valid,robot);sp=captured.pop('x').reshape(1,len(fs),513,256);w=torch.tensor(np.stack(weights),device='cuda',dtype=torch.float32)[None];v=w.sum(-1)/32.;pool=torch.einsum('bton,btnd->btod',w,sp)/w.sum(-1,keepdim=True).clamp_min(1);o=enc.observation(pool)*(v>0)[...,None];arrays.append({'objects':o[0].cpu().numpy(),'context':g[0].cpu().numpy(),'availability':v[0].cpu().numpy()})
 assert statehash(enc)==before;hook.remove();del enc;torch.cuda.empty_cache();np.savez_compressed(S/f'cache/front_seed{seed}.npz',**{k:np.concatenate([a[k]for a in arrays])for k in arrays[0]});save(S/f'results/front_seed{seed}.json',{'utc':now(),'windows':len(items),'seconds':time.time()-started,'frozen_encoder_unchanged':True})
def load(seed,arm):
 data=dict(np.load(S/f'cache/front_seed{seed}.npz'));norm=read(S/'configs/input_normalization.json')['values'][f'{arm}_seed{seed}']
 if arm=='geometry':raw=dict(np.load(S/'cache/geometry_raw.npz'))
 else:
  rng=np.random.default_rng(2610211000);po=rng.normal(size=(512,16)).astype(np.float32)/np.sqrt(512);pc=rng.normal(size=(512,32)).astype(np.float32)/np.sqrt(512);raw={'objects':data['objects']@po,'context':data['context']@pc}
 for name,x in raw.items():
  n=norm[name];out=(x-np.array(n['mean'],np.float32))/np.array(n['std'],np.float32)
  if arm=='geometry':
   width=8 if name=='objects'else 4
   for b in range(0,x.shape[-1],width):out[...,b:b+width]*=x[...,b+width-1:b+width]>0
  data[name]=np.concatenate([data[name],out.astype(np.float32)],-1)
 return{k:torch.from_numpy(v).cuda()for k,v in data.items()}
def main():
 guard();verify();budget();assert read(S/'results/collection_complete.json')['passed'];lock=(P/'runs/gpu.lock').open();fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(4);torch.set_float32_matmul_precision('high');torch.use_deterministic_algorithms(True)
 make_cache()
 for seed in [0,1]:extract(seed)
if __name__=='__main__':main()
