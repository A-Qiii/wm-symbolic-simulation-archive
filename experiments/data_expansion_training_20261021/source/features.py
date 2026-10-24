from common import *
import torch,ast
from wm_symbolic.vision_cache import FrozenVision,VisibleRegistry,WEIGHTS
from wm_symbolic.world_model import WorldModel
from wm_symbolic.public_geometry import visible_world_points,colour_masks
node=next(n for n in ast.parse((OLD/'source/data.py').read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='feature');exec(compile(ast.Module(body=[node],type_ignores=[]),'unchanged public RGBD feature','exec'))
ns={'np':np,'COLORS':COLORS,'VisibleRegistry':VisibleRegistry}
for f,n in [(P/'src/wm_symbolic/public_geometry.py','colour_masks'),(P/'experiments/object_identity_readout_20261019/source/run.py','public_routes')]:
 node=next(x for x in ast.parse(f.read_text()).body if isinstance(x,ast.FunctionDef)and x.name==n);exec(compile(ast.Module(body=[node],type_ignores=[]),str(f),'exec'),ns)
node=next(n for n in ast.parse((O/'source/data.py').read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='labels');exec(compile(ast.Module(body=[node],type_ignores=[]),'unchanged target packing','exec'))
def build(split):
 budget();path=S/f'cache/{split}_tokens_index.json'
 if path.exists():return
 eps=read(S/'configs/episodes.json')[split];encoder=FrozenVision(P,reserve_gpu=False);items=[];geometry=[];started=time.time()
 oldflag=torch.are_deterministic_algorithms_enabled();torch.use_deterministic_algorithms(False)
 try:
  for ei,e in enumerate(eps):
   budget();registry=VisibleRegistry();rr=e['rows'];d=Path(rr[0]['labels_path']).parent.parent
   windows=[[d/f'root_observation_{j}.npz'for j in range(4)]]+[[Path(r['labels_path']).parent/f'boundary_history_{j}.npz'for j in range(4)]for r in rr]
   outdir=S/'cache/tokens'/e['episode_id'];outdir.mkdir(parents=True,exist_ok=True);ee={**e,'features':[]};oo=[];cc=[]
   for t,paths in enumerate(windows):
    out=outdir/f'{t}.npz'
    if out.exists():registry.update([dict(np.load(p))for p in paths])
    else:
     temporary=out.with_suffix('.tmp.npz');np.savez(temporary,**encoder.encode(paths,registry));temporary.replace(out)
    item={'episode_id':e['episode_id'],'root_id':e['root_id'],'split':split,'step':t,'path':str(out),'public_inputs':[str(p)for p in paths]};ee['features'].append(item)
    with np.load(paths[-1])as z:f={k:z[k]for k in z.files}
    ob,ct,_=feature(f);oo.append(ob);cc.append(ct)
   items.append(ee);geometry.append({'objects':np.stack(oo),'context':np.stack(cc)})
   if(ei+1)%40==0:
    save(S/'results/cache_progress.json',{'utc':now(),'split':split,'episodes':ei+1,'total':len(eps),'seconds':time.time()-started});print('TOKENS',split,ei+1,len(eps),round(time.time()-started,1),flush=True)
 finally:torch.use_deterministic_algorithms(oldflag)
 del encoder;torch.cuda.empty_cache()
 np.savez_compressed(S/f'cache/{split}_geometry_raw.npz',objects=np.stack([g['objects']for g in geometry]),context=np.stack([g['context']for g in geometry]))
 save(path,items);save(S/f'results/tokens_{split}.json',{'utc':now(),'episodes':len(eps),'windows':7*len(eps),'seconds':time.time()-started,'public_boundary_only':True,'probe_windows_inserted':0,'frozen_DINO_sha256':sha(WEIGHTS/'model.safetensors')})
def extract(split,seed):
 path=S/f'cache/{split}_seed{seed}.npz'
 if path.exists():return
 eps=read(S/f'cache/{split}_tokens_index.json');norm=read(CELL/'configs/normalization.json');vm=np.array(norm['vision_mean'],np.float32);vs=np.array(norm['vision_std'],np.float32);rm=np.array(norm['robot_mean'],np.float32);rs=np.array(norm['robot_std'],np.float32)
 enc=WorldModel(skill_count=6,event_count=3);enc.load_state_dict(torch.load(CELL/f'models/joint_seed{seed}/latest.pt',map_location='cpu',weights_only=False)['model']);enc.cuda().eval().requires_grad_(False);before=statehash(enc);captured={};hook=enc.spatial.register_forward_hook(lambda m,i,o:captured.update(x=o));arrays=[];started=time.time()
 with torch.inference_mode():
  for ei,e in enumerate(eps):
   budget();fs=[dict(np.load(it['path']))for it in e['features']];weights=[]
   for it,f in zip(e['features'],fs):
    frames=[dict(np.load(p))for p in it['public_inputs']];ww,*_=ns['public_routes'](frames,f['visible_registry_ids'].tolist(),f['valid']);weights.append(ww)
   tok=torch.tensor(np.stack([(f['tokens'].astype(np.float32)-vm)/vs for f in fs]),device='cuda')[None];geom=torch.tensor(np.stack([f['geometry']for f in fs]),device='cuda')[None];valid=torch.tensor(np.stack([f['valid']for f in fs]),device='cuda')[None];robot=torch.tensor(np.stack([(f['robot']-rm)/rs for f in fs]),device='cuda')[None]
   g=enc.encode(tok,geom,valid,robot);sp=captured.pop('x').reshape(1,len(fs),513,256);w=torch.tensor(np.stack(weights),device='cuda',dtype=torch.float32)[None];v=w.sum(-1)/32.;pool=torch.einsum('bton,btnd->btod',w,sp)/w.sum(-1,keepdim=True).clamp_min(1);o=enc.observation(pool)*(v>0)[...,None]
   arrays.append({'objects':o[0].cpu().numpy(),'context':g[0].cpu().numpy(),'availability':v[0].cpu().numpy()})
   if(ei+1)%80==0:print('FRONT',split,seed,ei+1,len(eps),round(time.time()-started,1),flush=True)
 assert statehash(enc)==before;hook.remove();del enc;torch.cuda.empty_cache()
 ys=labels(eps);np.savez(path,**{k:np.stack([a[k]for a in arrays])for k in arrays[0]},**ys);save(S/f'cache/{split}_metadata.json',eps)
 save(S/f'results/front_{split}_seed{seed}.json',{'utc':now(),'episodes':len(eps),'seconds':time.time()-started,'frozen_encoder_unchanged':True,'sha256':sha(path)})
def phase(test=False):
 splits=['combination_test']if test else['train','development']
 if test:assert(S/'configs/models_frozen.json').exists()
 for split in splits:
  build(split)
  for seed in [0,1]:extract(split,seed)
