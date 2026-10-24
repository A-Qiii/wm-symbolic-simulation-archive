from common import *
import ast,torch
from wm_symbolic.world_model import WorldModel
class VisibleRegistry:
 @staticmethod
 def grouped(mask):
  mask=mask.copy();mask[(mask>=0)&((mask&((1<<24)-1))==0)]=0;return mask
ns={'np':np,'COLORS':COLORS,'VisibleRegistry':VisibleRegistry}
for f,n in[(P/'src/wm_symbolic/public_geometry.py','colour_masks'),(P/'experiments/object_identity_readout_20261019/source/run.py','public_routes')]:
 node=next(x for x in ast.parse(f.read_text()).body if isinstance(x,ast.FunctionDef)and x.name==n);exec(compile(ast.Module(body=[node],type_ignores=[]),str(f),'exec'),ns)
def metadata(split):
 member=read(S/'configs/data_membership.json')[split];roots={x['root_id']:x for x in read(R/'data/roots_manifest.json')};audits={x['root_id']:x for x in read(R/'results/root_audits.json')};out=[]
 for kind,manifest,index,ids in [('core',P/'data/core1200_v1/data_manifest.json',P/'data/core1200_v1/features/index.json',member['core_root_ids']),('role',R/'data/data_manifest.json',R/'data/features/index.json',member['role_root_ids'])]:
  ids=set(ids);by=collections.defaultdict(list);features={(f['episode_id'],f['step']):f for f in read(index)['items']if f['root_id']in ids}
  for r in read(manifest)['records']:
   if r['root_id']in ids:by[r['episode_id']].append(r)
  for eid,rr in sorted(by.items()):
   rr.sort(key=lambda r:r['step']);n=len(rr);assert n==(4 if kind=='core'else 6);rid=rr[0]['root_id'];root=roots.get(rid,{})
   out.append({'episode_id':eid,'root_id':rid,'family_id':root.get('family_id',rid),'kind':kind,'split':split,'branch':int(eid.rsplit('_b',1)[1]),'length':n,'held_color':root.get('held_color',''),'target_color':root.get('matched_color',''),'valid_pair':audits[rid]['valid_pair']if kind=='role'else True,'rows':rr,'features':[features[eid,t]for t in range(n+1)]})
 return out
def labels(eps):
 out=[]
 for e in eps:
  ys=[read(r['labels_path'])for r in e['rows']];ll=[ys[0]['before']]+[x['after']for x in ys];n=len(ys);ll+=([ll[-1]]*(7-len(ll)));rr=e['rows']+[e['rows'][-1]]*(6-n);yy=ys+[ys[-1]]*(6-n)
  verbs=[];pointers=[]
  for r in rr:
   skill=r['skill'];verbs.append(0 if skill=='open_drawer'else 1 if skill=='close_drawer'else 3 if skill=='place_in_drawer'else 2);ptr=[0.,0.,0.]
   if skill.startswith('lift_'):
    assert skill.endswith('_table');ptr[COLORS.index(skill.split('_')[1])]=1.
   pointers.append(ptr)
  out.append({'verb':np.array(verbs,np.int64),'pointer':np.array(pointers,np.float32),'budget':np.array([r['executor']['budget_steps']for r in rr],np.float32),'step_valid':np.arange(6)<n,'cat':np.array([x['categorical']for x in ll],np.int64),'cat_valid':np.array([x['categorical_valid']for x in ll],bool),'binary':np.array([x['binary']for x in ll],np.float32),'binary_valid':np.array([x['binary_valid']for x in ll],bool),'outcomes':np.array([[float(y['success'])]+y['events']for y in yy],np.float32),'events_valid':np.array([y['events_valid']for y in yy],bool)})
 return {k:np.stack([x[k]for x in out])for k in out[0]}
def extract(split,seed):
 budget();eps=metadata(split);path=S/f'cache/{split}_seed{seed}.npz'
 if path.exists():return
 if split=='historical_evaluation':assert (S/'configs/models_frozen.json').exists()
 norm=read(CELL/'configs/normalization.json');vm=np.array(norm['vision_mean'],np.float32);vs=np.array(norm['vision_std'],np.float32);rm=np.array(norm['robot_mean'],np.float32);rs=np.array(norm['robot_std'],np.float32)
 enc=WorldModel(skill_count=6,event_count=3);enc.load_state_dict(torch.load(CELL/f'models/joint_seed{seed}/latest.pt',map_location='cpu',weights_only=False)['model']);enc.cuda().eval().requires_grad_(False);before=statehash(enc);captured={};hook=enc.spatial.register_forward_hook(lambda m,i,o:captured.update(x=o));arrays=[];visible=missing=0;start=time.time()
 with torch.inference_mode():
  for ei,e in enumerate(eps):
   budget();fs=[dict(np.load(x['path']))for x in e['features']];weights=[]
   for item,f in zip(e['features'],fs):
    frames=[dict(np.load(p))for p in item['public_inputs']];ww,*_=ns['public_routes'](frames,f['visible_registry_ids'].tolist(),f['valid']);weights.append(ww)
   tok=torch.tensor(np.stack([(f['tokens'].astype(np.float32)-vm)/vs for f in fs]),device='cuda')[None];geom=torch.tensor(np.stack([f['geometry']for f in fs]),device='cuda')[None];valid=torch.tensor(np.stack([f['valid']for f in fs]),device='cuda')[None];robot=torch.tensor(np.stack([(f['robot']-rm)/rs for f in fs]),device='cuda')[None]
   g=enc.encode(tok,geom,valid,robot);sp=captured.pop('x').reshape(1,len(fs),513,256);w=torch.tensor(np.stack(weights),device='cuda',dtype=torch.float32)[None];v=w.sum(-1)/32.;pool=torch.einsum('bton,btnd->btod',w,sp)/w.sum(-1,keepdim=True).clamp_min(1);o=enc.observation(pool)*(v>0)[...,None];visible+=int((v>0).sum());missing+=int((v==0).sum());a={'objects':o[0].cpu().numpy(),'context':g[0].cpu().numpy(),'availability':v[0].cpu().numpy()};arrays.append({k:np.concatenate([x,np.repeat(x[-1:],7-len(x),axis=0)])for k,x in a.items()})
   if(ei+1)%50==0:print('FEATURES',split,seed,ei+1,len(eps),round(time.time()-start,1),flush=True)
 assert statehash(enc)==before;hook.remove();del enc;torch.cuda.empty_cache();ys=labels(eps);np.savez(path,**{k:np.stack([x[k]for x in arrays])for k in arrays[0]},**ys);save(S/f'cache/{split}_metadata.json',eps);save(S/f'results/features_{split}_seed{seed}.json',{'utc':now(),'episodes':len(eps),'visible_routes':visible,'missing_routes':missing,'sha256':sha(path),'encoder_state_unchanged':True,'seconds':time.time()-start});print('FEATURES_DONE',split,seed,flush=True)
def load(split,seed,device='cuda'):
 return {k:torch.from_numpy(v).to(device)for k,v in dict(np.load(S/f'cache/{split}_seed{seed}.npz')).items()}
def batch(data,ids):return {k:v[ids]for k,v in data.items()}
