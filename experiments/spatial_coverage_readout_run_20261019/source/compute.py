from common import *
import torch,ast,fcntl,time,argparse,collections
os.environ['MPLCONFIGDIR']=str(S/'cache/matplotlib');os.environ['TRITON_CACHE_DIR']=str(S/'cache/triton');os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from wm_symbolic.world_model import WorldModel
from wm_symbolic.dataset import Episodes
from wm_symbolic.vision_cache import FrozenVision,VisibleRegistry
from wm_symbolic.public_geometry import colour_masks
ns={'nn':torch.nn,'torch':torch,'np':np,'COLORS':COLORS,'VisibleRegistry':VisibleRegistry,'colour_masks':colour_masks}
for file,name,kind in [(O/'source/common.py','Head',ast.ClassDef),(O/'source/run.py','public_routes',ast.FunctionDef),(O/'source/common.py','probabilities',ast.FunctionDef)]:
 node=next(n for n in ast.parse(file.read_text()).body if isinstance(n,kind)and n.name==name);exec(compile(ast.Module(body=[node],type_ignores=[]),str(file), 'exec'),ns)
Head=ns['Head'];public_routes=ns['public_routes'];probabilities=ns['probabilities']
def th(state):
 h=hashlib.sha256()
 for k,v in sorted(state.items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
class Data(Episodes):
 def __init__(self,split):
  self.project=P;self.split=split;self.event_count=3;self.skills=read(P/'runs/core1200_v1/protocol.json')['skills'];self.stats=read(B/'cells/roots40/configs/normalization.json');rows=[r for r in read(S/'data/data_manifest.json')['records']if r['split']==split];self.cache={(x['episode_id'],x['step']):x['path']for x in read(S/'data/features/index.json')['items']if x['split']==split};by={};self.episodes=[]
  for row in rows:by.setdefault(row['episode_id'],[]).append(row)
  for eid,rr in sorted(by.items()):
   rr.sort(key=lambda r:r['step']);assert [r['step']for r in rr]==[0,1,2,3];self.episodes.append({'episode_id':eid,'root_id':rr[0]['root_id'],'rows':rr,'labels':[read(r['labels_path'])for r in rr]})
  self.root_ids=sorted({r['root_id']for r in rows});self.transitions=len(rows);self.matched=list(range(len(self.episodes)))
def cache(splits):
 start=time.monotonic();torch.use_deterministic_algorithms(False);torch.set_float32_matmul_precision('highest');rows=read(S/'data/data_manifest.json')['records'];by={}
 for row in rows:
  if row['split']in splits:by.setdefault(row['episode_id'],[]).append(row)
 path=S/'data/features/index.json';index=read(path)['items']if path.exists()else[];existing={(x['episode_id'],x['step'])for x in index};enc=FrozenVision(P,reserve_gpu=False)
 reference=next(x for x in read(P/'data/core1200_v1/features/index.json')['items']if x['episode_id']=='A_core1200_02_b0'and x['step']==0);check=enc.encode(reference['public_inputs'],VisibleRegistry());original=dict(np.load(reference['path']));differences={}
 for key,value in check.items():
  assert np.array_equal(value,original[key]),'Historical cache mismatch: '+key;differences[key]=0.
 save(S/f'results/cache_numerical_check_{splits[0]}.json',{'utc':now(),'passed':True,'reference':reference['path'],'exact_keys':list(check),'max_differences':differences,'settings':'Historical cache defaults: deterministic=False, matmul highest; training/extraction deterministic=True, matmul high restored after cache','extra_fits':0})
 for eid,rr in sorted(by.items()):
  rr.sort(key=lambda r:r['step']);ed=Path(rr[0]['obs_before']).parent.parent;reg=VisibleRegistry();boundaries=[[ed/f'root_observation_{j}.npz'for j in range(4)]]+[[Path(r['obs_after']).parent/f'boundary_history_{j}.npz'for j in range(4)]for r in rr]
  for h,paths in enumerate(boundaries):
   out=S/'data/features'/f'{eid}_{h:02d}.npz';out.parent.mkdir(parents=True,exist_ok=True)
   if out.exists():reg.update([dict(np.load(p))for p in paths])
   else:np.savez(out,**enc.encode(paths,reg))
   if(eid,h)not in existing:index.append({'episode_id':eid,'root_id':rr[0]['root_id'],'split':rr[0]['split'],'step':h,'path':str(out),'public_inputs':[str(p)for p in paths]})
  save(path,{'items':index});print('CACHE',eid,len(index),flush=True)
 del enc;torch.cuda.empty_cache();torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('high');save(S/f'results/cache_{splits[0]}.json',{'utc':now(),'splits':splits,'items':len(index),'seconds':time.monotonic()-start})
def routing(split):
 records=[];audit=[];wrong=[];geometrymax=0.;data=Data(split);items={(x['episode_id'],x['step']):x for x in read(S/'data/features/index.json')['items']if x['split']==split}
 for ep in data.episodes:
  registry=VisibleRegistry()
  for h in range(5):
   item=items[ep['episode_id'],h];f=dict(np.load(item['path']));frames=[dict(np.load(p))for p in item['public_inputs']];assert registry.update(frames)==f['visible_registry_ids'].tolist();ids=f['visible_registry_ids'].tolist();weights,slots,chosen,candidates=public_routes(frames,ids,f['valid'])
   for t,frame in enumerate(frames):
    for vi,cam in enumerate(['static','gripper']):
     seg=VisibleRegistry.grouped(frame['seg_'+cam]);height,width=seg.shape;top=(224-height)//2;left=(224-width)//2
     for slot in range(12):
      yy,xx=np.where(seg==ids[slot]);vis=len(yy)>=2;box=[(xx.min()+left)/224,(yy.min()+top)/224,(xx.max()+left+1)/224,(yy.max()+top+1)/224,float(vis)]if vis else[0,0,1/224,1/224,0];g=f['geometry'][(t*2+vi)*64+slot*4:(t*2+vi)*64+slot*4+4];geometrymax=max(geometrymax,float(np.abs(g[:,:5]-np.asarray(box,np.float32)).max()))
   r=ep['rows'][max(h-1,0)];info=read(Path(r['labels_path']).parent/'endpoint_info_private.json')['before'if h==0 else'after']
   for j,c in enumerate(COLORS):
    uid=info['scene_info']['movable_objects']['block_'+c]['uid'];visible=any(np.sum(VisibleRegistry.grouped(frame['seg_'+cam])==uid)>=2 for frame in frames for cam in['static','gripper']);a={'root_id':ep['root_id'],'episode_id':ep['episode_id'],'h':h,'color':c,'visible':bool(visible),'chosen_uid':int(chosen[j]),'expected_uid_audit_only':uid,'wrong':bool(chosen[j]>=0 and chosen[j]!=uid),'unresolved_visible':bool(visible and not weights[j].any())};audit.append(a)
    if a['wrong']:wrong.append(a)
   records.append({'root_id':ep['root_id'],'episode_id':ep['episode_id'],'step':h,'weights':weights,'availability':weights.sum(1).astype(np.float32)/32,'slots':slots})
 stats={c:{'visible':sum(a['visible']for a in audit if a['color']==c),'unresolved_visible':sum(a['unresolved_visible']for a in audit if a['color']==c)}for c in COLORS};limit=.05 if split=='train'else.10;passed=not wrong and geometrymax==0 and all(v['unresolved_visible']/max(v['visible'],1)<=limit for v in stats.values())
 np.savez_compressed(S/f'data/routes_{split}.npz',**{k:np.array([r[k]for r in records])for k in records[0]});save(S/f'results/routing_{split}.json',{'utc':now(),'passed':passed,'geometry_max':geometrymax,'stats':stats,'audit':audit,'wrong':wrong});assert passed,'Public routing gate failed; no target repair'
def extract(seed,split):
 start=time.monotonic();data=Data(split);route=dict(np.load(S/f'data/routes_{split}.npz'));lookup={(e,int(h)):i for i,(e,h)in enumerate(zip(route['episode_id'],route['step']))};cp=B/f'cells/roots40/models/joint_seed{seed}/final6000.pt';expected=read(B/'configs/all_models_frozen.json')['sha256'][str(cp)];assert sha(cp)==expected
 model=WorldModel(skill_count=6,event_count=3).cuda();model.load_state_dict(torch.load(cp,map_location='cuda',weights_only=True)['model']);model.eval().requires_grad_(False);before=th(model.state_dict());capture={};hook=model.spatial.register_forward_hook(lambda mod,inputs,output:capture.update(tokens=output));records=[]
 with torch.inference_mode():
  for off in range(0,len(data.episodes),4):
   indices=list(range(off,min(off+4,len(data.episodes))));batch=data.batch(indices,'cuda');n=len(indices)
   with torch.autocast('cuda',dtype=torch.bfloat16):
    encoded=model.encode(batch['tokens'],batch['geometry'],batch['visible'],batch['robot']);spatial=capture.pop('tokens').reshape(n,7,513,256)[:,:5];ix=np.array([[lookup[data.episodes[j]['episode_id'],h]for h in range(5)]for j in indices]);weights=torch.tensor(route['weights'][ix],device='cuda',dtype=torch.float32);pooled=torch.einsum('bhot,bhtd->bhod',weights,spatial.float())/weights.sum(-1,keepdim=True).clamp_min(1);objects=model.observation(pooled).float();availability=torch.tensor(route['availability'][ix],device='cuda');objects=objects*(availability>0)[...,None]
   gf=encoded[:,:5].float().cpu().numpy();of=objects.cpu().numpy()
   for j,ei in enumerate(indices):
    ep=data.episodes[ei]
    for h in range(5):
     label=ep['labels'][0]['before']if h==0 else ep['labels'][h-1]['after'];i=lookup[ep['episode_id'],h];records.append({'root_id':ep['root_id'],'episode_id':ep['episode_id'],'step':h,'global':gf[j,h],'object':of[j,h],'availability':route['availability'][i],'cat':np.array(label['categorical'][:4],np.int64),'valid':np.array(label['categorical_valid'][:4],bool)})
 hook.remove();assert th(model.state_dict())==before and sha(cp)==expected;np.savez_compressed(S/f'data/features_seed{seed}_{split}.npz',**{k:np.array([r[k]for r in records])for k in records[0]});save(S/f'results/extraction_seed{seed}_{split}.json',{'utc':now(),'records':len(records),'encoder_unchanged':True,'checkpoint_sha256':expected,'normalization_sha256':sha(B/'cells/roots40/configs/normalization.json'),'seconds':time.monotonic()-start});del model;torch.cuda.empty_cache()
def train(seed):
 raw=dict(np.load(S/f'data/features_seed{seed}_train.npz'));assert len(raw['cat'])==600 and raw['valid'].all(),'Invalid labels retained; exact historical loss pairing cannot proceed'
 mapping=read(S/'configs/paired_boundary_indices.json')
 for m in mapping:
  j=m['index'];assert raw['root_id'][j]==m['new_root']and raw['episode_id'][j]==m['new_episode']and raw['step'][j]==m['boundary']
 schedule=np.load(O/f'configs/schedule_seed{seed}.npy');a={k:torch.tensor(raw[k],device='cuda')for k in['global','object','availability','cat']};torch.manual_seed(261019800+seed);model=Head().cuda();initial=th(model.state_dict());assert initial==read(O/f'results/train_object_seed{seed}.json')['initial_hash'];opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.0001);params_o=list(model.identity.parameters())+list(model.objects.parameters());params_d=list(model.drawer.parameters());curve=[];start=time.monotonic()
 for step,ii in enumerate(schedule,1):
  ix=torch.tensor(ii,device='cuda');ob,dr=model(a['object'][ix],a['global'][ix],a['availability'][ix]);y=a['cat'][ix];lo=torch.nn.functional.cross_entropy(ob.reshape(-1,8),y[:,:3].reshape(-1));ld=torch.nn.functional.cross_entropy(dr,y[:,3]);loss=.75*lo+.25*ld;assert torch.isfinite(loss);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(params_o,1.);torch.nn.utils.clip_grad_norm_(params_d,1.);opt.step()
  if step==1 or step%150==0:
   point={'update':step,'loss':float(loss),'object_CE':float(lo),'drawer_CE':float(ld)};curve.append(point);save(S/'results/training_progress.json',{'utc':now(),'seed':seed,**point});print('FIT',seed,step,float(loss),flush=True)
 cp=S/f'models/head_coverage_seed{seed}.pt';torch.save({'state':model.state_dict(),'seed':seed,'updates':1500,'initial_hash':initial},cp);save(S/f'results/train_seed{seed}.json',{'utc':now(),'seed':seed,'updates':1500,'parameters':sum(p.numel()for p in model.parameters()),'initial_hash':initial,'final_hash':th(model.state_dict()),'head_sha256':sha(cp),'schedule_sha256':sha(O/f'configs/schedule_seed{seed}.npy'),'index_mapping_exact':True,'curve':curve,'seconds':time.monotonic()-start});del model,opt,a;torch.cuda.empty_cache()
def predict():
 for seed in[0,1]:
  for arm in['control','coverage']:
   cp=O/f'models/head_object_seed{seed}.pt'if arm=='control'else S/f'models/head_coverage_seed{seed}.pt';model=Head().cuda();model.load_state_dict(torch.load(cp,map_location='cuda',weights_only=True)['state']);model.eval()
   for split in['old_band','gap','externalC']:
    file=O/f'data/features_seed{seed}_externalC.npz'if split=='externalC'else S/f'data/features_seed{seed}_{split}.npz';raw=dict(np.load(file));a={k:torch.tensor(raw[k],device='cuda')for k in['global','object','availability']}
    with torch.inference_mode():prob=probabilities(model(a['object'],a['global'],a['availability'])).cpu().numpy()
    np.savez_compressed(S/f'data/predictions_{arm}_seed{seed}_{split}.npz',prob=prob,**{k:raw[k]for k in['root_id','episode_id','step','cat','availability']},valid=raw.get('valid',np.ones_like(raw['cat'],bool)))
   del model;torch.cuda.empty_cache()
def numerical_control_preflight():
 namespace={'P':P,'B':B,'read':read,'MEMBERS':read(O/'configs/protocol.json')['members'],'Episodes':Episodes,'Path':Path}
 node=next(n for n in ast.parse((O/'source/run.py').read_text()).body if isinstance(n,ast.ClassDef)and n.name=='Data');exec(compile(ast.Module(body=[node],type_ignores=[]),'<immutable old Data>','exec'),namespace);data=namespace['Data']('train');route=dict(np.load(O/'data/routes_train.npz'));lookup={(e,int(h)):i for i,(e,h)in enumerate(zip(route['episode_id'],route['step']))};results=[]
 for seed in[0,1]:
  cp=B/f'cells/roots40/models/joint_seed{seed}/final6000.pt';model=WorldModel(skill_count=6,event_count=3).cuda();model.load_state_dict(torch.load(cp,map_location='cuda',weights_only=True)['model']);model.eval().requires_grad_(False);cap={};hook=model.spatial.register_forward_hook(lambda m,i,o:cap.update(tokens=o));batch=data.batch([0,1,2,3],'cuda');old=dict(np.load(O/f'data/features_seed{seed}_train.npz'))
  with torch.inference_mode():
   with torch.autocast('cuda',dtype=torch.bfloat16):
    g=model.encode(batch['tokens'],batch['geometry'],batch['visible'],batch['robot']);spatial=cap.pop('tokens').reshape(4,7,513,256)[:,:5];ix=np.array([[lookup[data.episodes[j]['episode_id'],h]for h in range(5)]for j in range(4)]);w=torch.tensor(route['weights'][ix],device='cuda',dtype=torch.float32);pool=torch.einsum('bhot,bhtd->bhod',w,spatial.float())/w.sum(-1,keepdim=True).clamp_min(1);obj=model.observation(pool).float()*(torch.tensor(route['availability'][ix],device='cuda')>0)[...,None]
   gf=g[:,:5].float().reshape(20,512).cpu().numpy();of=obj.reshape(20,3,512).cpu().numpy();assert np.array_equal(gf,old['global'][:20])and np.array_equal(of,old['object'][:20]),'Frozen numerical control feature mismatch'
   head=Head().cuda();head.load_state_dict(torch.load(O/f'models/head_object_seed{seed}.pt',map_location='cuda',weights_only=True)['state']);head.eval();prob=probabilities(head(torch.tensor(old['object'],device='cuda'),torch.tensor(old['global'],device='cuda'),torch.tensor(old['availability'],device='cuda'))).cpu().numpy();ref=np.load(O/f'data/predictions_object_seed{seed}_train.npz')['prob'];assert np.array_equal(prob,ref),'Old control prediction not bit-identical'
  results.append({'seed':seed,'encoder_boundaries_exact':20,'control_prediction_boundaries_exact':600,'max_feature_difference':0.,'max_probability_difference':0.});hook.remove();del model,head,batch;torch.cuda.empty_cache()
 save(S/'results/numerical_control_preflight.json',{'utc':now(),'passed':True,'results':results,'extra_fits':0})
def main():
 pa=argparse.ArgumentParser();pa.add_argument('--phase',choices=['train','evaluation'],required=True);args=pa.parse_args();sys.addaudithook(guard);lock=(P/'runs/gpu.lock').open('r');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);start=time.monotonic();torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('high')
 if args.phase=='train':
  save(S/'configs/implementation_frozen.json',{'utc':now(),'before_any_fit':True,'source_sha256':{str(S/'source'/name):sha(S/'source'/name)for name in ['common.py','compute.py','collect.py','control_audit.py','training_audit.py','analyze.py','preflight.py','preflight_feasible.py','preflight_joint.py']},'torch_deterministic_algorithms':True,'float32_matmul_precision':'high','num_threads':4,'same_as_control':True})
  assert not(S/'configs/heads_frozen.json').exists();m=read(S/'data/data_manifest.json')['records'];assert sum(r['split']=='train'for r in m)==480 and not any(r['split']!='train'for r in m);assert not(S/'results/BLOCKED_COLLECTION.json').exists();numerical_control_preflight();cache(['train']);routing('train')
  for seed in[0,1]:extract(seed,'train');train(seed)
  heads=[S/f'models/head_coverage_seed{seed}.pt'for seed in[0,1]];save(S/'configs/heads_frozen.json',{'utc':now(),'sha256':{str(p):sha(p)for p in heads},'new_head_updates':3000,'extra_control_fits':0,'new_RSSM_updates':0,'eval_collected':False})
 else:
  for p,h in read(S/'configs/heads_frozen.json')['sha256'].items():assert sha(p)==h
  cache(['old_band','gap'])
  for split in['old_band','gap']:
   routing(split)
   for seed in[0,1]:extract(seed,split)
  predict()
 save(S/f'results/compute_{args.phase}_complete.json',{'utc':now(),'seconds':time.monotonic()-start,'peak_cuda_GiB':torch.cuda.max_memory_allocated()/1024**3})
if __name__=='__main__':main()
