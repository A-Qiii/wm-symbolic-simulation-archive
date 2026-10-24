from pathlib import Path
import os,sys,json,hashlib,datetime,time,fcntl,csv
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';os.environ['PYTHONDONTWRITEBYTECODE']='1';os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import numpy as np,torch
T=Path(__file__).resolve().parent.parent;P=T.parent.parent;S=P/'experiments/data_expansion_training_20261021';sys.path.insert(0,str(T/'source'))
from models import ObjectAug,Interface,history

def read(p):return json.loads(p.read_text())
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  for q in iter(lambda:f.read(4194304),b''):h.update(q)
 return h.hexdigest()
def statehash(m):
 h=hashlib.sha256()
 for k,v in sorted(m.state_dict().items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def guard(e,args):
 if e=='socket.connect':raise RuntimeError('Extractionoffline')
 if e!='open'or not isinstance(args[0],(str,bytes)):return
 p=Path(os.fsdecode(args[0])).resolve();mode=args[1];flags=args[2]if len(args)>2 else 0;writing=(isinstance(mode,str)and any(c in mode for c in 'wa+'))or(isinstance(flags,int)and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
 if writing and p.is_relative_to(P)and not p.is_relative_to(T):raise RuntimeError('Protected original write '+str(p))
sys.addaudithook(guard)
torch.set_num_threads(4);torch.set_float32_matmul_precision('high');torch.use_deterministic_algorithms(True)
lock=(P/'runs/gpu.lock').open('r');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);cfg=read(T/'configs/protocol.json');frozen=read(S/'configs/models_frozen.json')['sha256'];inputhash={};stats=[]
for seed in cfg['model_seeds']:
 checkpoint=Path(cfg['checkpoints'][str(seed)]);assert sha(checkpoint)==frozen[str(checkpoint)];inputhash[str(checkpoint)]=sha(checkpoint);m=ObjectAug(Interface()).cuda();ck=torch.load(checkpoint,map_location='cuda',weights_only=False);m.load_state_dict(ck['model']);assert sum(p.numel()for p in m.parameters())==2490068;m.eval().requires_grad_(False);before=statehash(m);rows=[];hidden=[];contexts=[];stateall=[];maxdiff=0.;equal=0;started=time.time()
 for split in cfg['cohorts']:
  paths=[S/f'cache/{split}_seed{seed}.npz',S/f'cache/{split}_geometry_raw.npz',S/f'cache/{split}_metadata.json',S/'configs/input_normalization.json',S/f'results/predictions_{split}_new868_seed{seed}.npz']
  for p in paths:inputhash[str(p)]=sha(p)
  base=dict(np.load(paths[0]));raw=dict(np.load(paths[1]));norm=read(paths[3])['values'][f'geometry_seed{seed}'];meta=read(paths[2]);assert len(meta)==80 and all(e['length']==6 for e in meta)
  for name,a in raw.items():
   n=norm[name];v=(a-np.array(n['mean'],np.float32))/np.array(n['std'],np.float32);width=8 if name=='objects'else 4
   for j in range(0,a.shape[-1],width):v[...,j:j+width]*=a[...,j+width-1:j+width]>0
   base[name]=np.concatenate([base[name],v.astype(np.float32)],-1)
  data={k:torch.tensor(base[k],device='cuda')for k in ['objects','context','availability','verb','pointer','budget']};saved=dict(np.load(paths[4]));idx={(str(e),int(o),str(md),int(t)):i for i,(e,o,md,t)in enumerate(zip(saved['episode_id'],saved['origin'],saved['mode'],saved['relative_step']))};entries=[(i,s)for i in range(80)for s in range(6)]
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
   for begin in range(0,len(entries),8):
    part=entries[begin:begin+8];ids=[i for i,s in part];starts=torch.tensor([s for i,s in part],device='cuda');q={}
    for k,v in data.items():
     full=v[ids];ind=(starts[:,None]+torch.arange(v.shape[1],device='cuda')[None]).clamp_max(v.shape[1]-1);q[k]=full[torch.arange(len(part),device='cuda')[:,None],ind]
    posts,_=history(m,q['objects'],q['context'],q['availability'],q['verb'],q['pointer'],q['budget'])
    for j,(i,origin)in enumerate(part):
     if origin!=0:continue
     e=meta[i]
     for step in range(1,7):
      state=posts[step][j:j+1];sym=m.symbols(state);pr=torch.cat([sym['objects'].float().softmax(-1).flatten(1),sym['drawer'].float().softmax(-1),sym['slider'].float().softmax(-1),sym['lights'].float().sigmoid()],-1)[0].cpu().numpy();ref=saved['prob'][idx[e['episode_id'],0,'current',step]];delta=float(np.max(np.abs(pr-ref)));maxdiff=max(maxdiff,delta);equal+=int(np.array_equal(pr,ref));assert np.allclose(pr,ref,rtol=0,atol=1e-5),(split,e['episode_id'],step,delta)
      hs=state[0].float().cpu().numpy();contexts.append(hs[3]);stateall.append(hs)
      for ob,col in enumerate(['red','blue','pink']):
       h=hs[ob];assert h.shape==(128,)and np.isfinite(h).all();hidden.append(h);cat=int(base['cat'][i,step,ob]);valid=bool(base['cat_valid'][i,step,ob]);probIn=float(ref[ob*8+3]);probHeld=float(ref[ob*8+6]);trueIn=cat==3;predIn=probIn>=.5;err='invalid'if not valid else'TP'if trueIn and predIn else'FN'if trueIn else'FP'if predIn else'TN';rows.append({'point_id':len(rows),'model_seed':seed,'split':split,'episode_id':e['episode_id'],'root_id':e['root_id'],'family_id':e['family_id'],'step':step,'object':col,'object_index':ob,'latent_type':'object_GRU_observed_h128','true_category':cat,'true_In':int(trueIn),'true_Held':int(cat==6),'true_object_valid':int(valid),'pred_In':int(predIn),'pred_Held':int(probHeld>=.5),'prob_In_annotation_only':probIn,'prob_Held_annotation_only':probHeld,'In_error_type':err,'drawer_true':int(base['cat'][i,step,3]),'drawer_valid':int(base['cat_valid'][i,step,3]),'drawer_pred':int(ref[24:27].argmax()),'skill_just_executed':e['rows'][step-1]['skill'],'observed_update':'history_after_actual_observation'})
  del data,base,raw,saved;torch.cuda.empty_cache();print('EXTRACTED',seed,split,len(hidden),flush=True)
 assert len(rows)==2880 and len(contexts)==960 and statehash(m)==before and sha(checkpoint)==frozen[str(checkpoint)]
 np.savez_compressed(T/f'data/observed_latents_seed{seed}.npz',object_h=np.stack(hidden),context_c=np.stack(contexts),full_nodes=np.stack(stateall))
 with(T/f'data/samples_seed{seed}.csv').open('w',newline='')as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 save(T/f'data/samples_seed{seed}.json',rows);stats.append({'seed':seed,'object_points':len(rows),'object_dimensions':128,'context_boundaries':len(contexts),'root_states_excluded':True,'labels_in_inference_inputs':False,'model_state_hash_before_after':before,'checkpoint_sha256':sha(checkpoint),'saved_posterior_probabilities_compared':960,'posterior_probability_arrays_bitwise_equal':equal,'max_abs_probability_difference':maxdiff,'seconds':time.time()-started});del m;torch.cuda.empty_cache()
for p,h in inputhash.items():assert sha(Path(p))==h,p
save(T/'results/input_hashes.json',inputhash);save(T/'results/extraction.json',stats);print('EXTRACTION_COMPLETE',stats,flush=True)
