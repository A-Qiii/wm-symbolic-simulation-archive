from common import *
import time,fcntl,signal,re,collections,traceback
from wm_symbolic.world_model import WorldModel
from wm_symbolic.dataset import Episodes
from wm_symbolic.vision_cache import VisibleRegistry
from wm_symbolic.public_geometry import colour_masks
MEMBERS=read(S/'configs/protocol.json')['members']
STAGE='preflight'
def scope_guard(event,args):
 if event!='open' or not isinstance(args[0],(str,bytes)):return
 p=Path(os.fsdecode(args[0])).resolve()
 if p.is_relative_to(P/'data/core1200_v1'):
  match=re.search(r'([ABCD]_core1200_[0-9]{2})',str(p))
  allowed=set(MEMBERS['train'])|set(MEMBERS['validation']) if STAGE=='preflight' else set(MEMBERS['train']) if STAGE=='train' else set().union(*map(set,MEMBERS.values()))
  if match and match.group(1)not in allowed:raise RuntimeError('Sample forbidden in '+STAGE+': '+str(p))
 if STAGE=='train' and p.parent==S/'data' and any(k in p.name for k in ['_test','_externalC','_validation']):raise RuntimeError('Head training cannot read heldout arrays')
class Data(Episodes):
 def __init__(self,split):
  self.project=P;self.split=split;self.event_count=3;self.skills=read(P/'runs/core1200_v1/protocol.json')['skills'];self.stats=read(B/'cells/roots40/configs/normalization.json')
  ids=set(MEMBERS[split]);rows=[r for r in read(P/'data/core1200_v1/data_manifest.json')['records']if r['root_id']in ids]
  self.cache={(x['episode_id'],x['step']):x['path']for x in read(P/'data/core1200_v1/features/index.json')['items']if x['root_id']in ids};by={};self.episodes=[]
  for row in rows:by.setdefault(row['episode_id'],[]).append(row)
  for eid,rr in sorted(by.items()):
   rr.sort(key=lambda r:r['step']);assert [r['step']for r in rr]==[0,1,2,3]
   self.episodes.append({'episode_id':eid,'root_id':rr[0]['root_id'],'rows':rr,'labels':[read(r['labels_path'])for r in rr]})
  self.root_ids=sorted(ids);self.transitions=len(rows);self.matched=list(range(len(self.episodes)))
def public_routes(frames,cache_ids,valid):
 candidates=[set()for _ in COLORS]
 for f in frames:
  for cam in ['static','gripper']:
   seg=VisibleRegistry.grouped(f['seg_'+cam]);pixel_colors=colour_masks(f['rgb_'+cam])
   for uid in np.unique(seg):
    if uid<0:continue
    mask=seg==uid
    if mask.sum()<2:continue
    for color,name in enumerate(COLORS):
     if float(pixel_colors[name][mask].mean())>.5:candidates[color].add(int(uid))
 weights=np.zeros((3,513),bool);slots=np.full(3,-1,np.int64);chosen=np.full(3,-1,np.int64)
 for color,cc in enumerate(candidates):
  if len(cc)!=1:continue
  uid=next(iter(cc))
  if uid not in cache_ids:continue
  slot=cache_ids.index(uid);slots[color]=slot;chosen[color]=uid
  for view in range(8):weights[color,view*64+slot*4:view*64+slot*4+4]=valid[view*64+slot*4:view*64+slot*4+4]
 return weights,slots,chosen,[sorted(x)for x in candidates]
def routing(split):
 begin=time.perf_counter();data=Data(split);index={(x['episode_id'],x['step']):x for x in read(P/'data/core1200_v1/features/index.json')['items']if x['root_id']in MEMBERS[split]};rows=[];audit=[];wrong=[];geometry_max=0.
 for ep in data.episodes:
  historical=VisibleRegistry()
  for h in range(5):
   item=index[ep['episode_id'],h]
   with np.load(item['path'])as z:cache=dict(z)
   frames=[]
   for path in item['public_inputs']:
    with np.load(path)as z:frames.append(dict(z))
   assert historical.update(frames)==cache['visible_registry_ids'].tolist()
   ids=cache['visible_registry_ids'].tolist()
   for t,f in enumerate(frames):
    for vi,cam in enumerate(['static','gripper']):
     seg=VisibleRegistry.grouped(f['seg_'+cam]);height,width=seg.shape;top=(224-height)//2;left=(224-width)//2
     for slot in range(12):
      mask=seg==ids[slot];yy,xx=np.where(mask);vis=len(yy)>=2
      box=[(xx.min()+left)/224,(yy.min()+top)/224,(xx.max()+left+1)/224,(yy.max()+top+1)/224,float(vis)]if vis else[0,0,1/224,1/224,0]
      start=(t*2+vi)*64+slot*4;g=cache['geometry'][start:start+4]
      geometry_max=max(geometry_max,float(np.abs(g[:,:5]-np.asarray(box,np.float32)).max()))
      assert np.all(cache['valid'][start:start+4]==vis) and np.all(g[:,6]==vi) and np.all(g[:,7]==np.float32(t/3))
   assert np.array_equal(cache['robot'],frames[-1]['robot_obs'].astype(np.float32))
   weights,slots,chosen,candidates=public_routes(frames,ids,cache['valid'])
   # Private identities enter only the following audit, never public_routes or its saved model inputs.
   r=ep['rows'][max(h-1,0)];info=read(Path(r['labels_path']).parent/'endpoint_info_private.json')['before'if h==0 else'after']
   for color,c in enumerate(COLORS):
    uid=info['scene_info']['movable_objects']['block_'+c]['uid'];visible=any(int((VisibleRegistry.grouped(f['seg_'+cam])==uid).sum())>=2 for f in frames for cam in ['static','gripper'])
    correct=chosen[color]==uid;missing=not weights[color].any()
    record={'episode_id':ep['episode_id'],'root_id':ep['root_id'],'h':h,'color':c,'candidate_uids':candidates[color],'chosen_public_uid':int(chosen[color]),'audit_expected_uid':uid,'visible':visible,'correct_mapping':bool(correct),'missing_representation':bool(missing),'tokens':int(weights[color].sum())}
    audit.append(record)
    if chosen[color]>=0 and not correct:wrong.append(record)
   rows.append({'episode_id':ep['episode_id'],'root_id':ep['root_id'],'step':h,'weights':weights,'slots':slots,'availability':weights.sum(1).astype(np.float32)/32})
 print('ROUTE_PROGRESS',split,len(rows),flush=True)
 np.savez_compressed(S/f'data/routes_{split}.npz',**{k:np.asarray([r[k]for r in rows])for k in rows[0]})
 stats={}
 for c in COLORS:
  q=[a for a in audit if a['color']==c];visible=sum(a['visible']for a in q);miss=sum(a['visible']and a['missing_representation']for a in q)
  stats[c]={'object_boundary_records':len(q),'visible':visible,'unresolved_visible':miss,'unresolved_visible_rate':miss/max(visible,1),'invisible':sum(not a['visible']for a in q),'zero_fallback':sum(a['missing_representation']for a in q),'assigned_correct':sum(a['correct_mapping']for a in q)}
 limit=.05 if split=='train'else .10
 passed=not wrong and geometry_max==0 and all(v['unresolved_visible_rate']<=limit for v in stats.values())
 save(S/f'results/routing_{split}.json',{'utc':now(),'split':split,'current_four_frames_only':True,'records':len(rows),'independent_roots':len(data.root_ids),'stats':stats,'wrong_assignments':wrong,'geometry_max_difference':geometry_max,'passed':passed,'seconds':time.perf_counter()-begin,'audit':audit})
 print('ROUTE_GATE',split,json.dumps({'passed':passed,'stats':stats,'wrong_assignments':len(wrong),'seconds':time.perf_counter()-begin}),flush=True)
 assert passed,'Routing gate failed; no GT repair or silent exclusion permitted'
def extract(seed,split):
 start=time.perf_counter();data=Data(split);route=dict(np.load(S/f'data/routes_{split}.npz'));lookup={(e,int(h)):i for i,(e,h)in enumerate(zip(route['episode_id'],route['step']))}
 cp=B/f'cells/roots40/models/joint_seed{seed}/final6000.pt';expected=read(B/'configs/all_models_frozen.json')['sha256'][str(cp)];assert sha(cp)==expected
 model=WorldModel(skill_count=6,event_count=3).cuda();model.load_state_dict(torch.load(cp,map_location='cuda',weights_only=True)['model']);model.eval().requires_grad_(False);before=th(model.state_dict());capture={}
 hook=model.spatial.register_forward_hook(lambda mod,inputs,output:capture.update(tokens=output))
 original=dict(np.load(B/f'cells/roots40/data/features_joint_seed{seed}_{split}.npz'));ref={(e,int(h)):i for i,(e,h)in enumerate(zip(original['episode_id'],original['step']))};records=[];maximum=0.
 with torch.inference_mode():
  for off in range(0,len(data.episodes),4):
   indices=list(range(off,min(off+4,len(data.episodes))));batch=data.batch(indices,'cuda');n=len(indices)
   with torch.autocast('cuda',dtype=torch.bfloat16):
    encoded=model.encode(batch['tokens'],batch['geometry'],batch['visible'],batch['robot']);spatial=capture.pop('tokens').reshape(n,7,513,256)[:,:5]
    ix=np.array([[lookup[data.episodes[j]['episode_id'],h]for h in range(5)]for j in indices]);weights=torch.tensor(route['weights'][ix],device='cuda',dtype=torch.float32)
    pooled=torch.einsum('bhot,bhtd->bhod',weights,spatial.float())/weights.sum(-1,keepdim=True).clamp_min(1)
    objects=model.observation(pooled).float();availability=torch.tensor(route['availability'][ix],device='cuda');objects=objects*(availability>0)[...,None]
   global_features=encoded[:,:5].float().cpu().numpy();object_features=objects.cpu().numpy()
   for j,epidx in enumerate(indices):
    ep=data.episodes[epidx]
    for h in range(5):
     eid=ep['episode_id'];i=lookup[eid,h];reference=original['x'][ref[eid,h]];maximum=max(maximum,float(np.abs(reference-global_features[j,h]).max()))
     assert np.array_equal(reference,global_features[j,h]),'Frozen global E does not exactly reproduce original features'
     label=ep['labels'][0]['before']if h==0 else ep['labels'][h-1]['after'];assert all(label['categorical_valid'][:4])
     records.append({'episode_id':eid,'root_id':ep['root_id'],'step':h,'global':global_features[j,h],'object':object_features[j,h],'availability':route['availability'][i],'cat':np.array(label['categorical'][:4],np.int64)})
   print('EXTRACT',seed,split,off+len(indices),'/',len(data.episodes),flush=True)
 hook.remove();assert th(model.state_dict())==before and sha(cp)==expected
 np.savez_compressed(S/f'data/features_seed{seed}_{split}.npz',**{k:np.asarray([r[k]for r in records])for k in records[0]})
 save(S/f'results/extraction_seed{seed}_{split}.json',{'utc':now(),'records':len(records),'roots':len(data.root_ids),'global_matches_original_exactly':True,'global_max_abs_difference':maximum,'encoder_weights_unchanged':True,'checkpoint_sha256':expected,'seconds':time.perf_counter()-start})
 del model;torch.cuda.empty_cache()
def tensor_inputs(a,arm,indices=None):
 g=a['global']if indices is None else a['global'][indices];av=a['availability']if indices is None else a['availability'][indices]
 focal=(a['object']if indices is None else a['object'][indices])if arm=='object'else g[:,None].expand(-1,3,-1)*(av>0)[...,None]
 return focal,g,av
def train(seed,arm):
 start=time.perf_counter();raw=dict(np.load(S/f'data/features_seed{seed}_train.npz'));roots=sorted(set(raw['root_id']));assert roots==sorted(MEMBERS['train']) and len(raw['cat'])==600
 rng=np.random.default_rng(261019800+seed);groups=[np.where(raw['root_id']==r)[0]for r in roots];schedule=np.array([[groups[j][rng.integers(15)]for j in rng.integers(len(groups),size=BATCH)]for _ in range(UPDATES)])
 sp=S/f'configs/schedule_seed{seed}.npy'
 if sp.exists():assert np.array_equal(np.load(sp),schedule)
 else:np.save(sp,schedule)
 a={k:torch.tensor(raw[k],device='cuda')for k in ['global','object','availability','cat']};torch.manual_seed(261019800+seed);model=Head().cuda();initial=th(model.state_dict());opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.0001);curve=[]
 params_o=list(model.identity.parameters())+list(model.objects.parameters());params_d=list(model.drawer.parameters())
 for step,ii in enumerate(schedule,1):
  ix=torch.tensor(ii,device='cuda');ob,dr=model(*tensor_inputs(a,arm,ix));y=a['cat'][ix]
  lo=torch.nn.functional.cross_entropy(ob.reshape(-1,8),y[:,:3].reshape(-1));ld=torch.nn.functional.cross_entropy(dr,y[:,3]);loss=.75*lo+.25*ld
  assert torch.isfinite(loss);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(params_o,1.);torch.nn.utils.clip_grad_norm_(params_d,1.);opt.step()
  if step==1 or step%150==0:
   point={'update':step,'total':float(loss),'object_CE':float(lo),'drawer_CE':float(ld)};curve.append(point);print('FIT',seed,arm,step,round(point['total'],5),flush=True)
 path=S/f'models/head_{arm}_seed{seed}.pt';torch.save({'state':model.state_dict(),'seed':seed,'arm':arm,'updates':UPDATES,'initial_hash':initial},path)
 rec={'utc':now(),'seed':seed,'arm':arm,'updates':UPDATES,'parameters':sum(p.numel()for p in model.parameters()),'initial_hash':initial,'final_hash':th(model.state_dict()),'drawer_hash':th(model.drawer.state_dict()),'schedule_sha256':sha(sp),'train_roots':40,'train_boundaries':600,'seconds':time.perf_counter()-start,'curve':curve,'weights_sha256':sha(path)}
 save(S/f'results/train_{arm}_seed{seed}.json',rec);del model,opt,a;torch.cuda.empty_cache();return rec
def evaluate():
 frozen=read(S/'configs/all_heads_frozen.json')
 for p,h in frozen['sha256'].items():assert sha(p)==h
 checks=[]
 for seed in [0,1]:
  for arm in ARMS:
   cp=torch.load(S/f'models/head_{arm}_seed{seed}.pt',map_location='cuda',weights_only=True);m=Head().cuda();m.load_state_dict(cp['state']);m.eval().requires_grad_(False)
   for split in SPLITS:
    raw=dict(np.load(S/f'data/features_seed{seed}_{split}.npz'));a={k:torch.tensor(raw[k],device='cuda')for k in ['global','object','availability']}
    with torch.inference_mode():
     focal,g,av=tensor_inputs(a,arm);out=m(focal,g,av);prob=probabilities(out).cpu().numpy();n=len(prob)
     rng=np.random.default_rng(261019999);perms=np.array([rng.permutation(3)for _ in range(n)]);pi=torch.tensor(perms,device='cuda');bi=torch.arange(n,device='cuda')[:,None]
     reordered=m(focal[bi,pi],g,av[bi,pi],pi);expected=out[0][bi,pi];diff=float((reordered[0]-expected).abs().max());scene_diff=float((reordered[1]-out[1]).abs().max());assert diff<=1e-6 and scene_diff==0
    meta={k:raw[k]for k in ['episode_id','root_id','step','cat','availability']};np.savez_compressed(S/f'data/predictions_{arm}_seed{seed}_{split}.npz',prob=prob,**meta)
    checks.append({'seed':seed,'arm':arm,'split':split,'records':n,'max_abs_logit_permutation_difference':diff,'scene_difference':scene_diff,'passed':True})
   del m,a;torch.cuda.empty_cache()
 save(S/'results/permutation.json',{'utc':now(),'tests':checks,'meaning':'Output follows jointly permuted object bundle, requested identity and availability. Tests index handling/shared-head equivariance; not semantic color-swap robustness or accuracy.'})
 for seed in [0,1]:
  for split in SPLITS:
   a=dict(np.load(S/f'data/predictions_global_seed{seed}_{split}.npz'));b=dict(np.load(S/f'data/predictions_object_seed{seed}_{split}.npz'));assert np.array_equal(a['prob'][:,24:],b['prob'][:,24:]),'Scene branch not identical'
 print('ALL_PREDICTIONS_COMPLETE',flush=True)
def main():
 global STAGE
 started=time.perf_counter();signal.signal(signal.SIGALRM,lambda *_:(_ for _ in()).throw(TimeoutError('45-minute compute budget exceeded')));signal.alarm(45*60)
 assert not(S/'configs/all_heads_frozen.json').exists(),'Do not silently repeat completed fits'
 receipt=read(S/'configs/preregistration_receipt.json');assert sha(S/'configs/protocol.json')==receipt['protocol_sha256'];protected_seals()
 sys.addaudithook(audit_writes);sys.addaudithook(scope_guard)
 save(S/'configs/implementation_frozen.json',{'utc':now(),'before_any_training':True,'protocol_sha256':receipt['protocol_sha256'],'routing_amendment_sha256':sha(S/'configs/routing_amendment.json'),'source_sha256':{str(p):sha(p)for p in(S/'source').glob('*.py')}})
 for split in ['train','validation']:routing(split)
 save(S/'results/preflight.json',{'utc':now(),'passed':True,'parameters_per_head':309165,'routing_splits_before_fit':['train','validation'],'test_C_samples_before_fits':0,'public_route_current_four_frames':True,'head_updates':1500,'total_head_fits':4})
 print('PREFLIGHT_PASSED',now(),flush=True)
 assert torch.cuda.is_available(),'Local CUDA unavailable';lock=(P/'runs/gpu.lock').open('r');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('high')
 STAGE='train'
 for seed in [0,1]:extract(seed,'train')
 fits=[]
 for seed in [0,1]:
  for arm in ARMS:fits.append(train(seed,arm))
  pair=fits[-2:];assert pair[0]['initial_hash']==pair[1]['initial_hash'] and pair[0]['schedule_sha256']==pair[1]['schedule_sha256'] and pair[0]['drawer_hash']==pair[1]['drawer_hash']
 save(S/'configs/all_heads_frozen.json',{'utc':now(),'before_heldout_read':True,'sha256':{str(p):sha(p)for p in(S/'models').glob('head_*.pt')}})
 print('ALL_FOUR_HEADS_FROZEN',now(),flush=True);STAGE='evaluate'
 for split in ['test','externalC']:routing(split)
 for seed in [0,1]:
  for split in ['validation','test','externalC']:extract(seed,split)
 evaluate();after=protected_seals();before=read(S/'configs/protected_before.json');assert after==before
 save(S/'results/compute_complete.json',{'utc':now(),'status':'COMPLETED_FIXED_PROTOCOL','wall_seconds':time.perf_counter()-started,'head_fits':4,'head_updates_total':6000,'full_RSSM_updates':0,'encoder_weights_unchanged':True,'protected_files_unchanged':len(after['sha256']),'protocol_sha256':sha(S/'configs/protocol.json'),'gpu_peak_allocated_GiB':torch.cuda.max_memory_allocated()/1024**3,'test_C_after_all_heads_frozen':True,'all_global_E_features_exact_original':True,'all_drawer_probabilities_identical':True,'no_new_collection':True,'no_upload':True,'fits':fits})
 signal.alarm(0);print('COMPUTE_COMPLETE',round(time.perf_counter()-started,3),flush=True)
if __name__=='__main__':
 class Tee:
  def __init__(self,stream,log):self.stream,self.log=stream,log
  def write(self,s):self.stream.write(s);self.log.write(s);return len(s)
  def flush(self):self.stream.flush();self.log.flush()
 logfile=(S/'logs/run.log').open('a',buffering=1)
 sys.stdout=Tee(sys.stdout,logfile);sys.stderr=Tee(sys.stderr,logfile)
 try:main()
 except BaseException as exc:
  save(S/'results/failure.json',{'utc':now(),'stage':STAGE,'error':str(exc),'traceback':traceback.format_exc()});raise
