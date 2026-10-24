from pathlib import Path
import sys,os,json,hashlib,datetime,time,csv,collections,itertools,fcntl
import numpy as np
P=Path('/home/robot/research/projects/wm_symbolic');S=P/'experiments/failure_recovery_control_20261019';D=P/'experiments/initial_state_audit_20261019';B=P/'experiments/rssm_scaling_run_20261019';CELL=B/'cells/roots40'
sys.path.insert(0,str(S/'source'));os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';os.environ['MPLCONFIGDIR']=str(D/'cache');os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
from common import read,sha,save,csvwrite,rel,truth,correct,statehash,COLORS
from dataset import NewEpisodes
from old_dataset import CoreEpisodes
from wm_symbolic.vision_cache import VisibleRegistry
from wm_symbolic.world_model import WorldModel
import torch
def guard():
 def audit(event,args):
  if event=='socket.connect':raise RuntimeError('Offline audit')
  if event!='open'or not isinstance(args[0],(str,bytes)):return
  p=Path(os.fsdecode(args[0])).resolve();mode=args[1];flags=args[2]if len(args)>2 else 0
  if not p.is_relative_to(P):return
  writing=(isinstance(mode,str)and any(c in mode for c in 'wa+'))or(isinstance(flags,int)and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
  if writing and not p.is_relative_to(D):raise RuntimeError('Read-only prior results '+str(p))
  if any(t in str(p)for t in ['externalC','C_core1200','D_core1200','independent_identity_D','independent_objective_D','calvin_scene_C','calvin_scene_D']):raise RuntimeError('C/D forbidden')
 sys.addaudithook(audit)
def probs(m,state):
 p=m.symbols(state);return torch.cat([p['objects'].float().softmax(-1).flatten(-2),p['drawer'].float().softmax(-1),p['slider'].float().softmax(-1),p['lights'].float().sigmoid()],-1).mean(0).cpu().numpy()
def predname(p):
 bits=rel(p)[:3]>=.5;return '+'.join(c for c,bit in zip(COLORS,bits)if bit)or'empty'
def main():
 start=time.monotonic();guard();D.mkdir(exist_ok=True);(D/'results').mkdir(exist_ok=True);(D/'figures').mkdir(exist_ok=True);norm=read(CELL/'configs/normalization.json');old=CoreEpisodes(P,'train',norm);new={s:NewEpisodes(s)for s in ['train','development','evaluation']};roots_old={r['root_id']:r for r in read(P/'data/core1200_v1/roots_manifest.json')if r['root_id']in old.root_ids};roots_new={r['root_id']:r for r in read(S/'data/roots_manifest.json')};rootrows=[];inputrows=[];pairs=[];refs={str(S/'MANIFEST.json'):sha(S/'MANIFEST.json'),str(S/'SEALED.json'):sha(S/'SEALED.json'),str(CELL/'configs/normalization.json'):sha(CELL/'configs/normalization.json')};sources={}
 for scope,data in [('old_train',old)]+[('new_'+k,d)for k,d in new.items()]:
  for ix,ep in enumerate(data.episodes):
   if not ep['episode_id'].endswith('_b0'):continue
   rid=ep['root_id'];recipe=(roots_old if scope=='old_train'else roots_new)[rid];l=ep['labels'][0]['before'];cat=l['categorical'];held=next((c for j,c in enumerate(COLORS)if cat[j]==6),'empty');ins=[c for j,c in enumerate(COLORS)if cat[j]==3];d=Path(ep['rows'][0]['obs_before']).parent.parent;paths=[d/f'root_observation_{j}.npz'for j in range(4)];frames=[dict(np.load(p))for p in paths];f=dict(np.load(data.cache[ep['episode_id'],0]));reg=VisibleRegistry();ids=reg.update(frames);assert np.array_equal(ids,f['visible_registry_ids']);assert np.array_equal(frames[-1]['robot_obs'].astype(np.float32),f['robot']);x=data.get(ix);assert np.array_equal(x['tokens'][0].numpy(),(f['tokens'].astype(np.float32)-np.array(norm['vision_mean'],np.float32))/np.array(norm['vision_std'],np.float32));assert x['tokens'].shape[0]==7
   pos=l['continuous']['objects'][held]['position_m']if held!='empty'else [None]*3
   rootrows.append({'scope':scope,'root_id':rid,'environment':recipe['environment'],'held':held,'InDrawer':','.join(ins)or'none','drawer':int(cat[3]),'held_x':pos[0],'held_y':pos[1],'held_z':pos[2],'planned_setup':','.join(recipe['setup_skills']),'actual_cat':cat,'held_contacts':l['continuous']['objects'][held]['finger_contacts']if held!='empty'else[]})
   tokens=x['tokens'][0].numpy()[f['valid']];robot=x['robot'][0].numpy();visible={};means={}
   for color,uid in reg.colour_ids.items():
    for cam in ['static','gripper']:
     mask=reg.grouped(frames[-1]['seg_'+cam])==uid;visible[COLORS[color]+'_'+cam]=int(mask.sum());means[COLORS[color]+'_'+cam]=frames[-1]['rgb_'+cam][mask].mean(0).tolist()if mask.any()else[]
   inputrows.append({'scope':scope,'root_id':rid,'held':held,'registry_ids':ids,'colour_uid':{COLORS[k]:v for k,v in reg.colour_ids.items()},'colour_slots':{COLORS[k]:ids.index(v)for k,v in reg.colour_ids.items()},'visible_pixels':visible,'meanRGB':means,'vision_abs_p99':float(np.quantile(abs(tokens),.99)),'vision_abs_max':float(abs(tokens).max()),'robot_abs_max':float(abs(robot).max()),'robot_normalized':robot.tolist(),'finite':bool(np.isfinite(tokens).all()and np.isfinite(robot).all()),'frame_shapes':{k:list(frames[-1][k].shape)for k in ['rgb_static','rgb_gripper','robot_obs']},'calibration_sha256':hashlib.sha256(b''.join(frames[-1][k].tobytes()for k in sorted(frames[-1])if k.startswith('calib_'))).hexdigest(),'frames_distinct':len({hashlib.sha256(fr['rgb_static'].tobytes()+fr['rgb_gripper'].tobytes()+fr['robot_obs'].tobytes()).hexdigest()for fr in frames})})
   sources[rid]={'scope':scope,'data':data,'ix':ix,'recipe':recipe,'ep':ep}
   for p in paths+[Path(data.cache[ep['episode_id'],0]),Path(ep['rows'][0]['labels_path'])]:refs[str(p)]=sha(p)
   if scope!='old_train':
    for branch in [1,2]:
     other=d.with_name(rid+f'_b{branch}');ok=[]
     for j,frame in enumerate(frames):
      z=dict(np.load(other/f'root_observation_{j}.npz'));ok.append(set(z)==set(frame)and all(np.array_equal(z[k],frame[k])for k in z))
     a=dict(np.load(d/'root_reset_private.npz'));b=dict(np.load(other/'root_reset_private.npz'));stateexact=all(np.array_equal(a[k],b[k])for k in a);lab=read(next(p for p in other.glob('00_*/labels_private.json')))['before'];pairs.append({'root_id':rid,'branch':branch,'all_h0_pixels_robot_calibration_exact':all(ok),'physical_state_exact':stateexact,'initial_labels_exact':lab==l})
 save(D/'results/root_inventory.json',rootrows);csvwrite(D/'results/root_inventory.csv',rootrows);save(D/'results/input_audit.json',inputrows);save(D/'results/pair_audit.json',pairs);assert all(r['all_h0_pixels_robot_calibration_exact']and r['physical_state_exact']and r['initial_labels_exact']for r in pairs);print('INPUT_AUDIT',len(rootrows),len(pairs),flush=True)
 # Root h0 for missing new-train and post-continuation old-train predictions. Existing dev/eval are reused.
 lock=(P/'runs/gpu.lock').open('r');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('high');outputs=[];lengthchecks=[];seal=read(S/'configs/models_frozen.json')
 for seed in [0,1]:
  for arm in ['original','nominal','recovery']:
   path=CELL/f'models/joint_seed{seed}/final6000.pt'if arm=='original'else Path(next(r['path']for r in seal['selection']if r['seed']==seed and r['arm']==arm));refs[str(path)]=sha(path);m=WorldModel(skill_count=6,event_count=3).cuda();cp=torch.load(path,map_location='cuda',weights_only=False);m.load_state_dict(cp['model']);m.eval().requires_grad_(False);before=statehash(m)
   with torch.inference_mode():
    for rid,r in sources.items():
     if r['scope']not in ['old_train','new_train']:continue
     b=r['data'].batch([r['ix']],'cuda');rng=81831+int(r['recipe']['seed']);torch.manual_seed(rng)
     with torch.autocast('cuda',dtype=torch.bfloat16):
      e=m.encode(b['tokens'],b['geometry'],b['visible'],b['robot']);h,q=m.initial(e[:,0].repeat_interleave(8,0));p=probs(m,h)
      if rid in ['A_recovery_train_00','A_core1200_02']:
       e1=m.encode(b['tokens'][:,:1],b['geometry'][:,:1],b['visible'][:,:1],b['robot'][:,:1]);torch.manual_seed(rng);h1,q1=m.initial(e1[:,0].repeat_interleave(8,0));p1=probs(m,h1);lengthchecks.append({'seed':seed,'arm':arm,'root_id':rid,'E_max_abs_difference':float((e[:,0]-e1[:,0]).abs().max()),'q_max_abs_difference':float((q-q1).abs().max()),'readout_max_abs_difference':float(abs(p-p1).max()),'held_decision_same':predname(p)==predname(p1),'full_seven_frames_versus_h0_only':True})
     outputs.append({'seed':seed,'arm':arm,'scope':r['scope'],'root_id':rid,'prob':p,'cat':r['ep']['labels'][0]['before']['categorical'],'provenance':'necessary_frozen_h0_forward','rng':rng})
   assert statehash(m)==before;del m,cp;torch.cuda.empty_cache();print('H0_FORWARD',seed,arm,flush=True)
   for split in ['development','evaluation']:
    path=S/f'results/predictions_{arm}_seed{seed}_{split}.npz';refs[str(path)]=sha(path)
    with np.load(path)as z:
     ids=np.where((z['mode']=='current')&(z['step']==0)&(z['branch']==0))[0]
     for i in ids:outputs.append({'seed':seed,'arm':arm,'scope':'new_'+split,'root_id':str(z['root_id'][i]),'prob':z['prob'][i],'cat':z['cat'][i],'provenance':'existing_sealed_prediction','rng':81831+int(roots_new[str(z['root_id'][i])]['seed'])})
 np.savez_compressed(D/'results/h0_predictions.npz',**{k:np.array([r[k]for r in outputs])for k in outputs[0]});save(D/'results/length_checks.json',lengthchecks)
 detail=[];groups=collections.defaultdict(list)
 for r in outputs:
  info=next(i for i in rootrows if i['root_id']==r['root_id']);ok=correct(rel(r['prob']),truth(r['cat']));row={'seed':r['seed'],'arm':r['arm'],'scope':r['scope'],'root_id':r['root_id'],'environment':info['environment'],'held_truth':info['held'],'held_predicted':predname(r['prob']),'held_correct':bool(ok[0]),'InDrawer_correct':bool(ok[1]),'drawer_correct':bool(ok[2]),'joint_correct':bool(ok[3]),'held_prob_red':float(r['prob'][6]),'held_prob_blue':float(r['prob'][14]),'held_prob_pink':float(r['prob'][22]),'provenance':r['provenance']};detail.append(row);groups[r['scope'],r['seed'],r['arm'],'all'].append(row);groups[r['scope'],r['seed'],r['arm'],'held'if info['held']!='empty'else'empty'].append(row)
 stats=[{'scope':scope,'seed':seed,'arm':arm,'subset':subset,'roots':len(rr),**{k:sum(r[k]for r in rr)for k in ['held_correct','InDrawer_correct','drawer_correct','joint_correct']}}for(scope,seed,arm,subset),rr in groups.items()];csvwrite(D/'results/h0_detail.csv',detail);csvwrite(D/'results/h0_summary.csv',stats)
 # Decompose the exact 84 nominal-branch boundaries, including earliest correct/incorrect h0.
 chains=[];boundary=[]
 for seed in [0,1]:
  for arm in ['original','nominal','recovery']:
   with np.load(S/f'results/predictions_{arm}_seed{seed}_evaluation.npz')as z:d={k:z[k]for k in z.files}
   ok=correct(rel(d['prob']),truth(d['cat']));curr=(d['mode']=='current')&(d['branch']<2)
   for eid in sorted(set(d['episode_id'][curr])):
    ids=np.where(curr&(d['episode_id']==eid))[0];assert len(ids)==7;h0ok=bool(ok[ids[0],3]);bad=[i for i in ids if not ok[i,3]];chains.append({'seed':seed,'arm':arm,'episode_id':str(eid),'root_id':str(d['root_id'][ids[0]]),'h0_joint_correct':h0ok,'correct_boundaries':int(ok[ids,3].sum()),'first_wrong_boundary':int(d['step'][bad[0]])if bad else -1,'initial_held_truth':predname(np.array([0]*32))if False else str(truth(d['cat'][ids[0]])[:3].tolist()),'first_wrong_components':','.join(['held','InDrawer','drawer'][j]for j in range(3)if not ok[bad[0],j])if bad else'none'})
    for i in ids:boundary.append({'seed':seed,'arm':arm,'root_id':str(d['root_id'][i]),'episode_id':str(eid),'step':int(d['step'][i]),'h0_correct':h0ok,'held_truth':'+'.join(c for j,c in enumerate(COLORS)if d['cat'][i,j]==6)or'empty','held_predicted':predname(d['prob'][i]),'InDrawer_truth':'+'.join(c for j,c in enumerate(COLORS)if d['cat'][i,j]==3)or'none','InDrawer_predicted':'+'.join(c for j,c in enumerate(COLORS)if rel(d['prob'][i])[3+j]>=.5)or'none','joint_correct':bool(ok[i,3]),'held_correct':bool(ok[i,0]),'InDrawer_correct':bool(ok[i,1]),'drawer_correct':bool(ok[i,2])})
 csvwrite(D/'results/nominal_episode_decomposition.csv',chains);csvwrite(D/'results/nominal_boundary_detail.csv',boundary)
 for p,h in refs.items():assert sha(p)==h,p
 save(D/'results/complete.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'seconds':time.monotonic()-start,'roots':64,'new_roots':24,'old_training_roots':40,'newly_computed_h0_predictions':312,'existing_dev_eval_h0_predictions_reused':72,'no_new_training':True,'no_new_collection':True,'no_readout_or_coordinate_probes':True,'C_D_samples':0,'sha256':refs});print('AUDIT_COMPLETE',time.monotonic()-start,flush=True)
if __name__=='__main__':main()
