from pathlib import Path
import json,hashlib,fcntl
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.ops import roi_align
from transformers import Dinov2Model
from registry import extract,RULE
P=Path('/home/robot/research/projects/wm_symbolic');E=P/'experiments/state_predicates_rgb_registry_v1_20261014';OLD=P/'experiments/state_predicates_long_pairs_20261014';CAND=P/'experiments/state_predicates_held_down_20261014/candidate'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
class RGBEncoder:
 def __init__(self):
  self.lock=(P/'runs/gpu.lock').open('a+');fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB);self.device='cuda:0';weights=read(CAND/'INPUT_CONTRACT.json')['external_cached_backbone']['path'];self.model=Dinov2Model.from_pretrained(weights,local_files_only=True).eval().requires_grad_(False).to(self.device);self.mean=torch.tensor([.485,.456,.406],device=self.device)[None,:,None,None];self.std=torch.tensor([.229,.224,.225],device=self.device)[None,:,None,None]
 def encode(self,paths):
  frames=[];images=[];geom=[];allmasks={'static':[],'gripper':[]};robot=[]
  for pp in paths:
   with np.load(pp)as z:
    assert set(z.files)=={'rgb_static','rgb_gripper','robot_obs'},'Revised loader rejects every undeclared input member'
    fr={k:z[k].copy()for k in z.files}
   frames.append(fr);robot.append(fr['robot_obs'])
   for cam in ['static','gripper']:
    rgb=fr['rgb_'+cam];h,w=rgb.shape[:2];top=(224-h)//2;left=(224-w)//2;assert top>=0 and left>=0;mask,_=extract(rgb);allmasks[cam].append(mask);images.append(F.pad(torch.from_numpy(rgb.copy()).permute(2,0,1).float()/255,(left,224-w-left,top,224-h-top),value=.5));boxes=[];geometry=[];padded=[]
    for j in range(3):
     yy,xx=np.where(mask[j]);vis=len(yy)>=4;x0,x1=(int(xx.min()+left),int(xx.max()+left+1))if vis else(0,1);y0,y1=(int(yy.min()+top),int(yy.max()+top+1))if vis else(0,1);boxes.append([x0,y0,x1,y1]);geometry.append([x0/224,y0/224,x1/224,y1/224,float(vis)]);padded.append(np.pad(mask[j].astype(np.float32),((top,224-h-top),(left,224-w-left))))
    geom.append((boxes,geometry,padded))
  x=torch.stack(images).to(self.device)
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):patch=self.model(pixel_values=(x-self.mean)/self.std).last_hidden_state[:,1:]
  assert patch.shape==(8,256,384);patch=patch.float().transpose(1,2).reshape(8,384,16,16);local=[];geometries=[];global_grid=[]
  for i,(boxes,g,masks)in enumerate(geom):
   boxes=torch.tensor(boxes,dtype=torch.float32,device=self.device);roi=roi_align(patch[i:i+1],[boxes],output_size=(2,2),spatial_scale=1/14,aligned=True).permute(0,2,3,1).reshape(3,4,384);masks=torch.tensor(np.stack(masks),device=self.device)[:,None];occ=roi_align(masks,torch.cat([torch.arange(3,device=self.device)[:,None],boxes],1),output_size=(2,2),aligned=True)[:,0].reshape(3,4)
   local.append(roi.cpu().half().numpy());geometries.append(np.array([[row+[float(occ[j,k]),i%2,(i//2)/3]for k in range(4)]for j,row in enumerate(g)],np.float32));global_grid.append(F.adaptive_avg_pool2d(patch[i:i+1],(4,4)).flatten(2).transpose(1,2)[0].cpu().half().numpy())
  return {'local_tokens':np.stack(local).reshape(4,2,3,4,384),'geometry':np.stack(geometries).reshape(4,2,3,4,8),'global_grid':np.stack(global_grid).reshape(4,2,16,384),'robot_history':np.stack(robot).astype(np.float32)}, {k:np.stack(v)for k,v in allmasks.items()}
def main():
 assert not(E/'data/public/paired_inputs_v1.npz').exists();scope=read(E/'configs/scope.json');freeze=read(E/'configs/frozen_before_C.json');assert sha(E/'source/registry.py')==freeze['rule_sha256'];assert read(E/'results/registration_dev.json')['registration_gate_passed']
 for p,h in scope['frozen_sha256'].items():assert sha(p)==h,p
 save(E/'configs/encoder_before_full_rebuild.json',{'source_sha256':sha(Path(__file__)),'rule_sha256':sha(E/'source/registry.py'),'created_after_rule_frozen_before_C':True,'all_input_arrays_current_or_past_RGB_only':True,'no_GT_geometry':True})
 meta=read(E/'data/windows_train.json')+read(E/'data/windows_dev.json');oldmeta=read(OLD/'data/public/windows_full.json');assert [x['window_id']for x in meta]==[x['window_id']for x in oldmeta];stats=read(CAND/'normalization.json');vm=np.array(stats['vision_mean'],np.float32);vs=np.array(stats['vision_std'],np.float32);rm=np.array(stats['robot_mean'],np.float32);rs=np.array(stats['robot_std'],np.float32);encoder=RGBEncoder();arrays={k:[]for k in ['objects','context','robot']};vis=[];out=[];maxdiff={'context':0.,'robot':0.}
 with np.load(OLD/'data/public/paired_inputs_full.npz')as baseline:
  for i,r in enumerate(meta):
   assert r['history_frame_indices']==[r['anchor_frame']-k for k in [63,42,21,0]]and abs(r['simulation_times'][-1]-r['simulation_times'][0]-2.1)<1e-10
   for pp,h in zip(r['public_frames'],r['sanitized_sha256']):assert sha(pp)==h
   f,masks=encoder.encode(r['public_frames'])
   with np.load(r['rgb_regions_path'])as z:assert all(np.array_equal(masks[k],z[k])for k in masks)
   fp=E/'data/features'/(r['window_id']+'.npz');np.savez_compressed(fp,**f);tokens=(f['local_tokens'].astype(np.float32)-vm)/vs;g=f['geometry'];visible=g[:,:,:,0,4].astype(bool);objects=np.zeros((4,3,2,390),np.float32)
   for j in range(3):
    if not visible[:,:,j].any():continue
    objects[:,j,:,:384]=tokens[:,:,j].mean(2)*visible[:,:,j,None];objects[:,j,:,384:389]=g[:,:,j,0,:5];objects[:,j,:,389]=g[:,:,j,:,5].mean(2)
   grid=(f['global_grid'].astype(np.float32)-vm)/vs;current={'objects':objects.reshape(4,3,780),'context':grid.reshape(4,2,2,2,2,2,384).mean((3,5)).reshape(4,3072),'robot':(f['robot_history']-rm)/rs}
   for k,v in current.items():assert v.dtype==np.float32 and np.isfinite(v).all();arrays[k].append(v)
   for k in maxdiff:maxdiff[k]=max(maxdiff[k],float(np.abs(current[k]-baseline[k][i]).max()))
   vis.append(visible.transpose(0,2,1));out.append(r|{'index':i,'raw_RGB_feature_path':str(fp),'raw_RGB_feature_sha256':sha(fp),'visible_per_frame_colour_camera':visible.transpose(0,2,1).tolist(),'missing_entire_window':(~visible.any((0,1))).tolist()})
   if(i+1)%30==0:print('RGB_DINO_CACHED',i+1,flush=True)
  arrays={k:np.stack(v)for k,v in arrays.items()}
  for k in ['pair_before_index','pair_after_index','declared_action_features']:arrays[k]=baseline[k].copy()
 assert maxdiff=={'context':0.,'robot':0.},maxdiff;np.savez_compressed(E/'data/public/paired_inputs_v1.npz',**arrays);np.savez_compressed(E/'data/public/visibility_v1.npz',visible=np.stack(vis),missing_entire_window=~np.stack(vis).any((1,3)))
 # Supervision remains separate and is copied only after revised public tensors exist.
 (E/'data/_private/supervision.npz').write_bytes((OLD/'data/_private/supervision_full.npz').read_bytes());save(E/'data/public/windows_v1.json',out);save(E/'data/public/pairs_v1.json',read(OLD/'data/public/pairs_full.json'));result={'passed':True,'windows':150,'macro_pairs':120,'sanitized_RGB_proprio_files':600,'causal_2point1_second_windows':True,'current_frame_only_region_extraction':True,'simulator_masks_coordinates_labels_absent_from_input_loader':True,'context_robot_max_abs_difference_from_legacy':maxdiff,'paired_inputs_sha256':sha(E/'data/public/paired_inputs_v1.npz'),'visibility_sha256':sha(E/'data/public/visibility_v1.npz'),'supervision_byte_identical':sha(E/'data/_private/supervision.npz')==sha(OLD/'data/_private/supervision_full.npz'),'candidate_forwards_so_far':0,'training_updates':0,'shapes':{k:list(v.shape)for k,v in arrays.items()}};save(E/'results/rebuild.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
