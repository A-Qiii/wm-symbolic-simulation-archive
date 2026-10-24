"""Frozen world-model environment, synchronous requests only. No simulator/private-label import."""
from experiment import *
import sys,importlib.util,time,traceback,argparse
import numpy as np
import torch
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def known(p):
 k=np.isfinite(p);conflict=(p[:,:3]>=.5).sum(1)>1;k[:,:3]&=~(conflict[:,None]&(p[:,:3]>=.5));return k
def action(decl):
 a=np.zeros((1,9),np.float32);skill=decl['skill'];j=0 if skill=='open_drawer'else 1 if skill.startswith('lift_')else 2 if skill=='place_in_drawer'else 3;a[0,j]=1
 if j==1:a[0,4+COLORS.index(decl['declared_arguments']['color'])]=1
 a[0,7]=decl['declared_primitive_budget']/900;a[0,8]=decl['declared_posthold_controls']/96;return a
class Engine:
 def __init__(self):
  self.q=frozen();torch.set_num_threads(4);torch.set_float32_matmul_precision('high');sys.path.insert(0,str(RGB/'source'));self.rgb=module('frozen_rgb_encoder_D',RGB/'source/cache.py').RGBEncoder();sys.path.insert(0,str(OLD/'source'));from common import reader,encode,logits,digest_model
  from transition import LatentTransition
  self.m=reader();self.encode_latent=encode;self.logits=logits;self.digest=digest_model(self.m);self.nets={}
  for name,cp in self.q['models'].items():
   net=LatentTransition('full').cuda().eval().requires_grad_(False);net.load_state_dict(torch.load(cp['path'],map_location='cuda',weights_only=False)['model']);self.nets[name]=net
  self.tree_module=module('frozen_shallow_tree_D',TRAIN/'source/trees.py');self.tree=read(TRAIN/'models/tree_aug132.json');self.copy_module=module('frozen_copy_D',COPY/'source/diagnostic.py');self.rules=read(COPY/'configs/rules_frozen.json')['rules'];stats=read(CAND/'normalization.json');self.vm=np.array(stats['vision_mean'],np.float32);self.vs=np.array(stats['vision_std'],np.float32);self.rm=np.array(stats['robot_mean'],np.float32);self.rs=np.array(stats['robot_std'],np.float32)
 def observe(self,paths):
  f,_=self.rgb.encode(paths);g=f['geometry'];vis=g[:,:,:,0,4].astype(bool);tok=(f['local_tokens'].astype(np.float32)-self.vm)/self.vs;o=np.zeros((4,3,2,390),np.float32)
  for j in range(3):
   if not vis[:,:,j].any():continue
   o[:,j,:,:384]=tok[:,:,j].mean(2)*vis[:,:,j,None];o[:,j,:,384:389]=g[:,:,j,0,:5];o[:,j,:,389]=g[:,:,j,:,5].mean(2)
  gg=(f['global_grid'].astype(np.float32)-self.vm)/self.vs;inputs={'objects':o.reshape(4,3,780),'context':gg.reshape(4,2,2,2,2,2,384).mean((3,5)).reshape(4,3072),'robot':(f['robot_history']-self.rm)/self.rs};tt={k:torch.as_tensor(v[None],device='cuda')for k,v in inputs.items()}
  with torch.inference_mode():h,s=self.encode_latent(self.m,**tt);p=self.logits(self.m,h,s).sigmoid()
  return f,inputs,h,s,p.cpu().numpy(),vis
 def predict(self,h,s,prob,decl):
  a=action(decl);out={'persistence':prob.copy()};lat={}
  with torch.inference_mode():
   for name,net in self.nets.items():
    hh,ss=net(h,s,torch.as_tensor(a,device='cuda'));out[name]=self.logits(self.m,hh,ss).sigmoid().cpu().numpy();lat[name+'_objects']=hh.cpu().numpy();lat[name+'_scene']=ss.cpu().numpy()
  sx=np.concatenate([np.where(known(prob),prob>=.5,-1),a],1).astype(np.float32);out['tree']=self.tree_module.predict(self.tree,sx);out['copy_effect'],k,audit=self.copy_module.predict(prob,[decl|{'pair_id':'constant_audit_id'}],self.rules);assert np.array_equal(k,known(out['copy_effect']));return out,lat,a,sx
 def handle(self,req):
  assert req['op']in ['before','after'];start=time.time_ns();wid=req['window_id'];assert '/'not in wid and '..'not in wid;paths=req['public_frames'];assert len(paths)==4 and len(set(paths))==4 and all(Path(p).is_relative_to(B/'episodes')for p in paths);f,inputs,h,s,prob,vis=self.observe(paths)
  raw=E/'data/features'/(wid+'.npz');norm=E/'data/public'/(wid+'.npz');obs=E/'data/public'/(wid+'_observed.npz');assert not raw.exists()and not norm.exists()and not obs.exists();np.savez_compressed(raw,**f);np.savez_compressed(norm,**inputs);np.savez_compressed(obs,current_objects=h.cpu().numpy(),current_scene=s.cpu().numpy(),probability=prob)
  result={'op':req['op'],'window_id':wid,'started_wall_ns':start,'input_sha256':[sha(p)for p in paths],'raw_features_path':str(raw),'raw_features_sha256':sha(raw),'normalized_path':str(norm),'normalized_sha256':sha(norm),'observed_path':str(obs),'observed_sha256':sha(obs),'missing_entire_window':(~vis.any((0,1))).tolist(),'current_frame_RGB_v1_only':True,'no_private_inputs':True}
  if req['op']=='before':
   decl=req['declaration'];assert set(decl)=={'skill','declared_arguments','declared_primitive_budget','declared_posthold_controls'};out,lat,a,sx=self.predict(h,s,prob,decl);pf=E/'data/predictions'/(wid+'.npz');assert not pf.exists();np.savez_compressed(pf,**out,**lat,declared_actions=a,symbolic_inputs=sx);result.update(predictions_path=str(pf),predictions_sha256=sha(pf),declaration=decl,predictive_input_keys=['current_objects','current_scene','declared_actions'],candidate_model='candidate_seed0')
  result['completed_wall_ns']=time.time_ns();result['created_utc']=now();jp=E/'data/public'/(wid+'_inference.json');save(jp,result);return {'ok':True,'audit_path':str(jp),'audit_sha256':sha(jp),**result}
def preflight():
 eng=Engine();meta=read(TRAIN/'data/_private/evaluation_metadata.json');reference=dict(np.load(TRAIN/'results/predictions.npz'));windows={w['window_id']:w for w in read(RGB/'data/public/windows_v1.json')};features=dict(np.load(RGB/'data/public/paired_inputs_v1.npz'));wi={w['window_id']:i for i,w in enumerate(read(RGB/'data/public/windows_v1.json'))};used=set();checks=[]
 for i,r in enumerate(meta):
  if r['source']!='old'or r['split']!='train'or r['skill']in used:continue
  used.add(r['skill']);w=windows[r['before_window_id']];f,inp,h,s,prob,vis=eng.observe(w['public_frames']);raw=dict(np.load(w['raw_RGB_feature_path']));assert all(np.array_equal(f[k],raw[k])for k in f);assert all(np.array_equal(inp[k],features[k][wi[w['window_id']]])for k in inp);decl={k:r[k]for k in ['skill','declared_arguments','declared_primitive_budget','declared_posthold_controls']};out,_,_,_=eng.predict(h,s,prob,decl);errors={}
  for name,key in [('candidate_seed0','aug132_seed0_selected'),('parallel_seed1','aug132_seed1_selected'),('parallel_seed2','aug132_seed2_selected'),('tree','tree_aug132'),('copy_effect','copy_effect'),('persistence','persistence')]:
   pp=out[name];rp=reference[key][i:i+1];assert np.array_equal(known(pp),known(rp));assert np.array_equal((pp>=.5)&known(pp),(rp>=.5)&known(rp));diff=float(np.nanmax(np.abs(pp-rp)));assert diff<=.0005;errors[name]=diff
  checks.append({'pair_id':r['pair_id'],'skill':r['skill'],'raw_features_byte_equal':True,'normalization_byte_equal':True,'output_probabilities_max_abs_difference':errors,'ternary_facts_identical':True})
 assert len(checks)==6;save(E/'results/online_AB_preflight.json',{'created_utc':now(),'passed':True,'checks':checks,'D_observations_used':0,'new_training':0,'runtime':{'python':sys.version,'torch':torch.__version__},'same_world_model_environment_as_training':True});print('AB_PREFLIGHT_PASSED',flush=True)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--preflight',action='store_true');a=ap.parse_args()
 if a.preflight:preflight();return
 eng=Engine();print(json.dumps({'ready':True,'lock_owner':'frozen_RGB_encoder','runtime_torch':torch.__version__}),flush=True)
 for line in sys.stdin:
  req=json.loads(line)
  if req.get('op')=='stop':break
  try:response=eng.handle(req)
  except Exception as e:response={'ok':False,'error':type(e).__name__+':'+str(e),'traceback':traceback.format_exc()}
  print(json.dumps(response,ensure_ascii=False,allow_nan=False),flush=True)
if __name__=='__main__':main()
