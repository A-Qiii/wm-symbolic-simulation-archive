"""Model worker uses public observation/action manifest only, never evaluator labels."""
from config import *
import sys,torch,fcntl
sys.path.insert(0,str(REP/'source'))
from runtime import load,normalize
from model import probs
from features import Encoder
from representation import transform
def main():
 lock=(P/'runs/gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(4)
 def guard(event,args):
  if event=='open'and isinstance(args[0],(str,bytes))and any(s in str(args[0])for s in['/_private/','/evaluator/']):raise RuntimeError('Frozen public model worker rejects evaluator metadata')
 sys.addaudithook(guard)
 manifest=read(E/'data/public/part1_manifest.json');states=manifest['states'];known=read(REP/'data/D2/public/states.json');lookup={content_hash(s['path']):i for i,s in enumerate(known)};d=dict(np.load(REP/'data/D2/public/inputs.npz'));norm=dict(np.load(REP/'data/public/normalization.npz'));objects=[None]*len(states);scene=[None]*len(states);missing=[]
 for i,s in enumerate(states):
  if s['content_sha256']in lookup:j=lookup[s['content_sha256']];objects[i]=d['objects'][j];scene[i]=d['scene'][j]
  else:missing.append(i)
 if missing:
  enc=Encoder()
  for start in range(0,len(missing),8):
   ids=missing[start:start+8];paths=[Path(states[i]['path'])for i in ids];fs=enc.encode(paths+[paths[-1]]*(8-len(paths)))[:len(ids)];oo,ss=transform({k:np.stack([f[k]for f in fs])for k in fs[0]},norm)
   for j,i in enumerate(ids):objects[i]=oo[j];scene[i]=ss[j]
  del enc;torch.cuda.empty_cache()
 objects=np.stack(objects);scene=np.stack(scene);np.savez_compressed(E/'data/public/part1_inputs.npz',objects=objects,scene=scene)
 m,q,stats=load(0,'canonical_matched');results=[];latent={k:[]for k in['current','actual','one_step','recursive']}
 with torch.inference_mode():
  z=m.encode(torch.tensor(objects,device='cuda'),torch.tensor(scene,device='cuda'));actualp=probs(q(normalize(z,stats))).cpu().numpy()
  for tr in manifest['trajectories']:
   forward=z[tr['steps'][0]['before']:tr['steps'][0]['before']+1]
   for row in tr['steps']:
    b,a=row['before'],row['after'];act=torch.tensor([row['action']],device='cuda');one=m.transition(z[b:b+1],act);forward=m.transition(forward,act)
    rr={'trajectory_id':tr['trajectory_id'],'source':tr['source'],'root_id':tr['root_id']}|row;rr['actual_probability']=actualp[a].tolist();rr['current_probability']=actualp[b].tolist();rr['one_step_probability']=probs(q(normalize(one,stats)))[0].cpu().tolist();rr['recursive_probability']=probs(q(normalize(forward,stats)))[0].cpu().tolist();rr['one_step_latent_MSE']=float((one-z[a:a+1]).square().mean());rr['recursive_latent_MSE']=float((forward-z[a:a+1]).square().mean());results.append(rr)
    for k,v in [('current',z[b:b+1]),('actual',z[a:a+1]),('one_step',one),('recursive',forward)]:latent[k].append(v[0].cpu().numpy())
 np.savez_compressed(E/'data/public/part1_latents.npz',**{k:np.stack(v)for k,v in latent.items()});save(E/'results/part1_predictions.json',results);sel=read(REP/'models/canonical_matched_seed0/selected.json');save(E/'results/part1_inference.json',{'model':sel,'query_sha256':sha(REP/'models/query_seed0.pt'),'unique_real_states':len(states),'cached_D2_inputs_reused':len(states)-len(missing),'new_frozen_DINO_encodings':len(missing),'scored_transitions':len(results),'window':'One current static RGB, one current gripper RGB, current15D robot observation at exact skill boundary. No past/future image windows.','action':'Logged8D canonical action, not reconstructed targets; timestamps/continuity verified by previous sealed horizon audit.','recursive_corrections':0,'training_updates':0,'new_physical_calls':0,'private_metadata_opened_by_model_worker':False})
 print('PART1_INFERENCE_COMPLETE',len(results),'missing_feature_states',len(missing),flush=True)
if __name__=='__main__':main()
