from common import *
from model import QueryRelationReader,parse_query
import torch,sys,time,importlib.util,argparse,traceback
class Engine:
 def __init__(self):
  torch.set_num_threads(4);sys.path.insert(0,str(RGB/'source'));from cache import RGBEncoder
  self.encoder=RGBEncoder();cp=read(E/'configs/reader_frozen.json');assert sha(cp['path'])==cp['sha256'];self.net=QueryRelationReader().cuda().eval().requires_grad_(False);self.net.load_state_dict(torch.load(cp['path'],map_location='cuda',weights_only=False)['model'])
  spec=importlib.util.spec_from_file_location('frozen_legacy_reader',CAND/'model.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);self.old=mod.PredicateReader().cuda().eval().requires_grad_(False);self.old.load_state_dict(torch.load(CAND/'reader_seed1.pt',map_location='cuda',weights_only=False)['model'])
 def observe(self,paths):
  assert len(paths)==4 and len(set(paths))==4;f,_=self.encoder.encode(paths);x=normalize(f);xx={k:torch.tensor(v[None],device='cuda')for k,v in x.items()}
  with torch.inference_mode():pro=self.net(**xx).sigmoid().cpu().numpy();old=self.old.probabilities(**xx).cpu().numpy()
  return f,x,pro,old
 def handle(self,req):
  assert set(req)=={'window_id','public_frames'};wid=req['window_id'];assert '/'not in wid and '..'not in wid and all(Path(p).is_relative_to(B/'episodes')for p in req['public_frames']);start=time.time_ns();f,x,pro,old=self.observe(req['public_frames']);out=E/'data/test_inference';out.mkdir(exist_ok=True)
  rp=out/(wid+'_features.npz');ip=out/(wid+'_inputs.npz');pp=out/(wid+'_predictions.npz');assert not pp.exists();np.savez_compressed(rp,**f);np.savez_compressed(ip,**x);np.savez_compressed(pp,new_reader=pro,old_reader=old)
  return {'ok':True,'window_id':wid,'started_wall_ns':start,'completed_wall_ns':time.time_ns(),'probabilities':pro[0].tolist(),'old_reader_probabilities':old[0].tolist(),'decoded':decode(pro)[0].tolist(),'public_frame_sha256':[sha(p)for p in req['public_frames']],'raw_features_path':str(rp),'inputs_path':str(ip),'predictions_path':str(pp),'predictions_sha256':sha(pp),'input_keys':['rgb_static','rgb_gripper','robot_obs'],'GT_or_action_input':False}
def preflight():
 engine=Engine();mm=read(BASE/'data/public/windows.json');x=dict(np.load(BASE/'data/public/reader_inputs.npz'));reference=np.load(E/'results/reader_probabilities.npy');checks=[]
 # One AB sample per source and colour-related location plus two hard states, fixed index order.
 ids=[0,35,75,150,258,260]
 for i in ids:
  assert mm[i]['split']=='train';f,cur,pro,old=engine.observe(mm[i]['public_frames']);diff={k:float(np.abs(cur[k]-x[k][i]).max())for k in cur};assert max(diff.values())<=1e-6;assert np.array_equal(decode(pro),decode(reference[i:i+1]));delta=float(np.abs(pro-reference[i:i+1]).max());assert delta<.0005;checks.append({'window_id':mm[i]['window_id'],'features_maxdiff':diff,'probability_maxdiff':delta})
 save(E/'results/runtime_preflight.json',{'passed':True,'checks':checks,'D_used':0,'new_training':0});print('PREFLIGHT_PASSED',flush=True)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--preflight',action='store_true');a=ap.parse_args()
 if a.preflight:preflight();return
 eng=Engine();print(json.dumps({'ready':True}),flush=True)
 for line in sys.stdin:
  req=json.loads(line)
  if req.get('op')=='stop':break
  try:r=eng.handle(req)
  except Exception as e:r={'ok':False,'error':str(e),'traceback':traceback.format_exc()}
  print(json.dumps(r),flush=True)
if __name__=='__main__':main()
