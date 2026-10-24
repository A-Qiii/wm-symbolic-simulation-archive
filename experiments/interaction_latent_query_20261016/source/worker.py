"""Py3.11 inference bridge: raw public observation -> learned query and search.
The simulator keeps its existing Py3.10 dependency environment unchanged.
"""
from common import *
from features import Encoder
from representation import transform
from model import probs
from train_queries import load_query,normalize
from plan import search
import torch,sys,time,argparse,traceback,fcntl
def guard(event,args):
 if event=='open'and isinstance(args[0],(str,bytes))and('/_private/'in str(args[0])or'/candidate/'in str(args[0])):raise RuntimeError('Inference process rejects private/evaluator/semantic candidate files')
class Engine:
 def __init__(self):
  sys.addaudithook(guard);torch.set_num_threads(4);self.encoder=Encoder();self.norm=dict(np.load(E/'data/public/normalization.npz'));self.models={k:load_query(k,0)for k in ['generic','ae','interaction']}
 def handle(self,req):
  assert set(req)=={'kind','public_path','target'};path=Path(req['public_path']);assert path.is_relative_to(E/'data')and'/public/'in str(path);assert req['target']in[0,1,2];start=time.time_ns();fs=self.encoder.encode([path]);f={k:np.stack([v[k]for v in fs])for k in fs[0]};o,s=transform(f,self.norm);m,q,stats=self.models[req['kind']]
  with torch.no_grad():z=m.encode(torch.tensor(o,device='cuda'),torch.tensor(s,device='cuda'));pr=probs(q(normalize(z,stats)))[0].cpu().tolist();plan=search(m,q,stats,z,req['target'])
  return {'ok':True,'current_probability':pr,'search':plan,'source_sha256':sha(path),'started_ns':start,'completed_ns':time.time_ns(),'GT_read':False}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--preflight',action='store_true');args=ap.parse_args()
 if args.preflight:
  lock=(P/'runs/gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 engine=Engine()
 if args.preflight:
  rows=read(E/'data/public/states.json');inputs=dict(np.load(E/'data/public/inputs.npz'));out=[]
  for i in [0,5,10,20]:
   assert rows[i]['split']=='train'
   for k in ['generic','ae','interaction']:
    ans=engine.handle({'kind':k,'public_path':rows[i]['path'],'target':i%3});m,q,stats=engine.models[k]
    with torch.no_grad():z=m.encode(torch.tensor(inputs['objects'][i:i+1],device='cuda'),torch.tensor(inputs['scene'][i:i+1],device='cuda'));expected=probs(q(normalize(z,stats)))[0].cpu().numpy()
    diff=float(np.max(np.abs(expected-np.array(ans['current_probability']))));assert diff<.01,(i,k,diff);assert np.array_equal(expected>=.5,np.array(ans['current_probability'])>=.5);out.append({'state':i,'arm':k,'probability_max_difference':diff,'search_status':ans['search']['status']})
  save(E/'results/runtime_bridge_preflight.json',{'passed':True,'checks':out,'D_predictions_read':False,'all_GT_opens_rejected':True});print('BRIDGE_PREFLIGHT_PASS',flush=True);return
 print(json.dumps({'ready':True}),flush=True)
 for line in sys.stdin:
  req=json.loads(line)
  if req.get('op')=='stop':break
  try:r=engine.handle(req)
  except Exception as exc:r={'ok':False,'error':str(exc),'traceback':traceback.format_exc()}
  print(json.dumps(r),flush=True)
if __name__=='__main__':main()
