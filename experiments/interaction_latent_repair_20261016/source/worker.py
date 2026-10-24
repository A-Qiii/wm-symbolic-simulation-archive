from common import *
from runtime import *
from features import Encoder
from representation import transform
from model import probs
from plan import search
import sys,time,traceback
def guard(event,args):
 if event=='open'and isinstance(args[0],(str,bytes))and('/_private/'in str(args[0])or'/candidate/'in str(args[0])):raise RuntimeError('Inference process rejects evaluator-private files')
def main():
 sys.addaudithook(guard);torch.set_num_threads(4);enc=Encoder();norm=dict(np.load(E/'data/public/normalization.npz'));models={a:load(0,a)for a in ARMS};print(json.dumps({'ready':True}),flush=True)
 for line in sys.stdin:
  req=json.loads(line)
  if req.get('op')=='stop':break
  try:
   assert set(req)=={'arm','public_path','target'};path=Path(req['public_path']);assert path.is_relative_to(E/'data')and'/public/'in str(path);assert req['target']in[0,1,2];fs=enc.encode([path]*8)[:1];f={k:np.stack([v[k]for v in fs])for k in fs[0]};o,s=transform(f,norm);m,q,stats=models[req['arm']]
   with torch.no_grad():z=m.encode(torch.tensor(o,device='cuda'),torch.tensor(s,device='cuda'));pr=probs(q(normalize(z,stats)))[0].cpu().tolist();plan=search(m,q,stats,z,req['target'])
   out={'ok':True,'current_probability':pr,'search':plan,'source_sha256':sha(path),'GT_read':False}
  except Exception as ex:out={'ok':False,'error':str(ex),'traceback':traceback.format_exc()}
  print(json.dumps(out),flush=True)
if __name__=='__main__':main()
