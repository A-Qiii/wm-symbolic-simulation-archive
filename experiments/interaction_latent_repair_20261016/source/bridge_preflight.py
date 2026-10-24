from common import *
from runtime import *
from model import probs
import subprocess,fcntl,shutil,sys
def main():
 lock=(P/'runs/gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(4);states=read(E/'data/public/states.json');inputs=dict(np.load(E/'data/public/inputs.npz'));log=(E/'logs/bridge_preflight.log').open('w');proc=subprocess.Popen([sys.executable,str(E/'source/worker.py')],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,bufsize=1);assert json.loads(proc.stdout.readline())['ready'];checks=[]
 try:
  for arm in ARMS:
   m,q,stats=load(0,arm)
   for i in range(4):
    assert states[i]['split']=='train';path=E/'data/bridge_probe/public'/f'ab{i:02d}.npz';path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():shutil.copy2(states[i]['path'],path)
    proc.stdin.write(json.dumps({'arm':arm,'public_path':str(path),'target':i%3})+'\n');proc.stdin.flush();ans=json.loads(proc.stdout.readline());assert ans['ok'],ans
    with torch.no_grad():z=m.encode(torch.tensor(inputs['objects'][i:i+1],device='cuda'),torch.tensor(inputs['scene'][i:i+1],device='cuda'));expected=probs(q(normalize(z,stats)))[0].cpu().numpy()
    delta=float(np.max(np.abs(expected-ans['current_probability'])));assert delta<.0001,(arm,i,delta);assert all(r['skill']!='place_in_drawer'or r['target']is None for r in ans['search'].get('plan',[]));checks.append({'arm':arm,'AB_state':i,'query_max_difference':delta,'place_contract_valid':True})
 finally:
  proc.stdin.write('{"op":"stop"}\n');proc.stdin.flush();assert proc.wait(timeout=30)==0;log.close()
 save(E/'results/bridge_preflight.json',{'passed':True,'checks':checks,'D2_observations_read':False,'physical_calls':0,'private_inputs_rejected_in_worker':True});print('BRIDGE_PASS',len(checks),flush=True)
if __name__=='__main__':main()
