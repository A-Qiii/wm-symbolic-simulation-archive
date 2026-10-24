from config import *
import sys,torch,fcntl,argparse
sys.path.insert(0,str(REP/'source'))
from runtime import load
from model import Query,probs
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['C_backtest','D4'],required=True);a=ap.parse_args();bundle=read(E/'configs/before_D4_frozen.json')
 for p,h in bundle['sha256'].items():assert sha(p)==h,p
 def guard(event,args):
  if event=='open'and isinstance(args[0],(str,bytes))and any(x in str(args[0])for x in ['/_private/','/query_evaluator/']):raise RuntimeError('Prediction cannot read evaluator data')
 sys.addaudithook(guard);lock=(P/'runs/gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(4);d=PRIOR/'data/ABC'if a.phase=='C_backtest'else E/'data/D4';states=read(d/'public/states.json');pairs=read(d/'public/pairs.json');ids=[i for i,s in enumerate(states)if a.phase=='D4'or s['split']=='dev_validate'];rr=[p for p in pairs if a.phase=='D4'or p['split']=='dev_validate'];raw=np.load(d/'public/inputs.npz')['z'];z=torch.tensor(raw,device='cuda');m,oq,stats=load(0,'canonical_matched');heads={'original_head':oq};files={f'starting_seed{s}':Path(v['path'])for s,v in bundle['starting_heads'].items()}|{k:Path(v['path'])for k,v in bundle['selected_heads'].items()}
 for name,path in files.items():q=Query().cuda();q.load_state_dict(torch.load(path,weights_only=True,map_location='cuda')['state']);q.eval().requires_grad_(False);heads[name]=q
 trajectories={}
 for r in rr:trajectories.setdefault((r['root_id'],r['branch']),[]).append(r)
 records=[];one=[];rec=[];out={};stateprobs={}
 with torch.inference_mode():
  for key,rows in trajectories.items():
   rows.sort(key=lambda r:r['slot']);f=z[rows[0]['before']:rows[0]['before']+1]
   for j,r in enumerate(rows):
    assert r['slot']==j
    if j:assert rows[j-1]['after']==r['before']
    act=torch.tensor([r['action']],device='cuda');o=m.transition(z[r['before']:r['before']+1],act);f=m.transition(f,act);one.append(o[0]);rec.append(f[0].clone());records.append(r)
  one=torch.stack(one);rec=torch.stack(rec);actual=z[[r['after']for r in records]]
  for name,q in heads.items():
   ps={k:probs(q((value-stats['mean'])/stats['std'])).cpu().numpy()for k,value in [('actual',actual),('one_step',one),('recursive',rec)]};stateprobs[name]=probs(q((z[ids]-stats['mean'])/stats['std'])).cpu().numpy();out[name]=[r|{k+'_probability':ps[k][j]for k in ps}|{'one_step_raw_MSE':float((one[j]-actual[j]).square().mean()),'recursive_raw_MSE':float((rec[j]-actual[j]).square().mean())}for j,r in enumerate(records)]
 np.savez_compressed(E/'results'/f'latents_{a.phase}.npz',one_step=one.cpu().numpy(),recursive=rec.cpu().numpy(),actual=actual.cpu().numpy(),pair_ids=np.array([r['id']for r in records]));save(E/'results'/f'predictions_{a.phase}.json',out);save(E/'results'/f'true_state_probabilities_{a.phase}.json',{'state_indices':ids,'probabilities':stateprobs});save(E/'results'/f'prediction_audit_{a.phase}.json',{'fixed_transition_and_encoder_sha256':sha(REP/'models/canonical_matched_seed0/update1000.pt'),'private_reads':0,'recursive_corrections':0,'model_updates':0,'physics_calls':0,'heads':list(heads),'observed_transitions':len(records)});print('PREDICTED',a.phase,len(records),len(heads),flush=True)
if __name__=='__main__':main()
