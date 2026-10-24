"""Frozen predictions over public observations; labels/statuses/private simulator traces blocked."""
from config import *
import torch,sys,fcntl,argparse
sys.path.insert(0,str(REP/'source'))
from runtime import load
from model import Query,probs
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['C_validation','D3'],required=True);a=ap.parse_args();bundle=read(E/'configs/before_D3_frozen.json')
 for p,h in bundle['sha256'].items():assert sha(p)==h,p
 def guard(event,args):
  if event=='open'and isinstance(args[0],(str,bytes))and any(s in str(args[0])for s in ['/_private/','/query_evaluator/']):raise RuntimeError('Predictor blocks evaluator material')
 sys.addaudithook(guard);lock=(P/'runs/gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(4);d=E/'data'/('ABC'if a.phase=='C_validation'else'D3');states=read(d/'public/states.json');pairs=read(d/'public/pairs.json');inputs=dict(np.load(d/'public/inputs.npz'));ids=[i for i,r in enumerate(states)if a.phase=='D3'or r['split']=='dev_validate'];rr=[r for r in pairs if a.phase=='D3'or r['split']=='dev_validate'];m,oq,stats=load(0,'canonical_matched');z=torch.tensor(inputs['z'],device='cuda');heads={'original_head':oq};headfiles={}
 for arm in ['old_coverage','new_coverage']:
  for seed in [0,1]:
   name=f'query_{arm}_seed{seed}';s=bundle['query_heads'][name];q=Query().cuda();q.load_state_dict(torch.load(s['path'],weights_only=True,map_location='cuda')['state']);q.eval().requires_grad_(False);heads[name]=q;headfiles[name]=s['sha256']
 choices=read(E/'configs/query_choices_frozen.json')['selected'];models={'fixed_transition':REP/'models/canonical_matched_seed0/update1000.pt'}|{k:Path(v['path'])for k,v in bundle['transition_models'].items()};predictions={};stateprob={};latentfiles={}
 with torch.inference_mode():
  for name,q in heads.items():stateprob[name]=probs(q((z[ids]-stats['mean'])/stats['std'])).cpu().numpy()
  trajectories={}
  for r in rr:trajectories.setdefault((r['root_id'],r['branch']),[]).append(r)
  for modelname,path in models.items():
   m.load_state_dict(torch.load(path,weights_only=True,map_location='cuda')['state']);m.eval();latents={};modelrows=[]
   for key,seq in trajectories.items():
    seq=sorted(seq,key=lambda r:r['slot']);assert seq[0]['slot']==0;forward=z[seq[0]['before']:seq[0]['before']+1]
    for j,r in enumerate(seq):
     if j:assert seq[j-1]['after']==r['before']and r['slot']==seq[j-1]['slot']+1
     act=torch.tensor([r['action']],device='cuda');one=m.transition(z[r['before']:r['before']+1],act);forward=m.transition(forward,act);latents[r['id']]={'one_step':one[0].cpu().numpy(),'recursive':forward[0].cpu().numpy()};modelrows.append(r)
   arrays={method:np.stack([latents[r['id']][method]for r in modelrows])for method in ['one_step','recursive']};np.savez_compressed(E/'results'/f'latents_{a.phase}_{modelname}.npz',**arrays,pair_ids=np.array([r['id']for r in modelrows]));latentfiles[modelname]=sha(E/'results'/f'latents_{a.phase}_{modelname}.npz')
   uses=list(heads)if modelname=='fixed_transition'else[f'query_{choices[modelname[-1]]["arm"]}_seed{modelname[-1]}']
   for qname in uses:
    q=heads[qname];condition=qname if modelname=='fixed_transition'else modelname;ps={method:probs(q((torch.tensor(arrays[method],device='cuda')-stats['mean'])/stats['std'])).cpu().numpy()for method in arrays};actual=probs(q((z[[r['after']for r in modelrows]]-stats['mean'])/stats['std'])).cpu().numpy();out=[]
    for j,r in enumerate(modelrows):out.append(r|{'model':modelname,'query':qname,'actual_probability':actual[j],'one_step_probability':ps['one_step'][j],'recursive_probability':ps['recursive'][j],'one_step_raw_MSE':float(np.mean((arrays['one_step'][j]-inputs['z'][r['after']])**2)),'recursive_raw_MSE':float(np.mean((arrays['recursive'][j]-inputs['z'][r['after']])**2))})
    predictions[condition]=out
 save(E/'results'/f'predictions_{a.phase}.json',predictions);save(E/'results'/f'true_state_probabilities_{a.phase}.json',{'state_indices':ids,'probabilities':stateprob});save(E/'results'/f'prediction_audit_{a.phase}.json',{'phase':a.phase,'states':len(ids),'transitions':len(rr),'conditions':list(predictions),'private_reads':0,'recursive_corrections':0,'new_training_or_physics':False,'query_hashes':headfiles,'latent_file_hashes':latentfiles});print('PREDICTED',a.phase,len(rr),'transitions',len(predictions),'conditions',flush=True)
if __name__=='__main__':main()
