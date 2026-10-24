from config import *
import sys,torch,fcntl,argparse
sys.path.insert(0,str(REP/'source'))
from runtime import load,tensor_hash
from model import Query,probs

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['C_backtest','D5'],required=True);a=ap.parse_args();bundle=read(E/'configs/choices_frozen.json')
    for p,h in bundle['sha256'].items():assert sha(p)==h,p
    def guard(event,args):
        if event=='open'and isinstance(args[0],(str,bytes))and any(x in str(args[0])for x in ['/_private/','/query_evaluator/']):raise RuntimeError('Prediction cannot read evaluator truth')
    sys.addaudithook(guard);lock=(P/'runs/gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(4)
    d=PRIOR/'data/ABC' if a.phase=='C_backtest' else E/'data/D5';states=read(d/'public/states.json');pairs=read(d/'public/pairs.json');ids=[i for i,s in enumerate(states)if a.phase=='D5'or s['split']=='dev_validate'];rr=[r for r in pairs if a.phase=='D5'or r['split']=='dev_validate'];z=torch.tensor(np.load(d/'public/inputs.npz')['z'],device='cuda');m,_,stats=load(0,'canonical_matched');frozen=tensor_hash({k:v for k,v in m.state_dict().items()if not k.startswith('transition.')})
    trajectories={}
    for r in rr:trajectories.setdefault((r['root_id'],r['branch']),[]).append(r)
    out={};stateprobs={};latents={}
    with torch.inference_mode():
        for seed in [0,1]:
            q=Query().cuda();q.load_state_dict(torch.load(bundle['query_heads'][str(seed)]['path'],weights_only=True,map_location='cuda')['state']);q.eval()
            for arm in ['initial','old_chains','new_chains']:
                name=f'{arm}_seed{seed}';path=REP/'models/canonical_matched_seed0/update1000.pt'if arm=='initial'else Path(bundle['selected'][name]['path']);m.load_state_dict(torch.load(path,weights_only=True,map_location='cuda')['state']);assert tensor_hash({k:v for k,v in m.state_dict().items()if not k.startswith('transition.')})==frozen
                records=[];one=[];rec=[]
                for _,rows in trajectories.items():
                    rows.sort(key=lambda r:r['slot']);cur=z[rows[0]['before']:rows[0]['before']+1]
                    for j,r in enumerate(rows):
                        assert r['slot']==j
                        if j:assert rows[j-1]['after']==r['before']
                        act=torch.tensor([r['action']],device='cuda');o=m.transition(z[r['before']:r['before']+1],act);cur=m.transition(cur,act);one.append(o[0]);rec.append(cur[0].clone());records.append(r)
                one=torch.stack(one);rec=torch.stack(rec);actual=z[[r['after']for r in records]];ps={k:probs(q((value-stats['mean'])/stats['std'])).cpu().numpy()for k,value in [('actual',actual),('one_step',one),('recursive',rec)]};stateprobs[name]=probs(q((z[ids]-stats['mean'])/stats['std'])).cpu().numpy()
                out[name]=[r|{k+'_probability':ps[k][j]for k in ps}|{'one_step_raw_MSE':float((one[j]-actual[j]).square().mean()),'recursive_raw_MSE':float((rec[j]-actual[j]).square().mean())}for j,r in enumerate(records)]
                latents[name+'_one_step']=one.cpu().numpy();latents[name+'_recursive']=rec.cpu().numpy()
    np.savez_compressed(E/'results'/f'latents_{a.phase}.npz',**latents,actual=actual.cpu().numpy(),pair_ids=np.array([r['id']for r in records]))
    save(E/'results'/f'predictions_{a.phase}.json',out);save(E/'results'/f'true_state_probabilities_{a.phase}.json',{'state_indices':ids,'probabilities':stateprobs})
    save(E/'results'/f'prediction_audit_{a.phase}.json',{'private_reads':0,'recursive_corrections':0,'model_updates':0,'physics_calls':0,'observed_transitions':len(records),'conditions':list(out),'encoder_decoder_hash':frozen})
    print('PREDICTED',a.phase,len(records),len(out),flush=True)
if __name__=='__main__':main()
