from config import *
import importlib.util,argparse,csv
spec=importlib.util.spec_from_file_location('fixed_scoring',PRIOR/'source/score.py');scoring=importlib.util.module_from_spec(spec);spec.loader.exec_module(scoring)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['C_backtest','D5'],required=True);a=ap.parse_args();d=PRIOR/'data/ABC'if a.phase=='C_backtest'else E/'data/D5'
    states=read(d/'public/states.json');labels=read(d/'_private/labels.json');outcomes=read(d/'_private/outcomes.json');preds=read(E/'results'/f'predictions_{a.phase}.json');sp=read(E/'results'/f'true_state_probabilities_{a.phase}.json');ids=sp['state_indices'];y=np.array([labels[str(i)]['y']for i in ids]);v=np.array([labels[str(i)]['v']for i in ids]);roots=np.array([states[i]['root_id']for i in ids]);rootN=len(set(roots));readouts={};trajectory={};scored={};csvrows=[];rootrows=[]
    for name,p in sp['probabilities'].items():
        p=np.array(p);readouts[name]=scoring.metric(p,y,v,roots);scored[name]=[]
        for r in preds[name]:
            lab=labels[str(r['after'])];s=r|{'truth':lab['y'],'valid':lab['v'],'status':outcomes[r['id']]['status']};scored[name].append(s)
            for j,fact in enumerate(FACTS):csvrows.append({'condition':name,'root':r['root_id'],'branch':r['branch'],'step':r['slot']+1,'skill':r['skill'],'status':s['status'],'fact':fact,'truth':lab['y'][j],'valid':lab['v'][j],**{k:bool(discrete(r[k+'_probability'])[j])for k in ['actual','one_step','recursive']}})
        trajectory[name]={branch:scoring.trajectory_summary([r for r in scored[name]if r['branch']==branch],rootN)for branch in ['normal','held_closed']}
        for root in sorted(set(roots)):
            for branch in ['normal','held_closed']:
                rr=sorted([r for r in scored[name]if r['root_id']==root and r['branch']==branch],key=lambda r:r['slot']);correct={k:[bool((discrete(r[k+'_probability'])==r['truth']).all()and all(r['valid']))for r in rr]for k in ['actual','one_step','recursive']}
                for k,cc in correct.items():
                    j=next((j for j,x in enumerate(cc)if not x),None);rootrows.append({'condition':name,'root':root,'branch':branch,'method':k,'observed_steps':len(rr),'all_four_correct':len(rr)==4 and all(cc),'first_wrong_step':None if j is None else rr[j]['slot']+1,'first_wrong_facts':[]if j is None else[f for i,f in enumerate(FACTS)if rr[j]['valid'][i]and discrete(rr[j][k+'_probability'])[i]!=rr[j]['truth'][i]]})
    for seed in [0,1]:
        for arm in ['old_chains','new_chains']:
            assert np.array_equal(sp['probabilities'][f'initial_seed{seed}'],sp['probabilities'][f'{arm}_seed{seed}'])
            for aa,bb in zip(scored[f'initial_seed{seed}'],scored[f'{arm}_seed{seed}']):assert aa['id']==bb['id']and aa['actual_probability']==bb['actual_probability']
    save(E/'results'/f'scored_{a.phase}.json',scored);save(E/'results'/f'root_trajectories_{a.phase}.json',rootrows)
    save(E/'results'/f'metrics_{a.phase}.json',{'phase':a.phase,'evaluated_utc':now(),'independent_roots':rootN,'true_state_readout':readouts,'trajectory_prediction':trajectory,'actual_readout_identical_across_transition_arms':True,'status':'Previously seen C development backtest; excluded from current fitting/selection'if a.phase=='C_backtest'else'First frozen-model D5 evaluation, now seen'})
    with(R/f'paired_propositions_{a.phase}.csv').open('w')as f:w=csv.DictWriter(f,fieldnames=list(csvrows[0]));w.writeheader();w.writerows(csvrows)
    print('SCORED',a.phase,rootN,len(csvrows),flush=True)
if __name__=='__main__':main()
