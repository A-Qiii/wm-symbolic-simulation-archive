from config import *
import importlib.util,sys,torch,argparse,csv
spec=importlib.util.spec_from_file_location('fixed_scoring',PRIOR/'source/score.py');scoring=importlib.util.module_from_spec(spec);spec.loader.exec_module(scoring)
from model import Query,probs
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['C_backtest','D4'],required=True);a=ap.parse_args();d=PRIOR/'data/ABC'if a.phase=='C_backtest'else E/'data/D4';states=read(d/'public/states.json');labels=read(d/'_private/labels.json');outcomes=read(d/'_private/outcomes.json');preds=read(E/'results'/f'predictions_{a.phase}.json');sp=read(E/'results'/f'true_state_probabilities_{a.phase}.json');ids=sp['state_indices'];y=np.array([labels[str(i)]['y']for i in ids]);v=np.array([labels[str(i)]['v']for i in ids]);roots=np.array([states[i]['root_id']for i in ids]);rootN=len(set(roots));readouts={};trajectory={};scored={};csvrows=[];pairs=[]
 for name,p in sp['probabilities'].items():
  p=np.array(p);groups={}
  for st in ['pre_grasp','held_open','hover_above_drawer','before_release','post_place','held_closed','other_states']:
   mask=np.array([not states[i]['strata']if st=='other_states'else st in states[i]['strata']for i in ids]);groups[st]=scoring.metric(p[mask],y[mask],v[mask],roots[mask])
  readouts[name]={'all_states':scoring.metric(p,y,v,roots),'strata':groups}
 for name,rows in preds.items():
  scored[name]=[]
  for r in rows:
   lab=labels[str(r['after'])];s=r|{'truth':lab['y'],'valid':lab['v'],'status':outcomes[r['id']]['status']};scored[name].append(s)
   for j,fact in enumerate(FACTS):csvrows.append({'condition':name,'root':r['root_id'],'branch':r['branch'],'step':r['slot']+1,'skill':r['skill'],'status':s['status'],'fact':fact,'truth':lab['y'][j],'valid':lab['v'][j],**{k:bool(discrete(r[k+'_probability'])[j])for k in ['actual','one_step','recursive']}})
  trajectory[name]={b:scoring.trajectory_summary([r for r in scored[name]if r['branch']==b],rootN)for b in ['normal','held_closed']}
 for seed in[0,1]:
  for ref in [f'starting_seed{seed}',f'control_seed{seed}']:
   for scope,mask in [('all',np.ones(len(ids),bool))]+[(st,np.array([st in states[i]['strata']for i in ids]))for st in ['pre_grasp','held_open','hover_above_drawer','post_place','held_closed']]:
    aa=(discrete(sp['probabilities'][ref])==y).all(1);bb=(discrete(sp['probabilities'][f'empty_open_seed{seed}'])==y).all(1);valid=mask&v.all(1);pairs.append({'seed':seed,'reference':ref,'stratum':scope,'full9_N':int(valid.sum()),'reference_wrong_corrected_right':int((~aa&bb&valid).sum()),'reference_right_corrected_wrong':int((aa&~bb&valid).sum())})
 retention={}
 if a.phase=='C_backtest':
  cv=dict(np.load(E/'data/query_evaluator/old_C_validation.npz'));bundle=read(E/'configs/before_D4_frozen.json');files={f'starting_seed{s}':Path(v['path'])for s,v in bundle['starting_heads'].items()}|{k:Path(v['path'])for k,v in bundle['selected_heads'].items()}|{'original_head':REP/'models/query_seed0.pt'};torch.set_num_threads(4)
  with torch.no_grad():
   for name,path in files.items():q=Query();q.load_state_dict(torch.load(path,weights_only=True,map_location='cpu')['state']);p=probs(q(torch.tensor(cv['z']))).numpy();retention[name]=scoring.metric(p,cv['y'],cv['v'],cv['root_ids'])
 save(E/'results'/f'scored_{a.phase}.json',scored);save(E/'results'/f'metrics_{a.phase}.json',{'phase':a.phase,'evaluated_utc':now(),'independent_roots':rootN,'true_state_readout':readouts,'trajectory_prediction':trajectory,'paired_changes':pairs,'old_C_retention':retention,'status':'Previously seen C development backtest, not new independent validation'if a.phase=='C_backtest'else'First frozen-model D4 prediction evaluation; now seen'})
 with(R/f'paired_propositions_{a.phase}.csv').open('w')as f:w=csv.DictWriter(f,fieldnames=list(csvrows[0]));w.writeheader();w.writerows(csvrows)
 print('SCORED',a.phase,rootN,'roots',len(csvrows),'paired facts',flush=True)
if __name__=='__main__':main()
