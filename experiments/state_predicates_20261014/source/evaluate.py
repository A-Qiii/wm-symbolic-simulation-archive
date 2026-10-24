from pathlib import Path
import json,hashlib,csv
import numpy as np

P=Path('/home/robot/research/projects/wm_symbolic');E=P/'experiments/state_predicates_20261014'
GROUPS={'held':[0,1,2],'in_drawer':[3,4,5],'open_drawer':[6]}
ATOMS=['held(red)','held(blue)','held(pink)','in(red,drawer)','in(blue,drawer)','in(pink,drawer)','open(drawer)']
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2))
def ratio(a,b):return float(a/b) if b else None
def known_mask(p,thresholds):
 t=np.array([thresholds[0]]*3+[thresholds[1]]*3+[thresholds[2]])
 k=(p>=t)|(p<=1-t)
 conflict=(p[:,:3]>=t[:3]).sum(-1)>1
 k[:,:3]&=~(conflict[:,None]&(p[:,:3]>=t[:3]))
 return k
def metric(p,y,valid,known):
 p=p[valid];y=y[valid].astype(bool);known=known[valid];pred=p>=.5
 tp=int((pred&y).sum());tn=int((~pred&~y).sum());fp=int((pred&~y).sum());fn=int((~pred&y).sum())
 n=len(y);pos=int(y.sum());neg=n-pos;nk=int(known.sum());wrong=int(((pred!=y)&known).sum());kfp=int((pred&~y&known).sum())
 bal=(ratio(tp,pos)+ratio(tn,neg))/2 if pos and neg else None
 return {'n':n,'positive':pos,'negative':neg,'TP':tp,'TN':tn,'FP':fp,'FN':fn,'accuracy':ratio(tp+tn,n),'balanced_accuracy':bal,'false_positive_rate':ratio(fp,neg),'false_negative_rate':ratio(fn,pos),'Brier':float(np.mean((p-y)**2)) if n else None,'known':nk,'unknown':n-nk,'coverage':ratio(nk,n),'unknown_rate':ratio(n-nk,n),'known_wrong':wrong,'known_error':ratio(wrong,nk),'known_false_positive':kfp,'known_FPR_over_all_negatives':ratio(kfp,neg),'known_positive_precision':ratio(int((pred&y&known).sum()),int((pred&known).sum()))}
def summary(p,y,v,k):return {name:metric(p[:,cols],y[:,cols],v[:,cols],k[:,cols]) for name,cols in GROUPS.items()}
def calibrate(p,y,v,exclude=False):
 grid=[.5,.6,.7,.8,.85,.9,.95,.975,.99,.995];thresholds=[];details={}
 for gi,(name,cols) in enumerate(GROUPS.items()):
  cols=[c for c in cols if not(exclude and c in [2,5])];rows=[];chosen=None
  for t in grid:
   known=(p[:,cols]>=t)|(p[:,cols]<=1-t)
   if name=='held':known&=~(((p[:,:3]>=t).sum(-1)>1)[:,None]&(p[:,cols]>=t))
   d=metric(p[:,cols],y[:,cols],v[:,cols],known);ok=d['known']>=20 and d['known_error']<=.05 and (name!='held' or d['known_FPR_over_all_negatives']<=.01);rows.append({'tau':t,'eligible':ok,**d})
   if ok and (chosen is None or d['coverage']>chosen['coverage']):chosen={'tau':t,**d}
  thresholds.append(chosen['tau'] if chosen else 1.000001);details[name]={'selected':chosen,'all_unknown_fallback':chosen is None,'grid':rows}
 return thresholds,details
def main():
 with np.load(E/'data/inputs_targets.npz') as z:cat=z['cat'];v0=z['valid'];split=z['splits'];near=z['near_negative']
 y=np.concatenate([cat[:,:3]==6,cat[:,:3]==3,cat[:,3:4]==2],1);v=np.concatenate([v0[:,:3],v0[:,:3],v0[:,3:4]],1)
 meta=json.loads((E/'data/metadata.json').read_text());test=split=='diagnostic_D';cal=split=='calibration';train=split=='train';root=np.array([r['root_id'] for r in meta]);coarse=np.where(cat[:,:3]==6,2,np.where(cat[:,:3]==3,1,0));signature=[tuple(x) for x in np.c_[coarse,cat[:,3]].tolist()];seen={s for i,s in enumerate(signature) if train[i]};unseen=np.array([s not in seen for s in signature]);fine_seen={tuple(x) for x in cat[train]};fine_unseen=np.array([tuple(x) not in fine_seen for x in cat]);pos=np.array([r['object_positions_for_evaluation_only'] for r in meta]);nearest=np.sqrt(((pos[:,None]-pos[train][None])**2).sum((2,3))).min(1);novel_position=nearest>.02
 runs=json.loads((E/'results/training_runs.json').read_text());paths=[(r['tag'],E/'results'/f"{r['tag']}_probabilities.npy") for r in runs]+[('frozen_old_native_history',E/'results/frozen_old_native_history_probabilities.npy')]
 results={};calibration={};probs={};knowns={}
 for name,path in paths:
  p=np.load(path);t,c=calibrate(p[cal],y[cal],v[cal],exclude='leave_pink' in name);k=known_mask(p,t);probs[name]=p;knowns[name]=k;calibration[name]={'thresholds':t,'details':c,'fit_roots':sorted(set(root[cal])),'pink_labels_excluded':'leave_pink' in name}
  subsets={'diagnostic_D':test,'D_unseen_coarse_state_combination':test&unseen,'D_unseen_fine_state_combination':test&fine_unseen,'D_novel_3object_position_vector_gt2cm':test&novel_position}
  if 'frozen_old' not in name:subsets.update({'C_tune':split=='tune','C_calibration':cal,'AB_train':train})
  results[name]={'thresholds':t,'subsets':{sn:{'boundaries':int(s.sum()),'roots':len(set(root[s])),'predicates':summary(p[s],y[s],v[s],k[s])} for sn,s in subsets.items()},'atoms':{a:metric(p[test,i:i+1],y[test,i:i+1],v[test,i:i+1],k[test,i:i+1]) for i,a in enumerate(ATOMS)}}
  nearv=v.copy();nearv[:,:3]&=near;nearv[:,3:]=False;results[name]['near_not_held']=metric(p[test],y[test],nearv[test],k[test])
  occluded=np.zeros_like(v);objects=np.load(E/'data/inputs_targets.npz')['objects'];vis=objects[:,-1].reshape(-1,3,2,390)[:,:,:,388].any(-1);occluded[:,:3]=~vis;occluded[:,3:6]=~vis;results[name]['last_frame_invisible_objects']=metric(p[test],y[test],(v&occluded)[test],k[test])
  results[name]['held_positive_conflict_boundaries']=int(((p[test,:3]>=np.array(t)[0]).sum(1)>1).sum())
 save(E/'results/calibration.json',calibration)
 # Three seeds are separate trained models, not an ensemble; paired bootstrap by roots for seed0.
 aggregate={}
 for mode in ['matched_eight_class','direct_predicates']:
  aggregate[mode]={}
  for pred in GROUPS:
   aggregate[mode][pred]={key:{'mean':float(np.mean(vals)),'std':float(np.std(vals)),'per_seed':vals} for key in ['accuracy','balanced_accuracy','false_positive_rate','coverage','known_error'] if (vals:=[results[f'{mode}_seed{s}']['subsets']['diagnostic_D']['predicates'][pred][key] for s in [0,1,2]]) and all(x is not None for x in vals)}
 rng=np.random.default_rng(20261014);unique=sorted(set(root[test]));boot={pred:[] for pred in GROUPS};fprs=[]
 for _ in range(2000):
  inds=np.concatenate([np.flatnonzero(test&(root==r)) for r in rng.choice(unique,len(unique),replace=True)])
  for pred,cols in GROUPS.items():
   mm=[]
   for name in ['matched_eight_class_seed0','direct_predicates_seed0']:mm.append(metric(probs[name][inds][:,cols],y[inds][:,cols],v[inds][:,cols],knowns[name][inds][:,cols]))
   if all(m['balanced_accuracy'] is not None for m in mm):boot[pred].append(mm[1]['balanced_accuracy']-mm[0]['balanced_accuracy'])
  mm=metric(probs['direct_predicates_seed0'][inds,:3],y[inds,:3],v[inds,:3],knowns['direct_predicates_seed0'][inds,:3]);fprs.append(mm['false_positive_rate'])
 ci={k:np.quantile(x,[.025,.975]).tolist() for k,x in boot.items()};ci['direct_held_FPR']=np.quantile(fprs,[.025,.975]).tolist()
 criteria={}
 for mode in ['matched_eight_class','direct_predicates']:
  criteria[mode]={}
  for seed in [0,1,2]:
   ms=results[f'{mode}_seed{seed}']['subsets']['diagnostic_D']['predicates'];criteria[mode][str(seed)]={'absolute_pass':all(m['balanced_accuracy']>=.9 and m['known_error'] is not None and m['known_error']<=.05 and m['coverage']>=.7 for m in ms.values()) and ms['held']['false_positive_rate']<=.02}
 a=aggregate['matched_eight_class'];b=aggregate['direct_predicates'];ba_delta=float(np.mean([b[g]['balanced_accuracy']['mean']-a[g]['balanced_accuracy']['mean'] for g in GROUPS]));coverage_delta=float(np.mean([b[g]['coverage']['mean']-a[g]['coverage']['mean'] for g in GROUPS]));criteria['relative']={'macro_balanced_accuracy_delta':ba_delta,'macro_coverage_delta':coverage_delta,'pass':(ba_delta>=.02 or coverage_delta>=.10) and all(b[g]['known_error']['mean']<=.05 for g in GROUPS) and b['held']['false_positive_rate']['mean']-a['held']['false_positive_rate']['mean']<=.01}
 audit={'D_boundaries':int(test.sum()),'D_roots':len(unique),'unseen_coarse_state_boundaries':int((test&unseen).sum()),'unseen_fine_state_boundaries':int((test&fine_unseen).sum()),'novel_3object_position_gt2cm_boundaries':int((test&novel_position).sum()),'position_metric':'Euclidean distance of concatenated 3 object XYZ to every training boundary, evaluation only; not proof of out-of-range positions','held_near_negative_D':int(near[test].sum()),'intermediate_drawer_training_examples':int((cat[train,3]==1).sum()),'root_bootstrap_95_percent_intervals':ci,'bootstrap_repetitions':2000,'historical_D_is_not_blind':True}
 protocol=json.loads((E/'configs/protocol.json').read_text());integrity={p:hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in protocol['fixed_sha256'].items()};assert all(integrity.values())
 save(E/'results/evaluation.json',{'models':results,'three_seed_summary':aggregate,'preregistered_gates':criteria,'data_diagnostics':audit,'frozen_artifacts_unchanged':integrity})
 # Explicit atom-level probabilities and statuses, suitable for planning adapters.
 with (E/'results/D_predictions.csv').open('w') as f:
  w=csv.writer(f);w.writerow(['model','root','episode','boundary','predicate','valid','truth','probability','decision'])
  for name,p in probs.items():
   for i in np.flatnonzero(test):
    for j,a in enumerate(ATOMS):w.writerow([name,root[i],meta[i]['episode_id'],meta[i]['step'],a,bool(v[i,j]),int(y[i,j]),float(p[i,j]),int(p[i,j]>=.5) if knowns[name][i,j] else 'UNKNOWN'])
 print(json.dumps({'three_seed_summary':aggregate,'gates':criteria,'data':audit},indent=2),flush=True)
if __name__=='__main__':main()
