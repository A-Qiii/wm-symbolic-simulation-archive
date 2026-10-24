"""Traceable current/one-step/free-rollout development analysis; no test access."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import torch
from wm_symbolic.world_model import WorldModel
from wm_symbolic.dataset import Episodes
from wm_symbolic.evaluation import mean_symbols,evaluate_data
from wm_symbolic.gpu_phase import gpu_phase
P=Path('/home/robot/research/projects/wm_symbolic');a=argparse.ArgumentParser();a.add_argument('--checkpoint',required=True);a.add_argument('--output',required=True);args=a.parse_args()
out=Path(args.output);out.mkdir(exist_ok=False);ck=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
data=Episodes(P,'dev',normalizer=ck['normalization'],event_count=7);assert all(x.startswith('C_dev_') for x in data.root_ids)
torch.set_num_threads(4);torch.manual_seed(8139);model=WorldModel(event_count=7).eval();model.load_state_dict(ck['model']);samples=8
def record(pred,target,valid):
    cats=torch.cat([pred['objects'].argmax(-1),pred['drawer'].argmax(-1)[:,None],pred['slider'].argmax(-1)[:,None]],-1)[0]
    binary=(pred['lights'].sigmoid()[0]>=.5).long();values=torch.cat([cats,binary]).cpu().tolist()
    confidence=torch.cat([pred['objects'].softmax(-1).amax(-1)[0],pred['drawer'].softmax(-1).amax(-1),pred['slider'].softmax(-1).amax(-1),torch.maximum(pred['lights'].sigmoid()[0],1-pred['lights'].sigmoid()[0])]).cpu().tolist()
    return {'predicted':values,'target':target,'valid':valid,'correct':[bool(v==t) if ok else None for v,t,ok in zip(values,target,valid)],'confidence':confidence,
            'symbol_probabilities':{k:(v.sigmoid() if k=='lights' else v.softmax(-1))[0].float().cpu().tolist() for k,v in pred.items()}}
rows=[]
with gpu_phase(P,'pilot_development_analysis'),torch.inference_mode():
    model.cuda()
    metrics=evaluate_data(model,data,samples=8)
    for i,episode in enumerate(data.episodes):
        b=data.batch([i],'cuda')
        with torch.autocast('cuda',dtype=torch.bfloat16):
            encoded=model.encode(b['tokens'],b['geometry'],b['visible'],b['robot']).repeat_interleave(samples,0)
            actions=model.action(b['skill'],b['arguments'],b['budget']).repeat_interleave(samples,0)
            posts,_,priors,_=model.history(encoded,actions);free=posts[0]
            for step,r in enumerate(episode['rows']):
                free,_=model.imagine(free,actions[:,step])
                before=b['cat'][0,step].tolist()+b['binary'][0,step].long().tolist();after=b['cat'][0,step+1].tolist()+b['binary'][0,step+1].long().tolist()
                before_valid=b['cat_valid'][0,step].tolist()+b['binary_valid'][0,step].tolist();after_valid=b['cat_valid'][0,step+1].tolist()+b['binary_valid'][0,step+1].tolist()
                current=record(mean_symbols(model.symbols(posts[step]),1,samples),before,before_valid)
                prior=record(mean_symbols(model.symbols(priors[step]),1,samples),after,after_valid)
                rolled=record(mean_symbols(model.symbols(free),1,samples),after,after_valid)
                events=model.outcome_logits(posts[step],actions[:,step],priors[step]).float().sigmoid().mean(0).cpu().tolist()
                rows.append({'episode_id':episode['episode_id'],'root_id':episode['root_id'],'step':step,'skill':r['skill'],'before_public':r['obs_before'],'after_public':r['obs_after'],
                             'current':current,'one_step_real_history':prior,'free_from_root':rolled,'outcome_probabilities':events,
                             'actual_outcomes':b['outcomes'][0,step].cpu().tolist(),'changed':[bool(x!=y and a and z) for x,y,a,z in zip(before,after,before_valid,after_valid)],
                             'scoring_only_label_path':r['labels_path']})
        print('ANALYZED',episode['episode_id'],flush=True)
with (out/'predictions.jsonl').open('w') as f:
    for row in rows:f.write(json.dumps(row)+'\n')
summary={}
for kind in ['current','one_step_real_history','free_from_root']:
    result=[]
    for j,(name,classes) in enumerate(zip(['red_location','blue_location','pink_location','drawer','slider','light','led'],[8,8,8,3,3,2,2])):
        confusion=np.zeros((classes,classes),int);unknown=0
        for row in rows:
            item=row[kind]
            if item['valid'][j]:confusion[item['target'][j],item['predicted'][j]]+=1;unknown+=item['confidence'][j]<.5
        tp=np.diag(confusion);denom=confusion.sum(0)+confusion.sum(1);f1=np.divide(2*tp,denom,out=np.zeros(classes,float),where=denom>0)
        result.append({'predicate_family':name,'facts':int(confusion.sum()),'accuracy':float(tp.sum()/max(confusion.sum(),1)),
                       'macro_F1_present_classes':float(f1[denom>0].mean()) if (denom>0).any() else None,'unknown_rate_at_diagnostic_0_5':float(unknown/max(confusion.sum(),1)),
                       'confusion_true_rows_predicted_columns':confusion.tolist()})
    summary[kind]=result
cases=[];used=set()
for criterion in ['current','one_step_real_history','free_from_root']:
    candidates=[]
    for row in rows:
        if row['root_id'] in used:continue
        if False not in row[criterion]['correct']:continue
        if criterion=='one_step_real_history' and False in row['current']['correct']:continue
        if criterion=='free_from_root' and not (row['step']>=2 and False not in row['one_step_real_history']['correct']):continue
        candidates.append(row)
    if candidates:
        selected=min(candidates,key=lambda r:(r['step'],r['episode_id']));used.add(selected['root_id']);cases.append({'first_error_type':criterion,'record':selected,
            'replacement_diagnostic':'Real-history one-step output shown beside free rollout; does not claim physical recovery' if criterion=='free_from_root' else 'Root-label replacement/actual execution still required for causal attribution'})
for row in sorted(rows,key=lambda r:(r['step'],r['episode_id'])):
    if len(cases)>=3:break
    if row['root_id'] not in used and False in row['current']['correct']:
        used.add(row['root_id']);cases.append({'first_error_type':'current','record':row,'replacement_diagnostic':'Current readout fails first; causal root-label replacement remains to be executed'})
report={'checkpoint':str(args.checkpoint),'checkpoint_sha256':hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),'selected_update':ck['update'],
        'split':'C development only','roots':len(data.root_ids),'transitions':len(rows),'metrics':metrics,'by_predicate_family':summary,'cases':cases,
        'sampling':'Eight paired stochastic trajectories; mean probabilities, no best-sample selection','threshold':.5,'calibrated':False,
        'interpretation':'Offline development evidence; neither closed-loop success nor independent D test','test_data_used':False}
(out/'analysis.json').write_text(json.dumps(report,indent=2));print('ANALYSIS_COMPLETE',json.dumps({k:v for k,v in report.items() if k not in ['metrics','by_predicate_family','cases']}))
