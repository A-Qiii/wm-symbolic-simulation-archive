from pathlib import Path
import sys,json,hashlib,collections
import numpy as np
P=Path('/home/robot/research/projects/wm_symbolic');OLD=P/'experiments/interaction_latent_query_20261016';E=P/'experiments/interaction_latent_repair_review_20261016';R=P/'reports/interaction_latent_repair_review_20261016'
def read(p):return json.loads(Path(p).read_text())
def save(p,x):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=lambda a:a.tolist()if isinstance(a,np.ndarray)else a.item()if isinstance(a,np.generic)else str(a))+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 E.mkdir(exist_ok=True);R.mkdir(exist_ok=True);pairs=read(OLD/'data/public/pairs.json');states=read(OLD/'data/public/states.json');outcomes={r['pair_id']:r['status']for r in read(OLD/'data/_private/pair_outcomes.json')};original={(r['episode_id'],r['step']):r for r in read(P/'data/core1200_v1/data_manifest.json')['records']if r['environment']in'ABC'};place=[]
 for r in pairs:
  if r['skill']!='place_in_drawer':continue
  row={k:r[k]for k in['id','root_id','split','dataset','episode_id','step']};row.update(bound_intent=bool(sum(r['action'][4:7])),target_bits=r['action'][4:7],status=outcomes[r['id']])
  if r['dataset']=='core':
   ori=original[r['episode_id'],r['step']];folder=Path(ori['obs_before']).parent;n=read(folder/'result_private.json');l=read(folder/'labels_private.json')['before'];assert n['status']==row['status'];row.update(reason=n.get('reason'),before_drawer_q=l['continuous']['drawer_m'],before_held=l['holding_derived'],native_source=str(folder/'result_private.json'))
  place.append(row)
 summary=[]
 for split in['train','dev']:
  for bound in[False,True]:
   rr=[r for r in place if r['split']==split and r['bound_intent']==bound];summary.append({'split':split,'bound_intent':bound,'roots':len({r['root_id']for r in rr}),'success':sum(r['status']=='success'for r in rr),'refused':sum(r['status']=='refused'for r in rr),'failed':sum(r['status']=='failed'for r in rr)})
 sd=P/'data/predicate_state_dependence_20261014';m=read(sd/'data_manifest.json');private={r['pair_id']:r for r in m['private_pairs']};windows={w['window_id']:w for w in m['windows']};extra=[]
 for r in m['records']:
  if r['skill']!='place_in_drawer':continue
  pr=private[r['pair_id']];prep=read(sd/'episodes'/r['episode_id']/'_private/preparation_condition.json');w=windows[r['before_window_id']];a=windows[r['after_window_id']]
  extra.append({'id':r['pair_id'],'root_id':r['root_id'],'episode_id':r['episode_id'],'split':r['split'],'environment':r['environment'],'condition':pr['condition'],'status':pr['primitive']['status'],'reason':pr['primitive'].get('reason'),'declared_arguments':r['declared_arguments'],'before_drawer_q':prep['actual_drawer_m'],'before_held':pr['before_targets']['y'][:3],'path':w['public_frames'][-1],'after_path':a['public_frames'][-1],'target_intent_evaluator_only':pr['planned_colour_evaluator_only'],'input_source_sha256':sha(w['public_frames'][-1]),'included_first_version':r['root_id']in{p['root_id']for p in pairs}})
 # Stored predictions, no new fits: inspect labeled training/development composition.
 labels=read(OLD/'data/_private/labels.json');budget=read(OLD/'configs/semantic_budget.json');counts={}
 for name,idx in [('all_AB',[i for i,r in enumerate(states)if r['split']=='train'and str(i)in labels]),('all_C',[i for i,r in enumerate(states)if r['split']=='dev'and str(i)in labels]),('train_budget',budget['train_state_indices']),('dev_selection_budget',budget['development_selection_state_indices'])]:
  groups=collections.Counter(('held'if any(labels[str(i)]['y'][:3])else'empty', ['closed','open','mid'][int(np.argmax(labels[str(i)]['y'][6:]))])for i in idx);counts[name]={'states':len(idx),'groups':[{'holding':k[0],'drawer':k[1],'states':v}for k,v in groups.items()]}
 current_metrics=[]
 for split in ['C','D']:
  for name,d in read(OLD/'results'/f'{split}_metrics.json').items():
   t=d['semantic_transition']['normal'];current_metrics.append({'split':split,'model':name,'current_all':{k:d['all_current_states'][k]for k in['states','roots','all_valid_facts_correct_states']},'normal_before':{k:t['actual_current'][k]for k in['states','roots','all_valid_facts_correct_states']},'normal_true_after':{k:t['actual_successor'][k]for k in['states','roots','all_valid_facts_correct_states']},'normal_predicted_after':{k:t['predicted_successor'][k]for k in['states','roots','all_valid_facts_correct_states']},'drawer_classes_correct':d['all_current_states']['drawer']['TP'],'drawer_class_states':d['all_current_states']['states']})
 first=[]
 for rid,step in [('D_interaction_chain_red_0',0),('D_interaction_chain_blue_0',0),('D_interaction_chain_pink_0',1)]:
  d=read(OLD/'data/closed_loop'/rid/'interaction/_private/decisions.json')[step];a=read(OLD/'data/closed_loop'/rid/'interaction/_private/actions.json')[step];first.append({'root_id':rid,'step':step,'current_probability':d['current_probability'],'true_y':d['actual_labels_evaluator_only']['y'],'true_joint_q':d['actual_labels_evaluator_only']['drawer_q_m'],'chosen_action':a['skill'],'declared_target':a['declared_target'],'reason':a['native'].get('reason'),'plan':d['search']['plan']})
 save(E/'results/place_used.json',place);save(E/'results/coverage_summary.json',summary);save(E/'results/extra_matched_place.json',extra);save(E/'results/semantic_budget_composition.json',counts);save(E/'results/score_denominators.json',current_metrics);save(E/'results/first_errors.json',first)
 print('USED',summary);print('LABEL_COVERAGE',counts);print('EXTRA',len(extra),'roots',len({r['root_id']for r in extra}));print('NO_TRAINING_NO_COLLECTION',flush=True)
if __name__=='__main__':main()
