from pathlib import Path
import json,csv,hashlib,collections,datetime,sys,os
import numpy as np
P=Path('/home/robot/research/projects/wm_symbolic');R=P/'experiments/failure_recovery_control_20261019';C=P/'experiments/paired_outcome_training_20261019';O=P/'experiments/observation_state_update_20261019';U=P/'experiments/observation_retention_audit_20261019';COLORS=['red','blue','pink'];refs={}
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):refs[str(p)]=sha(p);return json.loads(Path(p).read_text())
def save(p,d):Path(p).parent.mkdir(exist_ok=True,parents=True);Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def csvwrite(p,rows):
 if rows:
  with Path(p).open('w',newline='')as f:w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def rel(p):return np.concatenate([p[...,np.array([6,14,22])],p[...,np.array([3,11,19])],p[...,24:27]],-1)
def truth(c):return np.concatenate([c[...,:3]==6,c[...,:3]==3,np.eye(3)[c[...,3]]],-1)
def right(p,y):return np.concatenate([((p[:,:6]>=.5)==y[:,:6]),(p[:,6:].argmax(-1)==y[:,6:].argmax(-1))[:,None]],-1)
def guard(event,args):
 if event=='socket.connect':raise RuntimeError('Offline audit')
 if event!='open'or not isinstance(args[0],(str,bytes)):return
 p=Path(os.fsdecode(args[0])).resolve();mode=args[1];flags=args[2]if len(args)>2 else 0
 if p.is_relative_to(P):
  writing=(isinstance(mode,str)and any(c in mode for c in 'wa+'))or(isinstance(flags,int)and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
  if writing and not p.is_relative_to(U):raise RuntimeError('Historical write forbidden')
  if any(x in str(p)for x in ['externalC','C_core1200','D_core1200','calvin_scene_C','calvin_scene_D','calvin_table_C','calvin_table_D']):raise RuntimeError('C/D forbidden')
sys.addaudithook(guard)

def main():
 start=now();roots={r['root_id']:r for r in read(R/'data/roots_manifest.json')if r['split']in ['train','development']}
 ids=set(read(P/'experiments/rssm_scaling_run_20261019/configs/fixed_design.json')['members']['train40']);oldrows=[r for r in read(P/'data/core1200_v1/data_manifest.json')['records']if r['root_id']in ids];eps=collections.defaultdict(list)
 for r in oldrows:eps[r['episode_id']].append(r)
 skillseq=collections.Counter();coverage=[];oldlift=[]
 for eid,rr in eps.items():
  rr.sort(key=lambda r:r['step']);ll=[read(r['labels_path'])for r in rr];skills=[r['skill']for r in rr];skillseq[tuple(skills)]+=1
  for t,(r,l)in enumerate(zip(rr,ll)):
   target=next((c for c in COLORS if r['skill']==f'lift_{c}_block_table'),None)
   if target:
    before=l['before']['categorical'];prior=ll[t-1]if t else None
    for a in COLORS:
     if a!=target:
      context=before[COLORS.index(a)]==3 and 6 not in before[:3] and before[COLORS.index(target)]in [0,1,2]
      preceding_place=t>0 and rr[t-1]['skill']=='place_in_drawer' and prior['before']['categorical'][COLORS.index(a)]==6 and prior['after']['categorical'][COLORS.index(a)]==3
      oldlift.append({'episode_id':eid,'environment':r['environment'],'step':t,'A':a,'target':target,'A_InDrawer_emptyhand_target_table':context,'preceding_successful_place_A':preceding_place})
 for rid,root in roots.items():
  if root['split']!='train':continue
  for branch,key in [(0,'primary_color'),(1,'matched_color')]:
   a,b=root['held_color'],root[key];matching=[r for r in oldlift if r['environment']==root['environment']and r['A']==a and r['target']==b]
   motif=(f'lift_{b}_block_table','place_in_drawer',f'lift_{b}_block_table','place_in_drawer')
   coverage.append({'root_id':rid,'environment':root['environment'],'branch':branch,'initial_A':a,'target_B':b,'old_same_env_context_transitions':sum(r['A_InDrawer_emptyhand_target_table']for r in matching),'old_same_env_placeA_liftB_transitions':sum(r['preceding_successful_place_A']and r['A_InDrawer_emptyhand_target_table']for r in matching),'old_command_motif_episodes_any_environment':skillseq[motif],'old_exact_six_action_episodes':0,'old_initial_placeA_then_liftB_prefix':0})
 # Directly check exact initial requested prefix, rather than infer from episode length alone.
 for cov in coverage:
  count=0
  for eid,rr in eps.items():
   rr.sort(key=lambda r:r['step']);l=read(rr[0]['labels_path'])
   if rr[0]['environment']==cov['environment'] and rr[0]['skill']=='place_in_drawer' and rr[1]['skill']==f"lift_{cov['target_B']}_block_table" and l['before']['categorical'][COLORS.index(cov['initial_A'])]==6:count+=1
  cov['old_initial_placeA_then_liftB_prefix']=count
 csvwrite(U/'results/old_coverage.csv',coverage);csvwrite(U/'results/old_supported_lifts.csv',oldlift)
 boundaries=[];object_errors=[];updates=[];arrays={}
 for ref in read(O/'results/prediction_references.json'):
  if ref['variant']not in ['control','updated']or ref['split']not in ['train','development']:continue
  path=Path(ref['path']);refs[str(path)]=sha(path);assert refs[str(path)]==ref['sha256']
  with np.load(path)as z:d={k:z[k]for k in z.files}
  arrays[ref['variant'],ref['seed'],ref['split']]=d
 for seed in [0,1]:
  for split in ['train','development']:
   d=arrays['control',seed,split];n=arrays['updated',seed,split]
   assert all(np.array_equal(d[k],n[k])for k in ['root_id','episode_id','branch','mode','step','cat','cat_valid'])
   p,q=rel(d['prob']),rel(n['prob']);y=truth(d['cat']);rp,rq=right(p,y),right(q,y)
   for i in np.where(d['mode']=='current')[0]:
    rid=str(d['root_id'][i]);root=roots[rid];branch=int(d['branch'][i]);t=int(d['step'][i]);a=root['held_color'];b=root['primary_color']if branch==0 else root['matched_color'];third=next(c for c in COLORS if c not in [a,b]);target=a if t==1 else b if t in [2,3,4,5]else None
    row={'seed':seed,'split':split,'root_id':rid,'environment':root['environment'],'branch':branch,'step':t,'initial_A':a,'requested_B':b,'third_object':third,'control_complete':bool(rp[i].all()),'updated_complete':bool(rq[i].all()),'new_complete_regression':bool(rp[i].all()and not rq[i].all()),'new_complete_recovery':bool(not rp[i].all()and rq[i].all())};boundaries.append(row)
    for j,c in enumerate(COLORS):
     role='A_initially_held'if c==a else'B_requested'if c==b else'C_other'
     for relation,k in [('Held',j),('InDrawer',j+3)]:
      object_errors.append({**{k:row[k]for k in ['seed','split','root_id','branch','step']},'color':c,'role':role,'relation':relation,'truth':bool(y[i,k]),'control_probability':float(p[i,k]),'updated_probability':float(q[i,k]),'control_correct':bool(rp[i,k]),'updated_correct':bool(rq[i,k]),'regression':bool(rp[i,k]and not rq[i,k]),'recovery':bool(not rp[i,k]and rq[i,k])})
    if t:
     prior=np.where((d['episode_id']==d['episode_id'][i])&(d['mode']=='one_step')&(d['step']==t))[0][0];prev=np.where((d['episode_id']==d['episode_id'][i])&(d['mode']=='current')&(d['step']==t-1))[0][0]
     for j,c in enumerate(COLORS):
      if c==target:continue
      for relation,k in [('Held',j),('InDrawer',j+3)]:
       if y[prev,k]!=y[i,k]:continue
       for variant,prob,correct in [('control',p,rp),('updated',q,rq)]:
        updates.append({'variant':variant,'seed':seed,'split':split,'root_id':rid,'branch':branch,'step':t,'action_object':target,'protected_object':c,'role':'A_initially_held'if c==a else'B_requested'if c==b else'C_other','relation':relation,'truth':bool(y[i,k]),'prior':float(prob[prior,k]),'posterior':float(prob[i,k]),'prior_correct':bool(correct[prior,k]),'posterior_correct':bool(correct[i,k]),'observation_regression':bool(correct[prior,k]and not correct[i,k]),'observation_recovery':bool(not correct[prior,k]and correct[i,k])})
 csvwrite(U/'results/boundary_comparisons.csv',boundaries);csvwrite(U/'results/object_relation_comparisons.csv',object_errors);csvwrite(U/'results/nonaction_observation_changes.csv',updates)
 summary=[]
 for split in ['train','development']:
  for seed in [0,1]:
   for branch in [0,1,2]:
    rr=[r for r in boundaries if r['split']==split and r['seed']==seed and r['branch']==branch];summary.append({'split':split,'seed':seed,'branch':branch,'boundaries':len(rr),'control_correct':sum(r['control_complete']for r in rr),'updated_correct':sum(r['updated_complete']for r in rr),'extra_regressions':sum(r['new_complete_regression']for r in rr),'recoveries':sum(r['new_complete_recovery']for r in rr)})
 collateral=[]
 for split in ['train','development']:
  for seed in [0,1]:
   for branch in [0,1,2]:
    for variant in ['control','updated']:
     rr=[r for r in updates if r['split']==split and r['seed']==seed and r['branch']==branch and r['variant']==variant];positive=[r for r in rr if r['relation']=='InDrawer'and r['truth']]
     collateral.append({'split':split,'seed':seed,'branch':branch,'variant':variant,'stable_nonaction_relation_opportunities':len(rr),'observation_regressions':sum(r['observation_regression']for r in rr),'observation_recoveries':sum(r['observation_recovery']for r in rr),'positive_In_opportunities':len(positive),'positive_In_regressions':sum(r['observation_regression']for r in positive),'positive_In_recoveries':sum(r['observation_recovery']for r in positive),'positive_In_posterior_correct':sum(r['posterior_correct']for r in positive)})
 csvwrite(U/'results/summary.csv',summary);csvwrite(U/'results/collateral_summary.csv',collateral)
 save(U/'results/audit.json',{'utc':now(),'started':start,'old_roots':40,'old_episodes':len(eps),'old_transitions':len(oldrows),'old_skill_sequences':[{'skills':list(k),'episodes':v}for k,v in skillseq.items()],'coverage':coverage,'summary':summary,'collateral':collateral,'scope':'Train and development only. Read-only saved predictions/metadata. Identical public observations across models, but learned E/history/weights differ; comparison is total training change, not isolated residual causality. Prior-to-posterior changes inside each model describe actual observation updates. No new forwards or state splice.','training_updates':0,'new_simulation':0})
 save(U/'configs/input_hashes.json',refs);print(json.dumps({'summary':summary,'collateral':collateral,'coverage':coverage},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
