from pathlib import Path
import os,sys,json,hashlib,datetime,csv,itertools
from collections import Counter,defaultdict
import numpy as np
P=Path('/home/robot/research/projects/wm_symbolic')
OLD=P/'experiments/rssm_encoder_gradient_20261019'
X=P/'experiments/rssm_scaling_design_audit_20261019'
R=P/'reports/rssm_scaling_design_audit_20261019'
SALT='rssm-root-scale-proposal-20261019-v1:'
ARMS=['joint','route_current_only']
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,default=lambda v:v.item()if isinstance(v,np.generic)else v.tolist())+'\n')
def writecsv(p,rows):
 keys=list(dict.fromkeys(k for r in rows for k in r))
 with Path(p).open('w',newline='')as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def guard():
 def hook(event,args):
  if event!='open'or not isinstance(args[0],(str,bytes)):return
  p=Path(os.fsdecode(args[0])).resolve();mode=args[1];flags=args[2]if len(args)>2 else 0
  writing=(isinstance(mode,str)and any(c in mode for c in'wa+'))or(isinstance(flags,int)and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
  if writing and p.is_relative_to(P)and not(p.is_relative_to(X)or p.is_relative_to(R)):raise RuntimeError('Historical write forbidden '+str(p))
  if p.is_relative_to(P/'data/core1200_v1')and any(t.startswith('D_core1200')for t in p.parts):raise RuntimeError('D sample forbidden')
 sys.addaudithook(hook)

def design():
 rootlist=[r for r in read(P/'data/core1200_v1/roots_manifest.json')if r['environment']in ['A','B','C']];rootmeta={r['root_id']:r for r in rootlist};g=defaultdict(list)
 for r in rootlist:
  if r['environment']in ['A','B']:g[r['environment'],r['initial_state']].append(r)
 parts={k:[]for k in ['train20','train40','validation10','test10']}
 for (env,init),rr in sorted(g.items()):
  rr.sort(key=lambda r:hashlib.sha256((SALT+r['root_id']).encode()).hexdigest());nt={'closed_empty':3,'open_empty':1,'open_held':1}[init];nv={'closed_empty':2,'open_empty':2,'open_held':1}[init];ns={'closed_empty':5,'open_empty':3,'open_held':2}[init]
  parts['test10']+=rr[:nt];parts['validation10']+=rr[nt:nt+nv];parts['train40']+=rr[nt+nv:];parts['train20']+=rr[nt+nv:nt+nv+ns]
 parts['C20']=[r for r in rootlist if r['environment']=='C'];members={k:sorted(r['root_id']for r in rr)for k,rr in parts.items()}
 assert set(members['train20'])<set(members['train40'])
 for a,b in itertools.combinations(['train40','validation10','test10','C20'],2):assert not(set(members[a])&set(members[b]))
 assert len(set(members['train40']+members['validation10']+members['test10']))==60
 # Commit memberships before any saved-prediction analysis or model accuracy access.
 proposal={'utc':now(),'status':'PROPOSED_NOT_EXECUTED','selection':'Single fixed SHA256 ordering within environment and initial-state strata. No model prediction or metric used; no salt search.','salt':SALT,'members':members,'train_sizes':[20,40],'primary_evaluation':'same10 held-out A/B roots for every new fit','validation':'same10 A/B roots; descriptive monitoring only, fixedfinal6000, no checkpoint or threshold choice','external_diagnostic':'same20 already-seen C roots; not pure scene-shift causal estimate','initial_weights':'fresh random paired initial state for seeds0/1, no old trained encoder/RSSM/query warmstart','normalization':'Recompute visual/robot normalization and symbol classweights ONLY on commontrain20; freeze and reuse across20/40 and botharms. Never reuse old60-root normalization. Raw frozenDINO caches may be reused.','probe':'Same commontrain20 labeled boundary pool300records for all8probes; identical paired600x32 schedules and init; per-model E mean/std only on these same300records; final600.','main_fits':8,'main_updates_each':6000,'total_main_updates':48000,'probe_fits':8,'probe_updates_each':600,'total_probe_updates':4800,'main_schedule':'Original5000 one-step+1000 three-step, microbatch8x2, pairedseeds0/1, AdamW/warmup/cosine and independentE/Rclip1 as completedexperiment. Final6000only.','same_compute':'Eachsize gets96000 sampled4-request episodes;20roots=>1600 effectiveepisodepasses,40roots=>800. This compares diversity at fixedupdates/compute, not equalexposureperroot.','same_exposure':'NOT included.20roots3000 vs40roots6000 would equal expectedpasses but differ compute and require predeclared phase/LRschedule normalization. Old3000checkpoint lacks multisteploss, so cannot be silently used for this control.','estimand':'For eachseed/arm, paired root differences40minus20 on sameAB10; compare improvement differences betweenarms. Probe andfree3primary NLL plus fullstate/existingrelationmetrics. Original60-root models are historical only; they trained on prospectiveABheldout roots.','uncertainty':'Two trainingseeds andonefixednestedsubset per size; onlyone20-to40 contrast, no scalinglaw, no extrapolation, no claim of subset-selection robustness. Whole-root pairedbootstrap2000; roots/branches kept together.','disclosure':'All data/labels andC were seen in previous work. FutureABheldout is unseen only to this round optimization, not a freshblindtest.','new_fits_executed':0,'new_collection':0,'D_samples':0}
 save(X/'configs/proposed_protocol.json',proposal)
 manifest=read(P/'data/core1200_v1/data_manifest.json');byroot=defaultdict(list)
 for row in manifest['records']:
  if row['split']in ['train','dev']:byroot[row['root_id']].append(row)
 labels={};coverage=[];sequence_sets={};inputhashes=defaultdict(list);positions={};publickeys=['rgb_static','rgb_gripper','robot_obs']
 for rid,rr in byroot.items():
  rr.sort(key=lambda r:(r['episode_id'],r['step']));labels[rid]=[read(row['labels_path'])for row in rr]
  first=next(row for row in rr if row['episode_id'].endswith('_b0')and row['step']==0);lab=read(first['labels_path']);positions[rid]=np.array([lab['before']['continuous']['objects'][c]['position_m']for c in ['red','blue','pink']])
  h=hashlib.sha256()
  with np.load(first['obs_before'])as a:
   for key in publickeys:h.update(key.encode());h.update(a[key].tobytes())
  inputhashes[h.hexdigest()].append(rid)
 for name,rr in parts.items():
  ids=members[name];rows=[row for rid in ids for row in byroot[rid]];labs=[lab for rid in ids for lab in labels[rid]];seq={tuple(b)for r in rr for b in r['branches']};sequence_sets[name]=seq
  byep=defaultdict(list)
  for rid in ids:
   for row,lab in zip(byroot[rid],labels[rid]):byep[row['episode_id']].append(lab)
  counts={'group':name,'roots':len(ids),'episodes':len(byep),'transitions':len(rows),'boundary_records':len(byep)*5,'environments':dict(Counter(r['environment']for r in rr)),'initial_states':dict(Counter(r['initial_state']for r in rr)),'initial_held_colors':dict(Counter(r['held_color']for r in rr if r['held_color'])),'skills':dict(Counter(row['skill']for row in rows)),'nonzero':sum(l['control_samples_including_start']>1 for l in labs),'zero_control':sum(l['control_samples_including_start']==1 for l in labs),'success':sum(l['success']for l in labs),'executed_failure':sum(not l['success']and l['control_samples_including_start']>1 for l in labs),'rootstart3_all_actuated':sum(all(l['control_samples_including_start']>1 for l in ls[:3])for ls in byep.values()),'four_request_sequences':len(seq),'changed_field_counts':[sum(l['before']['categorical_valid'][j]and l['after']['categorical_valid'][j]and l['before']['categorical'][j]!=l['after']['categorical'][j]for l in labs)for j in range(5)],'post_Holding_positive_by_color':[sum(l['after']['categorical'][j]==6 for l in labs)for j in range(3)],'post_In_positive_by_color':[sum(l['after']['categorical'][j]==3 for l in labs)for j in range(3)]}
  coverage.append(counts)
 duplicates=[v for v in inputhashes.values()if len(v)>1];near=[]
 for a,b in itertools.combinations(sorted(positions),2):
  if a[0]==b[0]and max(np.linalg.norm(positions[a]-positions[b],axis=1))<=.012:near.append([a,b])
 groupof={r:g for g in ['train40','validation10','test10','C20']for r in members[g]};cross_exact=[v for v in duplicates if len({groupof[r]for r in v})>1];cross_near=[v for v in near if groupof[v[0]]!=groupof[v[1]]]
 assert not cross_exact and not cross_near,'Leakage candidate requires root grouping before any fitting'
 # Contextual composition differs even where action-only strings overlap.
 context={}
 for group in ['train40','test10','C20']:
  pairs=Counter()
  for r in parts[group]:
   if r['initial_state']=='open_held':
    final=r['branches'][0][-1];other=next(c for c in ['red','blue','pink']if '_'+c+'_'in final);pairs[r['held_color']+'→'+other]+=1
  context[group]=dict(pairs)
 fitsecs=[read(f)['elapsed_seconds_this_segment']for f in(OLD/'results').glob('train_*_seed*.json')]
 out={'utc':now(),'proposal_sha256':sha(X/'configs/proposed_protocol.json'),'coverage':coverage,'heldout_action_sequences_unseen_in_common20':{g:len(sequence_sets[g]-sequence_sets['train20'])for g in ['test10','validation10','C20']},'contextual_initial_held_to_second_object':context,'root_public_duplicate_groups':duplicates,'same_environment_all3positions_within12mm':near,'cross_split_exact_or12mm_matches':0,'duplicate_check_limit':'Exact2RGB+robot arrays and12mm all3 namedpositions checked. Previous fullnative12/20mm+yaw audit recorded no duplicates; this followup doesnotclaim new exhaustive visual-nearduplicate certification.','AB_same_index': 'A/B root seeds differ by1000 and layouts differ; same numeric suffix is not a shared physical root. Splitting uses entire actualroot, not action-script identity.','runtime_main_seconds_estimate':8*np.mean(fitsecs),'runtime_reference_seconds':fitsecs,'remaining_main_fit_budget_minutes':8*np.mean(fitsecs)/60,'current_optimizer_updates':0,'current_model_inference':0,'old20_40_60_plan_limit':'Fixed C only, noABholdout.60 includes allAB and cannot appear in prospective sameABtest comparison. Saved historical60 scores must not be relabeledtest.'}
 save(X/'results/design_feasibility.json',out)
 writecsv(R/'root_memberships.csv',[{'partition':g,'root_id':rid,'environment':rootmeta[rid]['environment'],'initial_state':rootmeta[rid]['initial_state'],'seed':rootmeta[rid]['seed']}for g,ids in members.items()for rid in ids])
 save(X/'configs/design_before_prediction_analysis.json',{'utc':now(),'proposal_sha256':sha(X/'configs/proposed_protocol.json'),'memberships_sha256':sha(R/'root_memberships.csv'),'selection_did_not_read_predictions':True})
 return rootmeta,proposal,out

def mean_root(v,valid,roots):
 return {r:float(v[(roots==r)&valid].mean())for r in sorted(set(roots))if ((roots==r)&valid).any()}
def ci(v):
 a=np.array(list(v),float)
 if not len(a):return None,None,None
 rng=np.random.default_rng(261019901);boot=a[rng.integers(len(a),size=(2000,len(a)))].mean(1)
 return float(a.mean()),float(np.quantile(boot,.025)),float(np.quantile(boot,.975))
def analysis(rootmeta):
 output=[];rootrows=[];confusion=[];initial=[];alignment=[]
 for seed in [0,1]:
  for arm in ARMS:
   for split in ['train','dev']:
    for prefix in ['predictions','probe_predictions']:
     a=dict(np.load(OLD/f'data/{prefix}_{arm}_seed{seed}_{split}.npz'));roots=a['root_id'];p=a['prob'];cat=a['cat'];valid=a['cat_valid'];n=len(p);rownum=np.arange(n)
     pred=np.stack([p[:,s].argmax(-1)for s in [slice(0,8),slice(8,16),slice(16,24),slice(24,27),slice(27,30)]],1)
     hold=p[:,[6,14,22]]>=.5;ins=p[:,[3,11,19]]>=.5;hy=cat[:,:3]==6;iy=cat[:,:3]==3;drawer=pred[:,3]==cat[:,3]
     catnll=np.stack([-np.log(np.clip(p[rownum,off+cat[:,j]],1e-9,1))for j,off in enumerate([0,8,16,24,27])],1)
     facts=[]
     for j,name in enumerate(['red_location','blue_location','pink_location','drawer','slider']):facts.append((name,(pred[:,j]==cat[:,j]).astype(float),catnll[:,j],valid[:,j],cat[:,j]!=a['base_cat'][:,j],a['base_valid'][:,j]))
     for j,c in enumerate(['red','blue','pink']):
      for name,k in [('Holding',6),('In',3)]:
       y=cat[:,j]==k;q=p[:,8*j+k];loss=-(y*np.log(np.clip(q,1e-9,1))+(~y)*np.log(np.clip(1-q,1e-9,1)));facts.append((name+'_'+c,((q>=.5)==y).astype(float),loss,valid[:,j],y!=(a['base_cat'][:,j]==k),a['base_valid'][:,j]))
     facts.extend([('primary4_mean_accuracy',(pred[:,:4]==cat[:,:4]).mean(1),catnll[:,:4].mean(1),valid[:,:4].all(1),None,None),('all5_exact',(pred==cat).all(1).astype(float),None,valid.all(1),None,None),('all7_exact',((pred==cat).all(1)&((p[:,30:]>=.5)==a['binary']).all(1)).astype(float),None,valid.all(1)&a['binary_valid'].all(1),None,None),('relation9_exact',((hold==hy).all(1)&(ins==iy).all(1)&drawer).astype(float),None,valid[:,:4].all(1),None,None),('multiple_holding',((hold.sum(1)>1)).astype(float),None,valid[:,:3].all(1),None,None)])
     initialtype=np.array([rootmeta[r]['initial_state']for r in roots])
     for mode in ['current','one_step','free1','free3','free4','probe']:
      base=a['mode']==mode
      if not base.any():continue
      strata={'all':np.ones(n,bool),'initial_closed_empty':initialtype=='closed_empty','initial_open_empty':initialtype=='open_empty','initial_open_held':initialtype=='open_held','initial_not_held':initialtype!='open_held'}
      if mode not in ['current','probe']:strata.update(zero_control=a['controls']==0,nonzero_control=a['controls']>0,success=a['status']=='success',executed_failure=a['status']=='executed_failure')
      if mode.startswith('free'):strata['all_prefix_actuated']=a['all_prefix_actuated']
      for st,ss in strata.items():
       sel=base&ss
       for fact,accuracy,loss,ok,changed,bok in facts:
        changes=['all','changed','unchanged']if changed is not None and st=='all'and mode in ['one_step','free3']else['all']
        for change in changes:
         mask=sel&ok
         if change!='all':mask=mask&bok&(changed if change=='changed'else~changed)
         key={'seed':seed,'arm':arm,'split':split,'mode':mode,'stratum':st,'fact':fact,'change':change}
         for metric,value in [('accuracy',accuracy),('NLL',loss)]:
          if value is None:continue
          rm=mean_root(value,mask,roots);m,lo,hi=ci(rm.values());output.append({**key,'metric':metric,'records':int(mask.sum()),'roots':len(rm),'value':m,'lo95':lo,'hi95':hi,'micro':float(value[mask].mean())if mask.any()else None})
          for rid,v in rm.items():rootrows.append({**key,'metric':metric,'root_id':rid,'value':v})
       for relation,ys,qs in [('Holding',hy,p[:,[6,14,22]]),('In',iy,p[:,[3,11,19]])]:
        mask=sel[:,None]&valid[:,:3];preds=qs>=.5
        tp=int((mask&preds&ys).sum());fp=int((mask&preds&~ys).sum());fn=int((mask&~preds&ys).sum());tn=int((mask&~preds&~ys).sum())
        precision=tp/(tp+fp)if tp+fp else None;recall=tp/(tp+fn)if tp+fn else None;spec=tn/(tn+fp)if tn+fp else None
        confusion.append({'seed':seed,'arm':arm,'split':split,'mode':mode,'stratum':st,'relation':relation,'root_count':len(set(roots[sel])),'valid_binary_answers':int(mask.sum()),'TP':tp,'FP':fp,'FN':fn,'TN':tn,'precision':precision,'recall':recall,'BA':(recall+spec)/2 if recall is not None and spec is not None else None})
     if prefix=='predictions':
      for eid in sorted(set(a['episode_id'])):
       i=np.where((a['episode_id']==eid)&(a['mode']=='current')&(a['step']==0))[0][0];j=np.where((a['episode_id']==eid)&(a['mode']=='free3'))[0][0]
       alignment.append({'seed':seed,'arm':arm,'split':split,'root_id':roots[i],'episode_id':eid,'root_current_all5':bool((pred[i]==cat[i]).all()),'free3_all5':bool((pred[j]==cat[j]).all()),'root_current_all9':bool((hold[i]==hy[i]).all()and(ins[i]==iy[i]).all()and drawer[i]),'free3_all9':bool((hold[j]==hy[j]).all()and(ins[j]==iy[j]).all()and drawer[j])})
 writecsv(R/'saved_prediction_strata.csv',output);writecsv(R/'relation_confusions.csv',confusion);writecsv(R/'root_current_to_free3.csv',alignment)
 keys=['seed','split','mode','stratum','fact','change','metric'];groups={}
 for r in rootrows:groups.setdefault(tuple(r[k]for k in keys),{}).setdefault(r['arm'],{})[r['root_id']]=r['value']
 pairs=[]
 for key,d in groups.items():
  if not all(a in d for a in ARMS):continue
  assert set(d[ARMS[0]])==set(d[ARMS[1]])
  v,lo,hi=ci(d['joint'][r]-d['route_current_only'][r]for r in sorted(d['joint']));pairs.append({**dict(zip(keys,key)),'roots':len(d['joint']),'joint_minus_control':v,'lo95':lo,'hi95':hi})
 writecsv(R/'saved_prediction_paired.csv',pairs)
 save(X/'results/analysis_summary.json',{'utc':now(),'rows':len(output),'paired_rows':len(pairs),'relations_rows':len(confusion),'new_fits':0,'new_model_inference':0,'D_samples':0,'does_not_score_proposed_ABheldout_with_old_models':True,'relation_definition':'Holding andIn from objectclass marginals threshold0.5, drawer3argmax; ninebinaryfacts have onlyseven degrees of labeled groups (3Holding+3In+1drawer categorical). SameobjectHolding/In sharethe8waysoftmax probabilitybudget (theirsum<=1); this outputconstraint isnot learnedconsistency evidence.','rollout_definition':'free3 is endpoint after3requestedskills fromroot with8sample probabilities averaged beforeNLL. Not all3boundaries correct. Savedfree2 absent, do not invent fullprefix accuracy.'})
 return output,confusion,pairs,alignment

def main():
 guard();meta,proposal,feasibility=design();out,conf,pairs,align=analysis(meta)
 print(json.dumps({'coverage':feasibility['coverage'],'sequence_support':feasibility['heldout_action_sequences_unseen_in_common20'],'context_pairs':feasibility['contextual_initial_held_to_second_object'],'exact_duplicates':feasibility['root_public_duplicate_groups'],'near12mm':feasibility['same_environment_all3positions_within12mm'],'budget_main_minutes':feasibility['remaining_main_fit_budget_minutes'],'prediction_rows':len(out),'new_fits':0},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
