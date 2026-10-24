from common import *
import shutil,collections,gzip
def main():
 for p in [E/'source',E/'configs',E/'data/public',E/'data/_private',E/'models',E/'results',E/'logs',R]:p.mkdir(parents=True,exist_ok=True)
 for p,h in read(REVIEW/'configs/proposal_freeze.json')['sha256'].items():assert sha(p)==h,p
 cfg=read(REVIEW/'configs/proposed_protocol.json');cfg['status']='authorized implementation; freeze before optimizer updates';cfg['current_execution']={'training_updates':0,'new_physical_calls':0}
 cfg['execution_limits']={'reset_branches':{'reference':'Use root drawer_reset_m; no skill preparation.','held_open':'Reset q=.22 then exactly one lift_target call plus96 hold.','held_closed':'Reset q=0 then exactly one lift_target call plus96 hold.','empty_open':'Reset q=.22; no preparation skill.','empty_closed':'Reset q=0; no preparation skill.'},'reset_settle_controls':127,'preparation_calls_total_max':24,'preparation_controls_per_call_max':900,'posthold_controls':96,'internal_executor_attempts_max':2,'external_retries':0,'prep_failure':'If native preparation is not success, or after hold evaluator condition is unmet, stop branch and count task as unreached_preparation; no further preparation/reset/replacement.','reference_failure':'Stop remaining reference skills after first non-success; retain unreached task slots.','matched_task_failure':'Exactly one place attempt if preparation valid; no retry.','closed_loop_failure':'Replan from actual observation after every call, including refusal, at most6; no special feasibility filter; infrastructure exception ends episode and is retained.','offline_task_slots':90,'closed_loop_task_calls_max':144,'all_task_calls_max':234,'all_primitive_calls_including_preparation_max':258,'episodes_total':84,'max_control_steps_including_holds_and_settle':267636,'max_physics_ticks':2141088,'infrastructure_failure':'Preserve partial episode; stop orchestration rather than silently replay. Completed episodes never rerun.'}
 cfg['budget_estimate']={'device':'single local RTX5080Laptop16GB; exclusive runs/gpu.lock','GPU_minutes_estimate':25,'wall_minutes_estimate':90,'wall_minutes_soft_ceiling':150,'storage_GiB_estimate':5,'basis':'4 small transition-only1000-update runs,36 added RGB endpoints,90 main skill slots,24 closed-loop rollouts; one DINO batch8 per boundary; continuous RGB streamed to compressed video. Estimates are not measured results.','serial_queue':['AB_C_features','4 training runs','C evaluation','D2 collection','D2 feature cache and offline evaluation','closed_loop_with_one_inference_worker','report'],'continuous_video':'Every recorded control frame at30fps (240Hz physics /8), dual camera; simulation time excludes model waiting. No interpolated frames.'}
 cfg['extra_C_label_limit']='Old matched records lack object orientation required for v2.1 contained-supported In. Score Holding and drawer only (6 valid facts), mask all3 In, explicitly separate from original full9-fact C subset. D2 newly records full physical snapshot and all9. No labels enter training.'
 save(E/'configs/protocol.json',cfg);shutil.copy2(REVIEW/'configs/proposed_D2_roots.json',E/'configs/D2_roots.json');shutil.copy2(REVIEW/'configs/proposed_D2_exclusion.json',E/'configs/D2_root_exclusion.json')
 for fn in ['normalization.npz','inputs.npz']:shutil.copy2(OLD/'data/public'/fn,E/'data/public'/('old_'+fn if fn=='inputs.npz'else fn))
 states=read(OLD/'data/public/states.json');pairs=read(OLD/'data/public/pairs.json');chains=read(OLD/'data/public/chains.json');labels=read(OLD/'data/_private/labels.json');oldout=read(OLD/'data/_private/pair_outcomes.json');status={x['pair_id']:x for x in oldout};extra=read(REVIEW/'results/extra_matched_place.json');raw=read(P/'data/predicate_state_dependence_20261014/data_manifest.json');private={r['pair_id']:r for r in raw['private_pairs']};paths={r['path']:i for i,r in enumerate(states)};first_extra=len(states);outcomes={}
 for i,r in enumerate(pairs):
  r['dataset_group']='original';r['action']=action(r['skill'],None,round(r['action'][7]*96));r['index']=i;outcomes[r['id']]=status[r['id']]
 for r in extra:
  inds=[];pr=private[r['id']];trace=json.load(gzip.open(P/'data/predicate_state_dependence_20261014/episodes'/r['episode_id']/'_private/trace.json.gz','rt'))
  for key in ['path','after_path']:
   path=r[key];assert path not in paths;i=len(states);paths[path]=i;inds.append(i);states.append({'id':f's{i:05d}','root_id':r['root_id'],'split':r['split'],'path':path,'dataset':'matched_place'})
   frame=int(Path(path).stem);tr=trace[frame];q=tr['drawer_m'];held=tr['labels']['holding_derived'];dy=[q<=.02,q>=.18,.02<q<.18];valid=min(abs(q-.02),abs(q-.18))>.003;labels[str(i)]={'y':held+[False]*3+dy,'v':[True]*3+[False]*3+[valid]*3,'In_unavailable_reason':'original trace has no full object orientations; never substitute legacy7-label In','drawer_q_m':q}
  i=len(pairs);pid=f'p{i:05d}';pairs.append({'id':pid,'index':i,'source_pair_id':r['id'],'root_id':r['root_id'],'split':r['split'],'dataset_group':'matched','before':inds[0],'after':inds[1],'skill':'place_in_drawer','action':action('place_in_drawer',None,96)});chains.append([pid]);outcomes[pid]={'pair_id':pid,'status':r['status'],'condition':r['condition'],'target':r['target_intent_evaluator_only'],'reason':r['reason']}
 save(E/'data/public/states.json',states);save(E/'data/public/pairs.json',pairs);save(E/'data/public/chains.json',chains);save(E/'data/_private/labels.json',labels);save(E/'data/_private/outcomes.json',outcomes)
 def groups(ix):return [[i for i in ix if pairs[i]['root_id']==r]for r in sorted({pairs[i]['root_id']for i in ix})]
 base=[i for i,r in enumerate(pairs)if r['split']=='train'and r['dataset_group']=='original'];control=[i for i in base if pairs[i]['skill']=='place_in_drawer'];matched=[i for i,r in enumerate(pairs)if r['split']=='train'and r['dataset_group']=='matched'];assert len(base)==888 and len(matched)==12
 for seed in [0,1]:
  rng=np.random.default_rng(16962000+seed)
  def sample(gs,n):return np.array([g[rng.integers(len(g))]for g in[gs[j]for j in rng.integers(len(gs),size=n)]])
  bb=np.stack([sample(groups(base),48)for _ in range(1000)]);cc=np.stack([sample(groups(control),16)for _ in range(1000)]);mm=np.stack([sample(groups(matched),16)for _ in range(1000)]);seq=np.full((1000,48,3),-1,np.int64)
  for u in range(1000):
   for j,i in enumerate(bb[u]):
    for t,pid in enumerate(chains[i][:3]):seq[u,j,t]=int(pid[1:])
  for u in range(1000):
   for j in range(48):
    for t in [1,2]:
     if seq[u,j,t]>=0:assert pairs[seq[u,j,t-1]]['after']==pairs[seq[u,j,t]]['before']
  np.savez_compressed(E/'configs'/f'schedule_seed{seed}.npz',base=bb,control_aux=cc,matched_aux=mm,observed_chains=seq)
  for part,dest in [('',f'initial_seed{seed}.pt'),('queries',f'query_seed{seed}.pt')]:
   sel=read(OLD/'models'/f'interaction_seed{seed}'/part/'selected.json');assert sha(sel['path'])==sel['sha256'];shutil.copy2(sel['path'],E/'models'/dest)
 save(E/'results/preparation.json',{'old_states':first_extra,'new_public_endpoints':len(states)-first_extra,'original_AB_pairs':len(base),'matched_AB_pairs':len(matched),'C_pairs':sum(r['split']=='dev'for r in pairs),'C_roots':len({r['root_id']for r in pairs if r['split']=='dev'}),'same_base_and_chain_schedule_both_arms':True,'source_review_seal_sha256':sha(REVIEW/'configs/review_delivery_manifest.json'),'source_old_seal_sha256':sha(OLD/'configs/delivery_manifest.json'),'all_place_targets_zero':all(not any(r['action'][4:7])for r in pairs if r['skill']=='place_in_drawer')});print('SETUP_COMPLETE',read(E/'results/preparation.json'),flush=True)
if __name__=='__main__':main()
