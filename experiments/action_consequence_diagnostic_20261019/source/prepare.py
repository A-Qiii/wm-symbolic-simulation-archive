from common import *
import collections,yaml
def main():
 sys.addaudithook(guard);assert not(S/'configs/protocol.json').exists();start=time.monotonic()
 for d in ['configs','data','results','models','figures','logs','cache']:(S/d).mkdir(parents=True,exist_ok=True)
 protocol={'utc':now(),'status':'FROZEN_BEFORE_NEW_FITS','goal':'Separate current task-relation recognition errors from action-consequence errors, including held identity, drawer membership/state, actual success/refusal and non-target movement','data':'Existing core1200_v1, fixed A/B train40 validation10 test10; same 3 branches x4 actions; no collection; no C/D','visual_models':'Two frozen roots40 joint RSSMs; reuse cached current/one_step/free1..4 predictions. Test includes existing h2 supplement; no encoder forward or RSSM fit','truth_inputs':{'relations':'held red/blue/pink + in-drawer red/blue/pink + drawer3 onehot, plus7 known flags','physics_enriched':'same relation state plus before-only object xyz/quaternion/linear/angular velocities, robot proprioception, door/button/switch/light scalars, contact summaries, static scene surface coordinates'},'physics_limit':'Enriched recorded boundary state, not proven complete Markov state: robot joint velocities, controller internal/collision-solver state not recorded. Do not label remaining errors as dynamics inability.','architecture':'Identical padded input dimension and parameter count: initial Linear->SiLU->Linear(128), GRUCell(action7,128), Linear128->SiLU->Linear15; relation arm zeros all physical dimensions. Output H3,I3,drawer3,status3,moved3.','updates':1500,'seeds':[0,1],'arms':['relations','physics_enriched'],'fits':4,'total_updates':6000,'batch_episodes':32,'optimizer':{'name':'AdamW','lr':.0003,'weight_decay':.0001,'grad_clip':1.},'loss':'Equal .5 teacher-reset one-step and .5 free4; within each: (heldBCE+inBCE+drawerCE)/3 + .5 statusCE + .25 movedBCE; validity masked, no class weighting','checkpoints':[750,1500],'selection':'Per arm and seed select lower fixed validation objective (.5 one-step + .5 free4), tie earlier; freeze selected hashes before test. No refit/search beyond2 prespecified checkpoints.','sampling':'Uniform training root then one of3 branches; seeded materialized1500x32 schedule shared arms within seed','normalization':'Physical mean/std from train action-before states t0..3 only; min std .001; contacts force log1p; no future/label/status/actual-duration input','known_actions':'Six declared skills one-hot plus planned budget/900, no actual success/control count/selected target oracle','extra_no_fit_control':'Reuse relation model with hard visual current relations substituted for GT initial relations, same weights/action sequence; pairing measures input sensitivity within this model, not pure perception contribution in original RSSM','metrics':'Per relation exact, changed-state exact, positive holding/in-drawer recall, action success Brier/recall/false-success-on-refused, non-target moved recall/FPR; root-cluster2000 intervals; strata visual initial relation correct vs wrong, actual success/refused/executed_failure','state_sufficiency':'Group same before coarse relation+action and full symbolic category+action; count conflicting observed successors/status on train; retain geometry-differing examples, not counterfactual causal effects','horizons':[1,2,3,4],'interpretation':'Current only-correct relation stratum does not guarantee latent/geometry correctness. Cross-architecture score gaps are not pure sensory causal contributions. Open-loop replayed actions are not new closed-loop planning.','limits':{'compute_seconds':1200,'round_target_minutes':30,'new_RSSM_fits':0,'new_encoder_forwards':0,'new_collection':0,'uploads':0},'no_post_test_tuning':True}
 save(S/'configs/protocol.json',protocol);protected={}
 def use(p):p=Path(p);protected[str(p)]=sha(p);return p
 inv=[{'candidate':'Existing SymbolTransition','source':str(P/'src/wm_symbolic/symbol_transition.py'),'finding':'Consumes learned symbolic probabilities from frozen perception, not GT state; older 25-skill/6-step dataset. Not a matching accurate-state baseline.'},{'candidate':'Copy-effect diagnostic 20261014','source':str(P/'experiments/state_predicates_copy_effect_diagnostic_20261014/source/diagnostic.py'),'finding':'AB96 GT labels induce unanimous changed-role effect table; inference starts from current learned readout; source explicitly says not failure/refusal dynamics. Different data, not suitable replacement.'},{'candidate':'Uninterrupted action prediction54','source':str(P/'reports/uninterrupted_action_prediction54_20261018/REPORT.md'),'finding':'Frozen coverage gate failed; report states zero model fits. No trained baseline to reuse.'},{'candidate':'roots40 RSSM and h2 supplement','source':str(CELL/'data'),'finding':'Existing same-root current/one-step/free1..4 and one-step success/movement probabilities reusable. No need to rerun perception.'}]
 for item in inv:
  path=Path(item['source'])
  if path.is_file():use(path)
 save(S/'results/baseline_inventory.json',inv)
 fixed=read(use(B/'configs/fixed_design.json'))['members'];members={s:fixed[{'train':'train40','validation':'validation10','test':'test10'}[s]]for s in SPLITS};allids=set(sum(members.values(),[]));rows=[r for r in read(use(P/'data/core1200_v1/data_manifest.json'))['records']if r['root_id']in allids];by=collections.defaultdict(list)
 for r in rows:by[r['episode_id']].append(r)
 configs={env:yaml.safe_load(use(P/f'vendor/calvin/calvin_env/conf/scene/calvin_scene_{env}.yaml').read_text())for env in ['A','B']};provenance=[];schema=[];counts=[];records=[];max_velocity=0.;imagekeys=[]
 def physical(info,env):
  vals=[];names=[]
  def add(prefix,v):
   for i,x in enumerate(v):vals.append(float(x));names.append(prefix+f'[{i}]')
  scene=info['scene_info'];rid=info['robot_info']['uid'];table=scene['fixed_objects']['table'];tid=table['uid'];drawerlink=table['links']['drawer_link'];planklink=table['links']['plank_link']
  for c in COLORS:
   o=scene['movable_objects']['block_'+c]
   for key in ['current_pos','current_orn','current_lin_vel','current_ang_vel']:
    a=np.asarray(o[key]);a=-a if key=='current_orn'and a[-1]<0 else a;add(c+'.'+key,a)
  robot=info['robot_info'];add('robot.tcp_pos',robot['tcp_pos']);add('robot.tcp_orn',robot['tcp_orn']);add('robot.gripper_width',[robot['gripper_opening_width']]);add('robot.joints',robot['arm_joint_states']);add('robot.gripper_action',[robot['gripper_action']])
  for cat in ['doors','buttons','switches','lights']:
   for name,v in sorted(scene[cat].items()):add(cat+'.'+name,[v['current_state']if 'current_state'in v else v['logical_state']])
  for c in COLORS:
   o=scene['movable_objects']['block_'+c];uid=o['uid'];force=np.zeros(6)
   for contact in o['contacts']:
    other,link=(contact[2],contact[4])if contact[1]==uid else(contact[1],contact[3]);j=0 if other==rid else 1 if other==tid and link==-1 else 2 if other==tid and link==drawerlink else 3 if other==tid and link==planklink else 4;force[j]+=max(0.,contact[9]);force[5]=max(force[5],max(0.,-contact[8]))
   force[:5]=np.log1p(force[:5]);add(c+'.contact_robot_base_drawer_plank_other_penetration',force)
  for key in ['table','slider_left','slider_right']:add('static_surface.'+key,np.asarray(configs[env]['surfaces'][key]).flatten())
  return np.asarray(vals,np.float32),names
 for split in SPLITS:
  items=[]
  for eid,rr in sorted(by.items()):
   if rr[0]['root_id']not in members[split]:continue
   rr=sorted(rr,key=lambda r:r['step']);assert [r['step']for r in rr]==[0,1,2,3];labels=[];infos=[];statuses=[];events=[];evvalid=[];frames=[];actions=[];actualcontrols=[]
   for t,r in enumerate(rr):
    labpath=use(r['labels_path']);lab=read(labpath);infopath=use(labpath.parent/'endpoint_info_private.json');info=read(infopath)
    if t==0:labels.append(lab['before']);infos.append(info['before']);frames.append(str(Path(r['obs_before']).parent.parent/'root_observation_3.npz'))
    else:
     assert labels[-1]['categorical']==lab['before']['categorical'];assert np.allclose([labels[-1]['continuous']['objects'][c]['position_m']for c in COLORS],[lab['before']['continuous']['objects'][c]['position_m']for c in COLORS],atol=1e-9)
    labels.append(lab['after']);infos.append(info['after']);frames.append(str(Path(r['obs_after']).parent/'boundary_history_3.npz'));controls=lab['control_samples_including_start']-1;status=0 if lab['success']else 1 if controls==0 else 2;statuses.append(status);actualcontrols.append(controls);events.append(lab['events']);evvalid.append(lab['events_valid']);act=np.zeros(7,np.float32);act[SKILLS.index(r['skill'])]=1;act[6]=r['executor']['budget_steps']/900;actions.append(act)
    result=read(use(labpath.parent/'result_private.json'));reason=result.get('reason',result.get('status',''));provenance.append({'split':split,'root_id':r['root_id'],'episode_id':eid,'t':t,'skill':r['skill'],'labels_path':str(labpath),'private_info_path':str(infopath),'before_public':r['obs_before'],'after_public':r['obs_after'],'status':STATUS[status],'controls':controls,'reason':reason,'result':result})
   cat=np.array([l['categorical'][:4]for l in labels]);v=np.array([l['categorical_valid'][:4]for l in labels]);y=rel_from_cat(cat);known=np.concatenate([v[:,:3],v[:,:3],v[:,3:4]],-1);mask=np.concatenate([v[:,:3],v[:,:3],np.repeat(v[:,3:4],3,axis=1)],-1);x=np.concatenate([y*mask,known],-1).astype(np.float32);phys=[]
   for info in infos:
    a,names=physical(info,rr[0]['environment']);phys.append(a)
    if not schema:schema=names
    assert schema==names
   item={'root_id':rr[0]['root_id'],'episode_id':eid,'initial_state':x,'physical':np.stack(phys),'y':y.astype(np.float32),'valid':mask,'cat':cat,'action':np.stack(actions),'status':np.array(statuses),'moved':np.array(events,np.float32),'moved_valid':np.array(evvalid,bool),'controls':np.array(actualcontrols),'public_frame':np.array(frames)};items.append(item)
   for t,r in enumerate(rr):records.append({'split':split,'root_id':r['root_id'],'episode_id':eid,'t':t,'skill':r['skill'],'before_relation':np.r_[y[t,:6],cat[t,3]].astype(int).tolist(),'before_full_category':cat[t].tolist(),'after_relation':np.r_[y[t+1,:6],cat[t+1,3]].astype(int).tolist(),'status':STATUS[statuses[t]],'before_physical':phys[t].tolist(),'after_moved':events[t]})
  assert len(items)==len(members[split])*3;out={k:np.array([a[k]for a in items])for k in items[0]};np.savez_compressed(S/f'data/dataset_{split}.npz',**out);counts.append({'split':split,'roots':len(members[split]),'episodes':len(items),'transitions':len(items)*4,'success':int((out['status']==0).sum()),'refused':int((out['status']==1).sum()),'executed_failure':int((out['status']==2).sum()),'invalid_relation_components':int((~out['valid'].astype(bool)).sum())})
  # Verify cached visual predictions against exact episode/label lineage; h2 test reuse.
  for seed in [0,1]:
   raw=dict(np.load(use(visual_path(seed,split))));lookup={(str(e),str(mode),int(h)):i for i,(e,mode,h)in enumerate(zip(raw['episode_id'],raw['mode'],raw['step']))};current=[];ones=[];free=[];success=[];motion=[]
   for a in items:
    eid=a['episode_id'];cc=[];oo=[];ff=[];ss=[];mm=[]
    for t in range(5):
     i=lookup[eid,'current',t];assert np.array_equal(raw['cat'][i,:4],a['cat'][t]);cc.append(rel_from_prob(raw['prob'][i]))
    for t in range(1,5):
     i=lookup[eid,'one_step',t];assert np.array_equal(raw['cat'][i,:4],a['cat'][t]);assert raw['controls'][i]==a['controls'][t-1];oo.append(rel_from_prob(raw['prob'][i]));ss.append(raw['event_prob'][i,0]);mm.append(raw['event_prob'][i,1:4]);key=(eid,'free'+str(t),t);ff.append(rel_from_prob(raw['prob'][lookup[key]]))if key in lookup else ff.append(np.full(9,np.nan))
    current.append(cc);ones.append(oo);free.append(ff);success.append(ss);motion.append(mm)
   np.savez_compressed(S/f'data/visual_seed{seed}_{split}.npz',current=np.array(current),one_step=np.array(ones),free=np.array(free),success=np.array(success),moved=np.array(motion),root_id=out['root_id'],episode_id=out['episode_id'])
 for s in SPLITS:
  for t in SPLITS:
   if s!=t:assert not set(members[s])&set(members[t])
 train=dataset('train');phys=train['physical'][:,:4].reshape(-1,len(schema));mu=phys.mean(0,dtype=np.float64).astype(np.float32);sd=np.maximum(phys.std(0,dtype=np.float64),.001).astype(np.float32);np.savez(S/'configs/physical_normalization.npz',mean=mu,std=sd);save(S/'configs/physical_schema.json',{'names':schema,'dimensions':len(schema),'model_state_dimensions':16+len(schema),'not_proven_complete':['robot joint velocities','controller internal state','collision solver cache'],'source':'actual pre-action endpoint info only; future states excluded from normalization'})
 rootidx={r:[i for i,e in enumerate(train['root_id'])if e==r]for r in sorted(set(train['root_id']))}
 for seed in [0,1]:
  rng=np.random.default_rng(2610191700+seed);schedule=np.array([[rng.choice(rootidx[r])for r in rng.choice(list(rootidx),32)]for _ in range(1500)],np.int64);np.save(S/f'configs/schedule_seed{seed}.npy',schedule)
 # Empirical ambiguity is a warning about state sufficiency, not a fitted dynamics model.
 aliases=[]
 for keyname in ['before_relation','before_full_category']:
  groups=collections.defaultdict(list)
  for r in records:
   if r['split']=='train':groups[(tuple(r[keyname]),r['skill'])].append(r)
  conflicts=[];minority=0
  for key,rr in groups.items():
   outputs=collections.Counter((tuple(x['after_relation']),x['status'])for x in rr);minority+=len(rr)-max(outputs.values())
   if len(outputs)>1:
    a=rr[0];bb=next(x for x in rr if (x['after_relation'],x['status'])!=(a['after_relation'],a['status']));conflicts.append({'input_state':key[0],'skill':key[1],'transitions':len(rr),'roots':len(set(x['root_id']for x in rr)),'outputs':[{'after_relation':k[0],'status':k[1],'count':n}for k,n in outputs.items()],'example_a':a,'example_b':bb,'physical_max_abs_difference':float(np.max(abs(np.array(a['before_physical'])-np.array(bb['before_physical']))))})
  aliases.append({'state_type':keyname,'training_groups':len(groups),'conflicting_groups':len(conflicts),'empirical_joint_target_minority':minority,'training_transitions':480,'conflicts':conflicts})
 save(S/'results/state_sufficiency.json',aliases);save(S/'data/provenance.json',provenance);save(S/'configs/roots.json',members);save(S/'results/data_counts.json',counts);save(S/'results/preflight.json',{'utc':now(),'passed':True,'seconds':time.monotonic()-start,'physical_dims':len(schema),'counts':counts,'cached_visual_aligned':True,'new_encoder_forwards':0,'new_collection':0,'C_D_samples_accessed':0})
 for p in list((S/'source').glob('*.py'))+list((S/'configs').glob('*'))+list((S/'data').glob('*')):
  if p.is_file():protected[str(p)]=sha(p)
 save(S/'configs/inputs_frozen.json',{'utc':now(),'before_new_fits':True,'files':len(protected),'sha256':protected});print('PREFLIGHT_COMPLETE',counts,'PHYS_DIM',len(schema),'ALIASES',[(a['state_type'],a['conflicting_groups'],a['empirical_joint_target_minority'])for a in aliases],flush=True)
if __name__=='__main__':main()
