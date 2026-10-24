"""Independent read-only source, physical record, label, split, training and scoring checks."""
from config import *
import sys,gzip,csv,ast,subprocess,time
import imageio.v2 as imageio
sys.path.insert(0,str(REP/'source'))
from labels_v2_1 import evaluate
def main():
 start=time.monotonic();hashes=0
 for fn in ['collection_frozen.json','query_training_frozen.json','query_choices_frozen.json','transition_training_frozen.json','before_D3_frozen.json','raw_ABC_hashes.json','raw_D3_hashes.json']:
  for p,h in read(E/'configs'/fn)['sha256'].items():assert sha(p)==h,p;hashes+=1
 roots=read(E/'configs/roots.json');assert len(roots)==len({r['root_id']for r in roots})==len({r['seed']for r in roots})==36;rootmap={r['root_id']:r for r in roots};before=read(E/'configs/before_D3_frozen.json');freezetime=datetime.datetime.fromisoformat(before['frozen_utc']).timestamp();labels_checked=0;trace_frames=0;callcount=0;video_records=[];content={};beforeobs={};status={};fullstates=0
 for phase in ['ABC','D3']:
  eps=read(E/'results'/f'collection_{phase}.json');assert len(eps)==(48 if phase=='ABC' else 24)
  for n,ep in enumerate(eps):
   root=rootmap[ep['root_id']];assert ep['split']==root['split'];folder=Path(ep['folder']);points=read(folder/'_private/states.json');pointmap={r['frame']:r for r in points};calls=read(folder/'_private/calls.json');assert len(calls)==ep['actual_calls'];assert ep['error']is None
   if phase=='D3':assert (folder/'_private/provenance.json').stat().st_mtime>=freezetime
   for p in points:
    lab=evaluate(p['physical']);assert lab==p['labels'];labels_checked+=1;fullstates+=all(lab['v']);obs=np.load(p['public_path']);assert set(obs.files)=={'rgb_static','rgb_gripper','robot_obs'}and obs['robot_obs'].shape==(15,);h=hashlib.sha256(b''.join(obs[k].tobytes()for k in sorted(obs.files))).hexdigest();content.setdefault(h,set()).add(root['root_id']);obs.close()
   for j,c in enumerate(calls):
    assert c['slot']==j and c['skill']==root['branches'][ep['branch']][j]and c['action']==action(c['skill']);assert c['start_frame']==c['before_frame']and c['end_frame']==c['after_frame'];assert c['primitive_end']-c['start_frame']==c['native']['controls']<=900;assert c['end_frame']-c['primitive_end']==96;assert c['native']['attempts']<=2
    if c['status']=='refused':assert c['native']['controls']==0
    if j:assert calls[j-1]['after_frame']==c['before_frame']
    status[c['status']]=status.get(c['status'],0)+1;callcount+=1
   p0=pointmap[calls[0]['before_frame']]['public_path'];z=np.load(p0);b={k:z[k]for k in z.files};z.close()
   if root['root_id']in beforeobs:assert all(np.array_equal(b[k],beforeobs[root['root_id']][k])for k in b)
   else:beforeobs[root['root_id']]=b
   count=0
   with gzip.open(folder/'_private/physical_control_trace.jsonl.gz','rt')as f:
    for line in f:
     r=json.loads(line);assert r['frame']==count and r['ticks']==count*8
     if count in pointmap:
      p=pointmap[count];assert p['physical']==r['physical'];obs=np.load(p['public_path']);assert np.array_equal(obs['robot_obs'],r['robot_obs']);obs.close()
     count+=1
   assert count==ep['frames'];trace_frames+=count;reader=imageio.get_reader(folder/'continuous.mp4');meta=reader.get_meta_data();frames=reader.count_frames();reader.close();assert frames==count and meta['fps']==30;video_records.append({'path':str(folder/'continuous.mp4'),'frames':frames,'fps':30})
   if(n+1)%12==0:print('VERIFIED_PHYSICAL',phase,n+1,flush=True)
  # Recount every condition, horizon, group and actual/predicted paired fact independently.
  key='C_validation'if phase=='ABC'else'D3';scores=read(E/'results'/f'scored_{key}.json');metrics=read(E/'results'/f'metrics_{key}.json');private=read(E/'data'/phase/'_private/labels.json');pairs={r['id']:r for r in read(E/'data'/phase/'public/pairs.json')};sourcepred=read(E/'results'/f'predictions_{key}.json')
  for cond,rr in scores.items():
   for r,raw in zip(rr,sourcepred[cond]):
    assert r['id']==raw['id'];assert r['truth']==private[str(pairs[r['id']]['after'])]['y'];assert r['valid']==private[str(pairs[r['id']]['after'])]['v']
   for branch in ['normal','held_closed']:
    for h in range(4):
     use=[r for r in rr if r['branch']==branch and r['slot']==h];m=metrics['trajectory_prediction'][cond]['branches'][branch]['all_available'][h];assert len(use)==m['observed_roots'];y=np.array([r['truth']for r in use]);v=np.array([r['valid']for r in use]);pp={k:discrete(np.array([r[k+'_probability']for r in use]))for k in ['actual','one_step','recursive']};full=v.all(1)
     for method,p in pp.items():
      ok=(p==y).all(1);assert int((ok&full).sum())==m[method]['full9_correct'];assert int(full.sum())==m[method]['full9_N']
      for g,lo,hi in [('Holding',0,3),('In',3,6),('drawer',6,9)]:assert int(((p[:,lo:hi]!=y[:,lo:hi]).any(1)&v[:,lo:hi].all(1)).sum())==m[method]['groups'][g]['wrong_states']
     for method in ['one_step','recursive']:
      for lev,aa,bb,valid in [('facts',pp['actual']==y,pp[method]==y,v),('states',(pp['actual']==y).all(1),(pp[method]==y).all(1),full)]:
       cc={'actual_right_prediction_wrong':int((aa&~bb&valid).sum()),'both_wrong':int((~aa&~bb&valid).sum()),'actual_wrong_prediction_right':int((~aa&bb&valid).sum()),'both_right':int((aa&bb&valid).sum())};assert cc==m[method+'_paired'][lev]
  sp=read(E/'results'/f'true_state_probabilities_{key}.json');yy=np.array([private[str(i)]['y']for i in sp['state_indices']]);vv=np.array([private[str(i)]['v']for i in sp['state_indices']]);full=vv.all(1)
  for name,probs in sp['probabilities'].items():assert int(((discrete(probs)==yy).all(1)&full).sum())==metrics['true_state_readout'][name]['all_states']['full9_correct']
  assert len(list(csv.DictReader((R/f'paired_propositions_{key}.csv').open())))==sum(len(r)*9 for r in scores.values())
 assert callcount==288 and len(video_records)==72 and fullstates==labels_checked==1649;assert all(len(rs)==1 for rs in content.values()),'Exact observation shared across distinct roots'
 # Explicit query/transition data exclusions and same-initialization budget checks.
 newstates=read(E/'data/ABC/public/states.json');selected=read(E/'configs/query_choices_frozen.json');qcount=tcount=0
 for seed in [0,1]:
  init=[];tinit=[]
  for arm in ['old_coverage','new_coverage']:
   qlog=read(E/'results'/f'train_query_{arm}_seed{seed}.json');assert qlog['updates']==1200 and qlog['draws']==38400 and qlog['pool_size']==144;init.append(qlog['initial_hash']);qcount+=qlog['updates'];tlog=read(E/'results'/f'train_transition_{arm}_seed{seed}.json');assert tlog['updates']==1000 and tlog['sample_draws']==64000;assert tlog['frozen_encoder_decoder_hash']==tlog['final_frozen_hash']and tlog['query_hash']==tlog['final_query_hash'];tinit.append(tlog['initial_model_hash']);tcount+=tlog['updates']
   for typ,log in [('query',qlog),('transition',tlog)]:
    chosen=read(E/'models'/f'{typ}_{arm}_seed{seed}/selected.json');best=min(log['log'],key=lambda x:x['score']);assert chosen['path'].endswith(f'update{best["update"]:04d}.pt')
   pack=np.load(E/'data/query_train'/f'{arm}.npz');assert all(r.startswith(('A_','B_'))for r in pack['root_ids']);assert len(pack['z'])==144
  assert len(set(init))==len(set(tinit))==1
 newcs=set(np.load(E/'data/query_train/C_selection.npz')['root_ids']);newcv=set(np.load(E/'data/query_evaluator/C_validation.npz')['root_ids']);assert not(newcs&newcv);trrows=read(E/'data/transition_train/states.json');assert all(r['environment']!='D'for r in trrows if'environment'in r);assert not any(r['root_id']in newcv for r in trrows)
 for f in(E/'source').glob('*.py'):ast.parse(f.read_text())
 old=subprocess.run([sys.executable,str(PREV/'source/verify.py')],capture_output=True,text=True,check=True);oldresult=json.loads(old.stdout);assert oldresult['passed']
 result={'passed':True,'frozen_hash_entries_verified':hashes,'roots':36,'physical_skill_calls':callcount,'status_counts':status,'sampled_states_and_full9_recomputed':labels_checked,'trace_frames_checked':trace_frames,'continuous_videos_checked':len(video_records),'paired_branch_initial_observations_identical':True,'exact_observation_overlap_between_distinct_roots':0,'query_updates':qcount,'transition_updates':tcount,'frozen_encoder_decoder_query_unchanged':True,'models_sealed_before_D_collection':True,'independent_horizon_and_paired_recount':True,'D3_now_seen':True,'previous80file_and_all_older_seals_passed':oldresult['new_sealed_files_verified']==80,'verification_seconds':time.monotonic()-start}
 if(E/'configs/delivery_manifest.json').exists():
  files=read(E/'configs/delivery_manifest.json')['sha256']
  for p,h in files.items():assert sha(p)==h,p
  result['new_sealed_files_verified']=len(files)
 save(E/'results/physical_media_verification.json',{'videos':video_records})if not(E/'configs/delivery_manifest.json').exists()else None
 print('VERIFICATION_RESULT',json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
