from config import *
import sys,gzip,csv,ast,time
import imageio.v2 as imageio
sys.path.insert(0,str(REP/'source'))
from labels_v2_1 import evaluate
def main():
 started=time.monotonic();hashes=0
 for fn in ['collection_frozen.json','training_frozen.json','before_D4_frozen.json','raw_D4_hashes.json']:
  for p,h in read(E/'configs'/fn)['sha256'].items():assert sha(p)==h,p;hashes+=1
 oldmanifest=read(PRIOR/'configs/delivery_manifest.json')
 for p,h in oldmanifest['sha256'].items():assert sha(p)==h,p
 roots=read(E/'configs/roots.json');assert len(roots)==len({r['root_id']for r in roots})==len({r['seed']for r in roots})==12;before=read(E/'configs/before_D4_frozen.json');stamp=datetime.datetime.fromisoformat(before['frozen_utc']).timestamp();episodes=read(E/'results/collection_D4.json');assert len(episodes)==24;pointsN=0;framesN=0;callsN=0;status={};contents={};initial={};full9=0;media=[]
 for ep in episodes:
  folder=Path(ep['folder']);assert (folder/'_private/provenance.json').stat().st_mtime>=stamp;points=read(folder/'_private/states.json');byframe={p['frame']:p for p in points};calls=read(folder/'_private/calls.json');assert ep['error']is None
  for p in points:
   assert evaluate(p['physical'])==p['labels'];pointsN+=1;full9+=all(p['labels']['v']);z=np.load(p['public_path']);h=hashlib.sha256(b''.join(z[k].tobytes()for k in sorted(z.files))).hexdigest();contents.setdefault(h,set()).add(ep['root_id']);z.close()
  for j,c in enumerate(calls):
   assert c['slot']==j and c['action']==action(c['skill']);assert c['primitive_end']-c['start_frame']==c['native']['controls']<=900;assert c['end_frame']-c['primitive_end']==96;assert c['native']['attempts']<=2
   if j:assert calls[j-1]['after_frame']==c['before_frame']
   status[c['status']]=status.get(c['status'],0)+1;callsN+=1
  with np.load(byframe[calls[0]['before_frame']]['public_path'])as z:obs={k:z[k]for k in z.files}
  if ep['root_id']in initial:assert all(np.array_equal(obs[k],initial[ep['root_id']][k])for k in obs)
  else:initial[ep['root_id']]=obs
  count=0
  with gzip.open(folder/'_private/physical_control_trace.jsonl.gz','rt')as f:
   for line in f:
    r=json.loads(line);assert r['frame']==count and r['ticks']==8*count
    if count in byframe:
     p=byframe[count];assert p['physical']==r['physical']
     with np.load(p['public_path'])as z:assert np.array_equal(z['robot_obs'],r['robot_obs'])
    count+=1
  assert count==ep['frames'];framesN+=count;reader=imageio.get_reader(folder/'continuous.mp4');meta=reader.get_meta_data();n=reader.count_frames();reader.close();assert n==count and meta['fps']==30;media.append({'path':str(folder/'continuous.mp4'),'frames':n,'fps':30})
 assert callsN<=96 and all(len(x)==1 for x in contents.values());newdata=read(E/'data/D4/public/states.json');label=read(E/'data/D4/_private/labels.json')
 for seed in[0,1]:
  sc=np.load(E/'configs'/f'schedule_seed{seed}.npz');a=sc['control'];b=sc['empty_open'];assert a.shape==b.shape==(600,32);assert np.array_equal(a[:,:17],b[:,:17])and np.array_equal(a[:,20:],b[:,20:]);assert a.max()<144 and b[:,17:20].min()>=144 and b.max()<156;inits=[]
  for arm in ['control','empty_open']:
   t=read(E/'results'/f'train_{arm}_seed{seed}.json');assert t['updates']==600 and t['draws']==19200 and not t['C_validation_or_D_access'];inits.append(t['initial_hash']);s=read(E/'models'/f'{arm}_seed{seed}/selected.json');best=min(t['log'],key=lambda x:x['score']);assert s['path'].endswith(f'update{best["update"]:04d}.pt')
  assert len(set(inits))==1
 for arm,N in [('control',144),('empty_open',156)]:
  data=np.load(E/'data/query_train'/f'{arm}.npz');assert len(data['z'])==N and all(r.startswith(('A_','B_'))for r in data['root_ids'])
 for phase in ['C_backtest','D4']:
  metrics=read(E/'results'/f'metrics_{phase}.json');scores=read(E/'results'/f'scored_{phase}.json')
  for name,rows in scores.items():
   for branch in ['normal','held_closed']:
    for h in range(4):
     rr=[r for r in rows if r['branch']==branch and r['slot']==h];m=metrics['trajectory_prediction'][name][branch][h];assert len(rr)==m['observed_roots'];y=np.array([r['truth']for r in rr]);v=np.array([r['valid']for r in rr]);full=v.all(1);pred={k:discrete(np.array([r[k+'_probability']for r in rr]))for k in ['actual','one_step','recursive']}
     for k,p in pred.items():assert int(((p==y).all(1)&full).sum())==m[k]['full9_correct']
     for k in ['one_step','recursive']:
      for level,aa,bb,mask in [('facts',pred['actual']==y,pred[k]==y,v),('states',(pred['actual']==y).all(1),(pred[k]==y).all(1),full)]:
       pair={'actual_right_prediction_wrong':int((aa&~bb&mask).sum()),'both_wrong':int((~aa&~bb&mask).sum()),'actual_wrong_prediction_right':int((~aa&bb&mask).sum()),'both_right':int((aa&bb&mask).sum())};assert pair==m[k+'_paired'][level]
  assert len(list(csv.DictReader((R/f'paired_propositions_{phase}.csv').open())))==sum(len(r)*9 for r in scores.values())
 for p in(E/'source').glob('*.py'):ast.parse(p.read_text())
 result={'passed':True,'frozen_hash_entries_verified':hashes,'prior_study_sealed_files_verified':len(oldmanifest['sha256']),'roots':12,'episodes':24,'physical_calls':callsN,'status_counts':status,'sampled_states_checked':pointsN,'full9_states':full9,'trace_frames_checked':framesN,'videos_checked':24,'paired_reset_initial_observations_identical':True,'exact_observation_overlap_across_new_roots':0,'query_updates':2400,'training_draws':76800,'same29of32_draws_per_update':True,'only12AB_states_added':True,'C_and_D_excluded_from_training':True,'all_models_frozen_before_D4_collection':True,'D4_now_seen':True,'independent_pair_recount':True,'encoder_transition_unchanged':True,'seconds':time.monotonic()-started}
 if(E/'configs/delivery_manifest.json').exists():
  files=read(E/'configs/delivery_manifest.json')['sha256']
  for p,h in files.items():assert sha(p)==h,p
  result['sealed_files_verified']=len(files)
 else:save(E/'results/media_verification.json',media)
 print('VERIFICATION_RESULT',json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
