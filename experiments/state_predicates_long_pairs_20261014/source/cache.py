from pathlib import Path
import json,hashlib,argparse
import numpy as np
from wm_symbolic.vision_cache import FrozenVision,VisibleRegistry
P=Path('/home/robot/research/projects/wm_symbolic');E=P/'experiments/state_predicates_long_pairs_20261014';B=P/'data/predicate_long_pairs_20261014';PRE=P/'experiments/state_predicates_held_down_20261014/candidate';COLORS=['red','blue','pink']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['pilot','full'],required=True);args=ap.parse_args();prot=read(E/'configs/protocol.json');manifest=read(B/'data_manifest.json');gate=read(E/'results'/('pilot_gate.json'if args.phase=='pilot'else'data_validation.json'));assert gate['passed']
 for p,h in prot['frozen_sha256'].items():assert sha(p)==h,p
 stats=read(PRE/'normalization.json');vm=np.array(stats['vision_mean'],np.float32);vs=np.array(stats['vision_std'],np.float32);rm=np.array(stats['robot_mean'],np.float32);rs=np.array(stats['robot_std'],np.float32)
 pubdir=E/'data/public';privdir=E/'data/_private';pubdir.mkdir(exist_ok=True);privdir.mkdir(exist_ok=True);(pubdir/'windows').mkdir(exist_ok=True);(privdir/'windows').mkdir(exist_ok=True);enc=FrozenVision(P);meta=[];targetrows=[];audits=[]
 for r in manifest['windows']:
  wid=r['window_id'];fp=E/'data/features'/(wid+'.npz');ip=pubdir/'windows'/(wid+'.npz');apath=privdir/'windows'/(wid+'.json');reg=VisibleRegistry()
  if ip.exists():
   old=read(apath);assert sha(ip)==old['input_sha256']and sha(fp)==old['raw_feature_sha256'];audits.append(old)
  else:
   assert not fp.exists()and not apath.exists();f=enc.encode(r['public_frames'],reg);np.savez_compressed(fp,**f);tok=((f['tokens'].astype(np.float32)-vm)/vs).reshape(4,2,64,384);geo=f['geometry'].reshape(4,2,64,8);v=f['valid'].reshape(4,2,64);ids=f['visible_registry_ids'].tolist();ob=np.zeros((4,3,2,390),np.float32)
   for j in range(3):
    uid=reg.colour_ids.get(j)
    if uid not in ids:continue
    slot=ids.index(uid);sl=slice(slot*4,slot*4+4);ob[:,j,:,:384]=tok[:,:,sl].mean(2)*v[:,:,slot*4,None];ob[:,j,:,384:389]=geo[:,:,slot*4,:5];ob[:,j,:,389]=geo[:,:,sl,5].mean(2)
   robot=[]
   for pp in r['public_frames']:
    with np.load(pp)as z:robot.append(z['robot_obs'])
   public={'objects':ob.reshape(4,3,780),'context':tok[:,:,48:].reshape(4,2,2,2,2,2,384).mean((3,5)).reshape(4,-1),'robot':(np.stack(robot).astype(np.float32)-rm)/rs}
   for k,shape in prot['features']['shapes'].items():assert list(public[k].shape)==shape and public[k].dtype==np.float32 and np.isfinite(public[k]).all()
   np.savez_compressed(ip,**public)
   # Private truth is opened ONLY after public feature tensors have been saved.
   private=read(B/'episodes'/r['root_id']/'_private'/(wid+'.json'));last=private['frames'][-1];identity={c:{'registered_public_uid':reg.colour_ids.get(j),'private_uid':last['objects_audit'][c]['object_uid_evaluator_only'],'match':reg.colour_ids.get(j)==last['objects_audit'][c]['object_uid_evaluator_only'],'missing':reg.colour_ids.get(j)not in ids}for j,c in enumerate(COLORS)}
   audit={'window_id':wid,'input_sha256':sha(ip),'raw_feature_sha256':sha(fp),'registry_audit':identity,'all7_targets':private['all_7_targets'],'truth_does_not_enter_features':True};save(apath,audit);audits.append(audit)
  meta.append(r|{'input_path':str(ip),'input_sha256':audits[-1]['input_sha256'],'raw_feature_path':str(fp),'raw_feature_sha256':audits[-1]['raw_feature_sha256']});targetrows.append(audits[-1]['all7_targets'])
  if len(meta)%10==0:print('CACHED',len(meta),flush=True)
 widx={r['window_id']:i for i,r in enumerate(meta)};inputs={k:[]for k in ['objects','context','robot']}
 for r in meta:
  with np.load(r['input_path'])as z:
   assert set(z.files)==set(inputs)
   for k in inputs:inputs[k].append(z[k])
 inputs={k:np.stack(v)for k,v in inputs.items()};inputs['pair_before_index']=np.array([widx[r['before_window_id']]for r in manifest['records']],np.int64);inputs['pair_after_index']=np.array([widx[r['after_window_id']]for r in manifest['records']],np.int64);actions=[]
 for r in manifest['records']:
  v=np.zeros(9,np.float32);kind=0 if r['skill']=='open_drawer'else 1 if r['skill'].startswith('lift_')else 2 if r['skill']=='place_in_drawer'else 3;v[kind]=1
  if kind==1:v[4+COLORS.index(r['declared_arguments']['color'])]=1
  v[7:]=1;actions.append(v)
 inputs['declared_action_features']=np.stack(actions);targetdata={'window_y':np.array([t['y']for t in targetrows],np.float32),'window_valid':np.array([t['valid']for t in targetrows],bool)}
 for key in ['before','after']:
  targetdata[key+'_y']=targetdata['window_y'][inputs['pair_'+key+'_index']];targetdata[key+'_valid']=targetdata['window_valid'][inputs['pair_'+key+'_index']]
 np.savez_compressed(pubdir/('paired_inputs_'+args.phase+'.npz'),**inputs);np.savez_compressed(privdir/('supervision_'+args.phase+'.npz'),**targetdata);save(pubdir/('windows_'+args.phase+'.json'),meta);save(pubdir/('pairs_'+args.phase+'.json'),manifest['records'])
 result={'passed':True,'phase':args.phase,'unique_windows':len(meta),'macro_pairs':len(manifest['records']),'public_only_encoding':True,'candidate_reader_forwards':0,'new_training_updates':0,'source_manifest_sha256':sha(B/'data_manifest.json'),'paired_inputs_sha256':sha(pubdir/('paired_inputs_'+args.phase+'.npz')),'supervision_sha256':sha(privdir/('supervision_'+args.phase+'.npz')),'identity_matches':sum(x['match']for a in audits for x in a['registry_audit'].values()),'identity_missing':sum(x['missing']for a in audits for x in a['registry_audit'].values()),'identity_mismatch_present':sum(not x['match']and not x['missing']for a in audits for x in a['registry_audit'].values()),'shapes':{k:list(v.shape)for k,v in inputs.items()},'action_feature_order':['open','lift','place','close','explicit_lift_red','explicit_lift_blue','explicit_lift_pink','declared_budget_div900','declared_posthold_div96'],'forbidden_predictor_inputs':['private_truth','duration','primitive_success','phase','contacts','actual_future_control_commands'],'audits':audits};save(E/'results'/('cache_'+args.phase+'.json'),result);print(json.dumps({k:v for k,v in result.items()if k!='audits'},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
