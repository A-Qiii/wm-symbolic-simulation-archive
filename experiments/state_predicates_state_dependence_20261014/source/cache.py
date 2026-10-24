from experiment import *
import sys,argparse,importlib.util
import numpy as np
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['pilot','full'],required=True);a=ap.parse_args();verify_frozen();assert read(E/'results'/('pilot_gate.json'if a.phase=='pilot'else'data_validation.json'))['passed'];m=read(B/'data_manifest.json')
 sys.path.insert(0,str(RGB/'source'));spec=importlib.util.spec_from_file_location('frozen_rgb_encoder',RGB/'source/cache.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);enc=mod.RGBEncoder();stats=read(CAND/'normalization.json');vm=np.array(stats['vision_mean'],np.float32);vs=np.array(stats['vision_std'],np.float32);rm=np.array(stats['robot_mean'],np.float32);rs=np.array(stats['robot_std'],np.float32);arrays={k:[]for k in ['objects','context','robot']};meta=[];targets=[];audits=[]
 for i,r in enumerate(m['windows']):
  fp=E/'data/features'/(r['window_id']+'.npz');auditpath=E/'data/_private'/(r['window_id']+'_cache.json')
  if fp.exists():
   au=read(auditpath);assert sha(fp)==au['raw_features_sha256'];f=dict(np.load(fp))
  else:
   f,masks=enc.encode(r['public_frames']);np.savez_compressed(fp,**f);au={'raw_features_sha256':sha(fp),'RGB_input_sha256':[sha(p)for p in r['public_frames']],'registry_sha256':sha(RGB/'source/registry.py')};save(auditpath,au)
  tok=(f['local_tokens'].astype(np.float32)-vm)/vs;g=f['geometry'];vis=g[:,:,:,0,4].astype(bool);ob=np.zeros((4,3,2,390),np.float32)
  for j in range(3):
   if not vis[:,:,j].any():continue
   ob[:,j,:,:384]=tok[:,:,j].mean(2)*vis[:,:,j,None];ob[:,j,:,384:389]=g[:,:,j,0,:5];ob[:,j,:,389]=g[:,:,j,:,5].mean(2)
  grid=(f['global_grid'].astype(np.float32)-vm)/vs;cur={'objects':ob.reshape(4,3,780),'context':grid.reshape(4,2,2,2,2,2,384).mean((3,5)).reshape(4,3072),'robot':(f['robot_history']-rm)/rs}
  for k,v in cur.items():assert np.isfinite(v).all()and v.dtype==np.float32;arrays[k].append(v)
  meta.append(r|{'raw_features_path':str(fp),'raw_features_sha256':sha(fp),'missing_entire_window':(~vis.any((0,1))).tolist(),'visible_per_frame_colour_camera':vis.transpose(0,2,1).tolist()});audits.append(au)
  if(i+1)%12==0:print('CACHED_REAL_WINDOWS',i+1,flush=True)
 arrays={k:np.stack(v)for k,v in arrays.items()};wi={r['window_id']:i for i,r in enumerate(meta)}
 for key in ['before','after']:arrays['pair_'+key+'_index']=np.array([wi[r[key+'_window_id']]for r in m['records']],np.int64)
 actions=[]
 for r in m['records']:
  v=np.zeros(9,np.float32);kind=0 if r['skill']=='open_drawer'else 1 if r['skill'].startswith('lift_')else 2 if r['skill']=='place_in_drawer'else 3;v[kind]=1
  if kind==1:v[4+COLORS.index(r['declared_arguments']['color'])]=1
  v[7]=r['declared_primitive_budget']/900;v[8]=r['declared_posthold_controls']/96;actions.append(v)
 arrays['declared_action_features']=np.stack(actions);np.savez_compressed(E/'data/public'/('inputs_'+a.phase+'.npz'),**arrays);save(E/'data/public'/('windows_'+a.phase+'.json'),meta);save(E/'data/public'/('pairs_'+a.phase+'.json'),m['records'])
 # Only after public feature tensors exist, load the independent labels.
 for r in meta:targets.append(read(B/'episodes'/r['episode_id']/'_private'/(r['window_id']+'.json'))['all_7_targets'])
 t={'window_y':np.array([x['y']for x in targets],np.float32),'window_valid':np.array([x['valid']for x in targets],bool)}
 for key in ['before','after']:
  ix=arrays['pair_'+key+'_index'];t[key+'_y']=t['window_y'][ix];t[key+'_valid']=t['window_valid'][ix]
 np.savez_compressed(E/'data/_private'/('supervision_'+a.phase+'.npz'),**t);result={'created_utc':now(),'passed':True,'phase':a.phase,'windows':len(meta),'pairs':len(m['records']),'inputs_sha256':sha(E/'data/public'/('inputs_'+a.phase+'.npz')),'supervision_sha256':sha(E/'data/_private'/('supervision_'+a.phase+'.npz')),'RGB_registry_unchanged':True,'candidate_reader_or_transition_forward_count':0,'training_updates':0,'shapes':{k:list(v.shape)for k,v in arrays.items()}};save(E/'results'/('cache_'+a.phase+'.json'),result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
