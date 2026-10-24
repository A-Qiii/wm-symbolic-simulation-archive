from pathlib import Path
import json,hashlib,argparse,time
import numpy as np
from wm_symbolic.vision_cache import FrozenVision,VisibleRegistry,WEIGHTS
P=Path('/home/robot/research/projects/wm_symbolic');OLD=P/'experiments/state_predicates_20261014';B=P/'data/predicate_joint_acceptance_20261014';E=P/'experiments/state_predicates_joint_20261014';C=['red','blue','pink']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2))
def main():
 pa=argparse.ArgumentParser();pa.add_argument('--splits',nargs='+',required=True);args=pa.parse_args()
 assert args.splits==['acceptance'] and (E/'results/candidate_freeze.json').exists(),'Candidate freeze required before acceptance feature extraction'
 protocol=json.loads((E/'configs/protocol.json').read_text());stats=json.loads((OLD/'data/normalization.json').read_text());vm=np.array(stats['vision_mean'],np.float32);vs=np.array(stats['vision_std'],np.float32);rm=np.array(stats['robot_mean'],np.float32);rs=np.array(stats['robot_std'],np.float32)
 enc=FrozenVision(P);manifest=json.loads((B/'data_manifest.json').read_text());rows=[r for r in manifest['records'] if r['split'] in args.splits];rows.sort(key=lambda r:(r['trajectory_id'],r['anchor_frame']));reg=None;previous=None;objects=[];contexts=[];robots=[];y=[];trainmask=[];evalmask=[];metadata=[];start=time.monotonic()
 if 'train' in args.splits:
  oldm=json.loads((OLD/'data/metadata.json').read_text())[0];probe=enc.encode(oldm['public_inputs'],VisibleRegistry())
  with np.load(oldm['feature_path']) as z:errors={k:float(np.max(np.abs(probe[k].astype(float)-z[k].astype(float)))) for k in ['tokens','geometry','robot']}
  assert errors['tokens']<.02 and errors['geometry']==0 and errors['robot']==0
  save(E/'results/frozen_feature_parity.json',{'passed':True,'same_old4frames_reencoded':oldm['public_inputs'],'max_absolute_errors':errors})
 for r in rows:
  if previous!=r['trajectory_id']:reg=VisibleRegistry();previous=r['trajectory_id']
  f=enc.encode(r['public_frames'],reg);dest=E/'data/features'/(r['sample_id']+'.npz');assert not dest.exists();np.savez_compressed(dest,**f)
  tok=((f['tokens'].astype(np.float32)-vm)/vs).reshape(4,2,64,384);geo=f['geometry'].reshape(4,2,64,8);valid=f['valid'].reshape(4,2,64);ids=f['visible_registry_ids'].tolist();ob=np.zeros((4,3,2,390),np.float32)
  for j in range(3):
   uid=reg.colour_ids.get(j);slot=ids.index(uid) if uid in ids else None
   if slot is None:continue
   sl=slice(slot*4,slot*4+4);v=valid[:,:,slot*4];ob[:,j,:,:384]=tok[:,:,sl,:].mean(2)*v[:,:,None];ob[:,j,:,384:389]=geo[:,:,slot*4,:5];ob[:,j,:,389]=geo[:,:,sl,5].mean(2)
  grid=tok[:,:,48:,:].reshape(4,2,4,4,384).reshape(4,2,2,2,2,2,384).mean((3,5)).reshape(4,-1)
  rb=[]
  for pp in r['public_frames']:
   with np.load(pp) as z:rb.append(z['robot_obs'])
  lab=json.loads(Path(r['labels_private']).read_text())['frames'][-1]['labels'];cat=np.array(lab['categorical'][:4]);yy=np.r_[cat[:3]==6,cat[:3]==3,cat[3]==2].astype(np.float32);j=C.index(r['target_color']);tm=np.zeros(7,bool);em=tm.copy();tm[[j,6]]=True;em[[j,j+3,6]]=True;tm[[j,6]]&=np.array(lab['categorical_valid'])[[j,3]];em[[j,j+3,6]]&=np.array(lab['categorical_valid'])[[j,j,3]]
  objects.append(ob.reshape(4,3,-1));contexts.append(grid);robots.append((np.stack(rb).astype(np.float32)-rm)/rs);y.append(yy);trainmask.append(tm);evalmask.append(em);metadata.append({**r,'feature_path':str(dest),'feature_sha256':sha(dest),'target_index':j,'registered_color_count':len(reg.colour_ids),'model_inputs_only':['objects','context','robot'],'category':cat.tolist()})
  if len(metadata)%20==0:print('CACHED',len(metadata),args.splits,flush=True)
 suffix='_'.join(args.splits);np.savez_compressed(E/'data'/f'supplement_{suffix}.npz',objects=np.stack(objects),context=np.stack(contexts),robot=np.stack(robots),y=np.stack(y),train_valid=np.stack(trainmask),eval_valid=np.stack(evalmask),splits=np.array([r['split'] for r in rows]),target=np.array([r['target_index'] for r in metadata]))
 save(E/'data'/f'metadata_{suffix}.json',metadata);save(E/'results'/f'cache_{suffix}.json',{'complete':True,'windows':len(metadata),'splits':args.splits,'seconds':time.monotonic()-start,'backbone_weights':str(WEIGHTS),'frozen':True,'normalization_unchanged':True,'unregistered_entities':sum(r['registered_color_count']<3 for r in metadata),'no_future_or_GT_to_encoder':True});print('CACHE_COMPLETE',len(metadata),flush=True)
if __name__=='__main__':main()
