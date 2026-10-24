from pathlib import Path
import json,hashlib,datetime
import numpy as np
from wm_symbolic.vision_cache import FrozenVision,VisibleRegistry,WEIGHTS
P=Path('/home/robot/research/projects/wm_symbolic');E=P/'experiments/state_predicates_causal_windows_20261014';B=P/'data/predicate_causal_replay_20261014';O=P/'experiments/state_predicates_20261014';C=['red','blue','pink']
def read(p):return json.loads(Path(p).read_text())
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    assert not(E/'data/causal_train_dev.npz').exists();manifest=read(B/'data_manifest.json');assert len(manifest['trajectories'])==18
    stats=read(O/'data/normalization.json');vm=np.array(stats['vision_mean'],np.float32);vs=np.array(stats['vision_std'],np.float32);rm=np.array(stats['robot_mean'],np.float32);rs=np.array(stats['robot_std'],np.float32)
    enc=FrozenVision(P);old=read(O/'data/metadata.json')[0];probe=enc.encode(old['public_inputs'],VisibleRegistry())
    with np.load(old['feature_path'])as z:parity={k:float(np.max(np.abs(probe[k].astype(float)-z[k].astype(float))))for k in ['tokens','geometry','robot']}
    assert parity['tokens']<.02 and parity['geometry']==0 and parity['robot']==0
    save(E/'results/feature_parity.json',{'passed':True,'max_errors':parity,'backbone':str(WEIGHTS),'backbone_sha256':sha(WEIGHTS/'model.safetensors')})
    arrays={k:[]for k in ['objects','context','robot','y','v']};meta=[];reg=None;last=None;identity=[];hashes={}
    for r in sorted(manifest['records'],key=lambda x:(x['trajectory_id'],x['anchor_frame'])):
        if last!=r['trajectory_id']:reg=VisibleRegistry();last=r['trajectory_id']
        f=enc.encode(r['public_frames'],reg);fp=E/'data/features'/(r['sample_id']+'.npz');np.savez_compressed(fp,**f);tok=((f['tokens'].astype(np.float32)-vm)/vs).reshape(4,2,64,384);geo=f['geometry'].reshape(4,2,64,8);valid=f['valid'].reshape(4,2,64);ids=f['visible_registry_ids'].tolist();ob=np.zeros((4,3,2,390),np.float32)
        for j in range(3):
            uid=reg.colour_ids.get(j);slot=ids.index(uid)if uid in ids else None
            if slot is None:continue
            sl=slice(slot*4,slot*4+4);vv=valid[:,:,slot*4];ob[:,j,:,:384]=tok[:,:,sl].mean(2)*vv[:,:,None];ob[:,j,:,384:389]=geo[:,:,slot*4,:5];ob[:,j,:,389]=geo[:,:,sl,5].mean(2)
        grid=tok[:,:,48:].reshape(4,2,4,4,384).reshape(4,2,2,2,2,2,384).mean((3,5)).reshape(4,-1);rgb=[];rob=[]
        for pp in r['public_frames']:
            with np.load(pp)as z:rob.append(z['robot_obs']);rgb.extend([z['rgb_static'].tobytes(),z['rgb_gripper'].tobytes()])
        # Labels and numeric IDs are read only after the public feature call.
        lab=read(r['labels_private'])['frames'][-1];cat=np.array(lab['labels']['categorical'][:4]);j=C.index(r['target_color']);y=np.r_[cat[:3]==6,cat[:3]==3,cat[3]==2].astype(np.float32);v=np.zeros(7,bool);v[[j,j+3,6]]=np.array(lab['labels']['categorical_valid'])[[j,j,3]]
        assert v[[j,j+3,6]].all()
        for k,c in enumerate(C):
            registered=reg.colour_ids.get(k);actual=lab['objects_audit'][c]['object_uid_evaluator_only'];identity.append({'sample_id':r['sample_id'],'colour':c,'match':registered==actual});assert registered==actual
        arrays['objects'].append(ob.reshape(4,3,780));arrays['context'].append(grid);arrays['robot'].append((np.stack(rob).astype(np.float32)-rm)/rs);arrays['y'].append(y);arrays['v'].append(v)
        h=hashlib.sha256(b''.join(rgb)).hexdigest();hashes.setdefault(h,set()).add(r['split']);meta.append(r|{'target_index':j,'category':cat.tolist(),'feature_path':str(fp),'feature_sha256':sha(fp),'public_window_rgb_sha256':h,'current_held':bool(y[j])})
        if len(meta)%12==0:print('CACHED',len(meta),flush=True)
    assert all(len(v)==1 for v in hashes.values());np.savez_compressed(E/'data/causal_train_dev.npz',**{k:np.stack(v)for k,v in arrays.items()},splits=np.array([r['split']for r in meta]),roles=np.array([r['dev_role']or'train'for r in meta]));save(E/'data/metadata.json',meta);save(E/'results/cache_audit.json',{'windows':len(meta),'identity_checks':identity,'exact_RGB_cross_split_duplicates':[],'private_labels_after_public_encoder_only':True,'normalization_unchanged':True,'new_roots_cross_split':False});print('CAUSAL_CACHE_COMPLETE',len(meta))
if __name__=='__main__':main()
