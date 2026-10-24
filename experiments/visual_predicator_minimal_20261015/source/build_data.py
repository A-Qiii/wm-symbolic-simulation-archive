from common import *
import sys,importlib.util
def main():
 assert not(E/'data/public/reader_inputs.npz').exists();q=read(E/'configs/protocol.json')
 for p,h in q['fixed_sha256'].items():assert sha(p)==h,p
 mm=read(E/'configs/windows_before_encoding.json');groups={g['name']:g for g in read(E/'configs/reused_groups.json')};inputs={k:[]for k in ['objects','context','robot']};yy=[];vv=[];out=[];enc=None
 for i,r in enumerate(mm):
  if r['source']in groups:
   g=groups[r['source']]
   if 'cache'not in g:g['cache']=dict(np.load(g['inputs']));g['labels']=dict(np.load(g['supervision']))
   cur={k:g['cache'][k][r['source_index']]for k in inputs};lab={k:g['labels'][k][r['source_index']]for k in ['window_y','window_valid']}
   # Sanity-check reuse against public raw RGB features and unchanged normalization.
   f=dict(np.load(r['feature_path']));check=normalize(f);assert all(np.allclose(cur[k],check[k],atol=1e-6,rtol=0)for k in cur)
  else:
   if enc is None:
    sys.path.insert(0,str(RGB/'source'));from cache import RGBEncoder;enc=RGBEncoder()
   paths=[]
   for j,pp in enumerate(r['public_frames']):
    zp=E/'data/sanitized'/(r['window_id']+f'_{j}.npz')
    with np.load(pp)as z:np.savez_compressed(zp,**{k:z[k]for k in ['rgb_static','rgb_gripper','robot_obs']})
    paths.append(str(zp))
   f,_=enc.encode(paths);fp=E/'data/features'/(r['window_id']+'.npz');np.savez_compressed(fp,**f);cur=normalize(f);r=r|{'original_public_frames':r['public_frames'],'public_frames':paths,'feature_path':str(fp)}
   lab0=read(r['labels_private'])['frames'][-1]['labels'];cat=np.array(lab0['categorical'][:4]);v=np.array(lab0['categorical_valid'][:4],bool);lab={'window_y':np.r_[cat[:3]==6,cat[:3]==3,cat[3]==2],'window_valid':np.r_[v[:3],v[:3],v[3]]}
  for k in inputs:inputs[k].append(cur[k])
  yy.append(lab['window_y']);vv.append(lab['window_valid']);out.append(r|{'index':i,'public_frame_sha256':[sha(p)for p in r['public_frames']]})
  if(i+1)%26==0:print('DATA',i+1,flush=True)
 arrays={k:np.stack(v)for k,v in inputs.items()};y=np.array(yy,bool);v=np.array(vv,bool);np.savez_compressed(E/'data/public/reader_inputs.npz',**arrays);np.savez_compressed(E/'data/_private/reader_targets.npz',y=y,v=v);save(E/'data/public/windows.json',out)
 counts={}
 for split in ['train','dev']:
  ii=np.array([r['split']==split for r in out]);counts[split]={'windows':int(ii.sum()),'roots':len({r['root_id']for r in out if r['split']==split}),'positive':(y[ii]&v[ii]).sum(0).tolist(),'negative':(~y[ii]&v[ii]).sum(0).tolist(),'invalid':(~v[ii]).sum(0).tolist()}
  assert np.all((y[ii]&v[ii]).sum(0)>0)and np.all((~y[ii]&v[ii]).sum(0)>0)
 save(E/'results/data_audit.json',{'passed':True,'counts':counts,'reencoded_hard_windows':54,'reused_windows':258,'GT_in_public_inputs':False,'old_D_used':False,'public_shapes':{k:list(v.shape)for k,v in arrays.items()}});print(json.dumps(counts),flush=True)
if __name__=='__main__':main()
