from common import *
import base64
def main():
 result=read(E/'results/results.json');assert [r['spec']['scene']for r in result]==list('ABCD')
 refs=read(E/'configs/source_references.json')['sha256'];assert all(sha(p)==h for p,h in refs.items())
 frozen=read(E/'configs/pre_run_frozen.json')['sha256'];assert all(sha(p)==h for p,h in frozen.items())
 rows=[];bytes_fields=0;opaque=0;total_files=0
 def check_bytes(x):
  nonlocal bytes_fields,opaque
  if isinstance(x,dict):
   if x.get('raw_bytes_encoding')=='base64':
    b=base64.b64decode(x['data'],validate=True);bytes_fields+=1
    if x['utf8_text']is None:
     try:b.decode('utf8');assert False,'validUTF8 improperly marked opaque'
     except UnicodeDecodeError:opaque+=1
    else:assert b.decode('utf8',errors='strict')==x['utf8_text']
   for v in x.values():check_bytes(v)
  elif isinstance(x,list):
   for v in x:check_bytes(v)
 for state in result:
  scene=state['spec']['scene'];d=E/'data'/scene/'annotation_private';map_=read(d/'body_link_mapping.json')['target_mapping'];check_bytes(read(d/'body_link_mapping.json'))
  assert read(d/'snapshot_before.json')==read(d/'snapshot_after.json');integrity=read(d/'integrity.json');assert integrity['post_primary_physics_ticks']==0 and integrity['same_tick_all4calls']
  calls=read(d/'render_call_ledger.json');assert len(calls)==4 and [r['phase']for r in calls]==['primary','primary','baseline','baseline']and len({r['tick']for r in calls})==1
  with np.load(E/'data'/scene/'public/rgb.npz',allow_pickle=False)as z:assert set(z.files)=={'rgb_static','rgb_gripper'}
  with np.load(d/'drawer_world_mesh.npz',allow_pickle=False)as z:world=z['vertices'];faces=z['faces'];labels=z['semantic_class_by_face']
  for ci,cam in enumerate(['static','gripper']):
   with np.load(d/f'primary_{cam}.npz',allow_pickle=False)as z:p={k:z[k]for k in z.files}
   with np.load(d/f'baseline_flag_off_{cam}.npz',allow_pickle=False)as z:assert np.array_equal(z['rgb'],p['rgb'])and np.array_equal(z['depth_buffer'],p['depth_buffer'])
   with np.load(d/f'capture_call{ci:03d}.npz',allow_pickle=False)as z:assert all(np.array_equal(z[k],p[k])for k in ['rgb','depth_buffer','object_link_mask'])
   with np.load(d/f'semantics_{cam}.npz',allow_pickle=False)as z:q={k:z[k]for k in z.files}
   seg=p['object_link_mask'];native=(seg>=0)&((seg%2**24)==map_['table_uid'])&((seg//2**24-1)==map_['drawer_link']);sem=q['drawer_parts'];assert np.array_equal(native,sem>0)
   assert np.array_equal(q['front_plus_handle'],(sem==1)|(sem==2));assert np.array_equal(q['unknown_drawer_pixels'],sem==255)
   for i,family in enumerate(map_['finger_families'],1):assert np.array_equal(q[f'finger{i}'],(seg>=0)&(seg%2**24==map_['robot_uid'])&np.isin(seg//2**24-1,family))
   ll=read(d/f'visible_labels_{cam}.json')
   for name,key in [('finger1','finger1'),('finger2','finger2'),('whole_drawer_diagnostic','whole_drawer'),('drawer_front_plus_handle','front_plus_handle')]:
    yy,xx=q[key].nonzero();bb=[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)]if len(xx)else None
    assert ll[name]['visible_bbox_xyxy']==bb and ll[name]['visible_pixels']==len(xx)
    valid=bool(len(xx)>0 and(name!='drawer_front_plus_handle'or not(sem==255).any()));assert ll[name]['bbox_valid']==valid
   # Independent plane/Gram-barycentric verification of all assigned faces.
   c=read(d/f'camera_{cam}.json');V=np.array(c['viewMatrix']).reshape(4,4).T;Q=np.array(c['projectionMatrix']).reshape(4,4).T;inv=np.linalg.inv(Q@V);h,w=seg.shape;y,x=np.where((sem>0)&(sem<255));face=q['drawer_face_id'][y,x];assert np.array_equal(labels[face],sem[y,x]);tri=world[faces[face]]
   clip=np.c_[2*(x+.5)/w-1,1-2*(y+.5)/h,np.zeros(len(x)),np.ones(len(x))];mid=clip@inv.T;mid=mid[:,:3]/mid[:,3:];eye=np.linalg.inv(V)[:3,3];directions=mid-eye
   n=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);t=np.einsum('ij,ij->i',n,tri[:,0]-eye)/np.einsum('ij,ij->i',n,directions);hit=eye+t[:,None]*directions
   camera_hit=np.c_[hit,np.ones(len(hit))]@V.T;depth=-camera_hit[:,2];near,far=c['near'],c['far'];native_depth=near*far/(far-(far-near)*p['depth_buffer'][y,x].astype(np.float64));err=np.abs(depth-native_depth);assert np.all(err<=.0005)
   a=tri[:,1]-tri[:,0];b=tri[:,2]-tri[:,0];dlt=hit-tri[:,0];aa=np.einsum('ij,ij->i',a,a);ab=np.einsum('ij,ij->i',a,b);bb=np.einsum('ij,ij->i',b,b);da=np.einsum('ij,ij->i',dlt,a);db=np.einsum('ij,ij->i',dlt,b);den=aa*bb-ab*ab;u=(da*bb-db*ab)/den;v=(db*aa-da*ab)/den;assert np.all(u>=-1e-7)&np.all(v>=-1e-7)&np.all(u+v<=1+1e-7)
   unknown=[]
   for yy,xx in zip(*np.where(sem==255)):
    unknown.append({'x':int(xx),'y':int(yy),'depth_error_m':float(q['depth_error_m'][yy,xx]),'native_depth_patch_5x5_m':p['depth_m'][max(0,yy-2):yy+3,max(0,xx-2):xx+3].tolist(),'semantic_patch_5x5':sem[max(0,yy-2):yy+3,max(0,xx-2):xx+3].tolist(),'interpretation':'At a local surface depth discontinuity; renderer coverage versus exact center-ray boundary discrepancy is plausible, not independently proven. Kept unknown without relabeling or gate relaxation.'})
   rows.append({'scene':scene,'camera':cam,'native_drawer_pixels':int(native.sum()),'independently_verified_assigned_pixels':len(x),'unknown_pixels':unknown,'max_assigned_pixel_depth_error_m':float(err.max())if len(err)else None,'front_bbox_valid':ll['drawer_front_plus_handle']['bbox_valid'],'finger_valid_boxes':int(ll['finger1']['bbox_valid'])+int(ll['finger2']['bbox_valid']),'RGB_depth_bitwise_equal':True,'state_unchanged':True,'source_file':str(d/f'primary_{cam}.npz')})
 for f in(E/'data').rglob('*.npz'):
  with np.load(f,allow_pickle=False)as z:
   for key in z.files:assert z[key].dtype!=object
  total_files+=1
 summary={'utc':now(),'passed_data_integrity':True,'records':8,'all_npz_files_readable_without_pickle':total_files,'public_RGB_only':True,'raw_bytes_fields_verified':bytes_fields,'nonUTF8_fields_preserved_losslessly':opaque,'native_drawer_pixels':sum(r['native_drawer_pixels']for r in rows),'independently_verified_assigned_pixels':sum(r['independently_verified_assigned_pixels']for r in rows),'unknown_pixels':sum(len(r['unknown_pixels'])for r in rows),'valid_finger_boxes':sum(r['finger_valid_boxes']for r in rows),'valid_front_boxes':sum(r['front_bbox_valid']for r in rows),'complete_semantic_gate_passed_all8':all(r['front_bbox_valid']for r in rows),'source_hash_entries_checked':len(refs),'frozen_hash_entries_checked':len(frozen),'rows':rows}
 save(E/'results/independent_verification.json',summary);shutil_copy=__import__('shutil').copy2;shutil_copy(__file__,E/'source/verify.py')
 print({k:v for k,v in summary.items()if k!='rows'})
if __name__=='__main__':main()
