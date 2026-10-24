"""Verify retrieval provenance, episode isolation, reconstruction and mask accounting."""
from pathlib import Path
import json,hashlib,zlib,csv,datetime,shutil
import numpy as np
P=Path('/home/robot/research/projects/wm_symbolic');E=P/'experiments/official_AB_state_pilot_20261018';R=P/'reports/official_AB_state_pilot_20261018'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def save(n,x):
 (E/'results'/f'{n}.json').write_text(json.dumps(x,ensure_ascii=False,indent=2,default=lambda v:v.tolist()if isinstance(v,np.ndarray)else v.item()if isinstance(v,np.generic)else str(v))+'\n')
 if isinstance(x,list)and x:
  with(R/f'{n}.csv').open('w',newline='')as f:
   keys=list(dict.fromkeys(k for r in x for k in r));w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows({k:json.dumps(v,ensure_ascii=False)if isinstance(v,(list,dict))else v for k,v in r.items()}for r in x)
def main():
 R.mkdir(parents=True,exist_ok=True);selection=read(E/'configs/combined_frame_selection.json');roots=selection['roots'];rows=read(E/'results/restoration_refined_masks.json');assert len(rows)==selection['unique_frames']==128;assert len(roots)==4
 ledger=read(E/'results/network_ledger.json');assert ledger['body_bytes_read']==sum(r['body_bytes_read']for r in ledger['requests'])and ledger['body_bytes_read']<=512*1024**2
 for r in ledger['requests']:
  assert r['status']=='complete'and r['http_status']==206 and int(r['content_length'])==r['end']-r['start']+1==r['body_bytes_read'];assert r['content_range'].startswith(f"bytes {r['start']}-{r['end']}/")
  if 'task_ABC_D.zip'in r['url']:assert r['etag']=='"814b0b4be1-5e8bad7e824c0"'
 for path,h in read(E/'configs/restoration_protocol.json')['source_sha256'].items():assert sha(path)==h,path
 mapping=read(E/'configs/verified_scene_ranges.json')['ranges'];factrows=[];metrics=[]
 for root in roots:
  lo,hi=mapping[root['scene']];a,b=root['official_episode_inclusive'];assert lo<=a<b<=hi;assert all(a<=f<=b for f in root['frame_ids']);assert len(root['frame_ids'])<=32
  rr=[r for r in rows if r['root_id']==root['root_id']];assert {r['frame_id']for r in rr}==set(root['frame_ids'])
  for r in rr:
   p=Path(r['source_path']);receipt=read(p.with_suffix('.npz.member.json'));assert sha(p)==r['source_sha256']==receipt['sha256'];assert zlib.crc32(p.read_bytes())&0xffffffff==receipt['crc32'];assert receipt['crc_verified'];assert r['sample_roles']==root['frame_roles'][str(r['frame_id'])]
   truth=r['original_provisional_labels'];assert not any(truth['y'])and truth['positive_labels_admitted']==0
   for j,c in enumerate(['red','blue','pink']):
    for group,index in [('Holding',j),('In',3+j)]:factrows.append({'scene':r['scene'],'root_id':r['root_id'],'frame_id':r['frame_id'],'roles':r['sample_roles'],'fact':f'{group}({c})','provisional_y':truth['y'][index],'known':truth['v'][index],'reason':truth['reason'][index],'restored_t0_y':r['restored_t0']['y'][index],'restored_t1_y':r['restored_t1']['y'][index],'restored_t1_valid':r['restored_t1']['v'][index]})
 for scope in ['all','continuous_restoration','uniform_coverage']:
  for scene in ['all','A','B']:
   rr=[r for r in rows if(scope=='all'or scope in r['sample_roles'])and(scene=='all'or r['scene']==scene)]
   drift=np.array([max(d['position_m']for d in r['one_tick_objects'].values())for r in rr]);pixel=np.array([r['rgb_MAE_0_255']['rgb_static']for r in rr]);tcp=np.array([r['t0_tcp_position_error_m']for r in rr])
   d={'scope':scope,'scene':scene,'frames':len(rr),'independent_episodes':len({r['root_id']for r in rr}),'max_object_drift_mm':float(drift.max()*1000),'median_object_drift_mm':float(np.median(drift)*1000),'p95_object_drift_mm':float(np.quantile(drift,.95)*1000),'frames_drift_gt2mm':int((drift>.002).sum()),'mean_static_RGB_MAE_0_255':float(pixel.mean()),'max_static_RGB_MAE':float(pixel.max()),'max_t0_TCP_error_mm':float(tcp.max()*1000),'max_t0_robot_joint_width_error':max(r['t0_robot_joint_width_max_abs_error']for r in rr),'frames_any_force_contacts_t0':sum(r['force_contacts_t0']>0 for r in rr),'frames_any_force_contacts_t1':sum(r['force_contacts_t1']>0 for r in rr)}
   for group,sl in [('Holding',slice(0,3)),('In',slice(3,6))]:
    d[group+'_restored_t0_positive_frames']=sum(any(r['restored_t0']['y'][sl])for r in rr);d[group+'_restored_t1_positive_frames']=sum(any(r['restored_t1']['y'][sl])for r in rr);d[group+'_known_negative_fact_cells']=sum(sum(r['original_provisional_labels']['v'][sl])for r in rr);d[group+'_unknown_fact_cells']=len(rr)*3-d[group+'_known_negative_fact_cells'];d[group+'_original_verified_positive_fact_cells']=0
   metrics.append(d)
 save('quality_summary',metrics);save('state_label_masks',factrows);save('frame_quality',[{k:r[k]for k in ['scene','root_id','frame_id','sample_roles','t0_scene_obs_max_abs_error','t0_robot_joint_width_max_abs_error','t0_tcp_position_error_m','rgb_MAE_0_255','contacts_t0','contacts_t1','force_contacts_t0','force_contacts_t1','one_tick_objects','original_gripper_action','original_gripper_width']}for r in rows]);save('verification',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'passed':True,'frames':128,'independent_official_episodes':4,'range_requests':len(ledger['requests']),'body_bytes_read':ledger['body_bytes_read'],'byte_cap':512*1024**2,'CRC_and_SHA_checked_frames':128,'label_fact_cells':len(factrows),'C_D_training_frames':0,'new_model_updates':0,'no_unpickling':True,'all_original_positive_force_labels_unknown':True,'sources_unchanged':True});shutil.copy2(__file__,E/'source/verify.py');print('PILOT_VERIFIED',ledger['body_bytes_read'],'bytes',len(factrows),'label cells',flush=True)
if __name__=='__main__':main()
