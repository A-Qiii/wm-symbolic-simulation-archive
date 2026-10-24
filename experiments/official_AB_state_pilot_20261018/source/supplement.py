"""Restore originally proposed evenly spaced anchors before any label inspection."""
try:
 from official_ab_fetch import *
except ModuleNotFoundError:
 from fetch import *
def main():
 assert read(E/'results/retrieval_complete.json')['frames']==64
 assert not(E/'configs/combined_frame_selection.json').exists()
 assert not(E/'results/restoration.json').exists()and not(E/'data/restored').exists(),'Must freeze added sampling before label/restoration inspection'
 base=read(E/'configs/frame_selection.json');combined=[];additional=set()
 for root in base['roots']:
  start,end=root['official_episode_inclusive'];anchors=np.linspace(start,end-1,8,dtype=np.int64).tolist();frames=sorted(set(anchors+[x+1 for x in anchors]));original=set(root['frame_ids']);additional.update(set(frames)-original)
  combined.append(root|{'continuous_frame_ids':root['frame_ids'],'coverage_anchor_ids':anchors,'coverage_frame_ids':frames,'frame_ids':sorted(original|set(frames)),'frame_roles':{str(x):[label for label,yes in [('continuous_restoration',x in original),('uniform_coverage',x in frames)]if yes]for x in original|set(frames)}})
 selected=[]
 for e in entries():
  n=Path(e['name']).name
  if '/training/'in e['name']and n.startswith('episode_')and n.endswith('.npz'):
   i=int(n[8:-4])
   if i in additional:selected.append(e|{'frame_id':i})
 assert len(selected)==len(additional);total=sum(len(r['frame_ids'])for r in combined);assert total<=128
 cost=sum(e['compressed_size']+30+65535 for e in selected);used=read(E/'results/network_ledger.json')['body_bytes_read'];assert cost+used<=BUDGET
 write(E/'configs/combined_frame_selection.json',{'utc':now(),'before_new_frame_download_and_any_restoration_or_label_inspection':True,'original_proposal':'8 uniformly spaced anchors plus successors per episode','initial_change':'Initial retrieval script selected16 consecutive frames to inspect technical reconstruction and contact temporal continuity, interpreting the requested short fragments; this change was not recorded in the proposal before execution. Original continuous clips preserved.','authorized_correction':'Parent explicitly requested restore uniform8-anchor+successor set in addition, same4 episodes, <=128 unique frames, <=512MiB total, no label-based selection.','seed':170818,'uniform_rule':'floor of linspace(start,end-1,8) within each original official episode; include each immediate successor; deduplicate','roots':combined,'additional_members':selected,'unique_frames':total,'additional_frames':len(additional),'additional_compressed_bytes':sum(e['compressed_size']for e in selected),'conservative_request_budget_bytes':cost,'network_used_before_addition':used,'independent_units':4,'no_training':True})
 shutil.copy2(__file__,E/'source/supplement.py');print('ADDITIONAL_FROZEN',len(additional),'TOTAL',total,'compressed',sum(e['compressed_size']for e in selected),flush=True)
 for j,e in enumerate(sorted(selected,key=lambda e:e['frame_id'])):
  root=next(r for r in combined if e['frame_id']in r['frame_ids']);p=extract_member(e,E/'data/frames'/root['root_id']/Path(e['name']).name)
  import zipfile
  with zipfile.ZipFile(p)as z:
   assert all(m.file_size<4*1024**2 for m in z.infolist());assert sum(m.file_size for m in z.infolist())<16*1024**2
   for m in z.infolist():
    with z.open(m)as f:
     ver=np.lib.format.read_magic(f);shape,order,dtype=np.lib.format._read_array_header(f,ver);assert not dtype.hasobject;assert np.prod(shape,dtype=np.int64)*dtype.itemsize<4*1024**2
  with np.load(p,allow_pickle=False)as a:assert a['robot_obs'].shape==(15,)and a['scene_obs'].shape==(24,)and np.isfinite(a['scene_obs']).all()
  print('ADDED_FRAME',j+1,'/',len(selected),e['frame_id'],flush=True)
 write(E/'results/combined_retrieval_complete.json',{'utc':now(),'frames':total,'independent_episodes':4,'continuous_frames':64,'coverage_anchor_pairs':32,'body_bytes_read':read(E/'results/network_ledger.json')['body_bytes_read'],'download_budget':BUDGET,'model_updates':0,'simulation_runs':0,'untrusted_pickle_executed':False});print('COMBINED_RETRIEVAL_COMPLETE',flush=True)
if __name__=='__main__':main()
