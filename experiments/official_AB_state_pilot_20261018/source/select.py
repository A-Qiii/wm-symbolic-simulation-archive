"""Episode-isolated fixed sampling from verified official metadata; no labels used."""
from official_ab_fetch import *
def main():
 initialize();assert not(E/'configs/frame_selection.json').exists()
 mapping=read(E/'configs/verified_scene_ranges.json');ep_path=E/'data/metadata/training/ep_start_end_ids.npy';episodes=np.load(ep_path,allow_pickle=False);assert episodes.ndim==2 and episodes.shape[1]==2
 rng=np.random.default_rng(170818);roots=[];ids=[]
 for scene in ['A','B']:
  lo,hi=mapping['ranges'][scene];candidates=[(i,int(a),int(b))for i,(a,b)in enumerate(episodes)if a>=lo and b<=hi and b-a+1>=64]
  assert len(candidates)>=2
  for k in rng.choice(len(candidates),size=2,replace=False):
   index,start,end=candidates[int(k)];clip=int(rng.integers(start,end-15+1));frames=list(range(clip,clip+16));ids.extend(frames);roots.append({'scene':scene,'official_episode_index':index,'official_episode_inclusive':[start,end],'clip_inclusive':[clip,clip+15],'frame_ids':frames,'root_id':f'official_ABC_{scene}_episode_{index:04d}','role':'quarantine_state_label_quality_only','sampled_before_reading_frames':True})
 assert len(set(ids))==64;selected=[]
 for e in entries():
  name=Path(e['name']).name
  if '/training/'in e['name']and name.startswith('episode_')and name.endswith('.npz'):
   i=int(name[8:-4])
   if i in ids:selected.append(e|{'frame_id':i})
 assert len(selected)==64
 budget=sum(e['compressed_size']+30+65535 for e in selected);ledger=read(E/'results/network_ledger.json');assert ledger['body_bytes_read']+budget<=BUDGET
 write(E/'configs/frame_selection.json',{'utc':now(),'seed':170818,'scene_mapping_sha256':sha(E/'configs/verified_scene_ranges.json'),'episode_boundary_sha256':sha(ep_path),'roots':roots,'members':selected,'compressed_frame_bytes':sum(e['compressed_size']for e in selected),'conservative_local_header_budget_bytes':budget,'network_used_before_frames':ledger['body_bytes_read'],'no_outcome_selection':True,'no_C_or_D':True,'no_language_windows_used':True})
 print('SELECTED',[(r['root_id'],r['clip_inclusive'])for r in roots],'frame bytes',sum(e['compressed_size']for e in selected),'conservative budget',budget,flush=True)
 shutil.copy2(__file__,E/'source/select.py')
 for j,e in enumerate(sorted(selected,key=lambda e:e['frame_id'])):
  r=next(r for r in roots if e['frame_id']in r['frame_ids']);p=extract_member(e,E/'data/frames'/r['root_id']/Path(e['name']).name)
  # Numeric NPZ only. Never load or allow arbitrary object arrays.
  import zipfile
  with zipfile.ZipFile(p)as z:
   assert all(m.file_size<4*1024**2 for m in z.infolist());assert sum(m.file_size for m in z.infolist())<16*1024**2
   for m in z.infolist():
    with z.open(m)as f:
     ver=np.lib.format.read_magic(f);shape,order,dtype=np.lib.format._read_array_header(f,ver);assert not dtype.hasobject;assert np.prod(shape,dtype=np.int64)*dtype.itemsize<4*1024**2
  with np.load(p,allow_pickle=False)as a:
   assert a['robot_obs'].shape==(15,)and a['scene_obs'].shape==(24,)and np.isfinite(a['robot_obs']).all()and np.isfinite(a['scene_obs']).all()
  print('FRAME',j+1,'/64',e['frame_id'],flush=True)
 write(E/'results/retrieval_complete.json',{'utc':now(),'frames':64,'independent_episodes':4,'scenes':['A','B'],'body_bytes_read':read(E/'results/network_ledger.json')['body_bytes_read'],'download_budget':BUDGET,'model_updates':0,'simulation_runs':0,'untrusted_pickle_executed':False});print('RETRIEVAL_COMPLETE',flush=True)
if __name__=='__main__':main()
