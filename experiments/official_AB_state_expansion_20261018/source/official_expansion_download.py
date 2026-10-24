from pathlib import Path
import zipfile,shutil
import numpy as np
import official_ab_fetch as F
from official_expansion_fetch import E,read,write,extract
def main():
 assert not(E/'results/retrieval_complete.json').exists()
 for p,h in read(E/'configs/frozen_selection.json')['sha256'].items():assert F.sha(p)==h,p
 inputs=read(E/'configs/inputs.json');index={r['frame_id']:r for r in inputs};quality=[]
 shutil.copy2(__file__,E/'source'/Path(__file__).name)
 for j,m in enumerate(sorted(read(E/'configs/frame_members.json'),key=lambda m:m['frame_id'])):
  r=index[m['frame_id']];p=extract(m,r['source_path'])
  with zipfile.ZipFile(p)as z:
   assert all(m.file_size<4*1024**2 for m in z.infolist());assert sum(m.file_size for m in z.infolist())<16*1024**2
   for m in z.infolist():
    with z.open(m)as f:
     ver=np.lib.format.read_magic(f);shape,order,dtype=np.lib.format._read_array_header(f,ver);assert not dtype.hasobject and np.prod(shape,dtype=np.int64)*dtype.itemsize<4*1024**2
  with np.load(p,allow_pickle=False)as a:
   assert a['robot_obs'].shape==(15,)and a['scene_obs'].shape==(24,)and np.isfinite(a['robot_obs']).all()and np.isfinite(a['scene_obs']).all()and float(a['robot_obs'][-1])in[-1,1]
   assert a['rgb_static'].shape==(200,200,3)and a['rgb_gripper'].shape==(84,84,3)
  r['source_sha256']=F.sha(p);quality.append({'frame_id':r['frame_id'],'root_id':r['root_id'],'split':r['split'],'numeric_npz_valid':True,'sha256':r['source_sha256']})
  if(j+1)%16==0:print('SOURCE_FRAMES',j+1,'/240','CUMULATIVE_MiB',read(E/'results/network_ledger.json')['body_bytes_read']/2**20,flush=True)
 for r in inputs:assert F.sha(r['source_path'])==r['source_sha256']
 write(E/'configs/realized_inputs.json',inputs);write(E/'results/frame_quality.json',quality)
 ledger=read(E/'results/network_ledger.json');write(E/'results/retrieval_complete.json',{'utc':F.now(),'new_frames':240,'new_source_episodes':8,'old_source_retests':16,'total_runs':256,'cumulative_body_bytes':ledger['body_bytes_read'],'new_body_bytes':ledger['body_bytes_read']-ledger['base_body_bytes_read'],'cumulative_cap_bytes':F.BUDGET,'no_training':True,'untrusted_pickle_executed':False,'network_requests_this_round':len(ledger['requests'])})
 print('EXPANSION_DOWNLOAD_COMPLETE',ledger['body_bytes_read']/2**20,flush=True)
if __name__=='__main__':main()
