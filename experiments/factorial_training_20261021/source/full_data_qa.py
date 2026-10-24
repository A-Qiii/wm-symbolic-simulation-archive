from pathlib import Path
import json,hashlib,numpy as np,time,datetime,subprocess,collections,imageio_ffmpeg
FFMPEG=imageio_ffmpeg.get_ffmpeg_exe()
P=Path('/home/robot/research/projects/wm_symbolic');D=P/'experiments/factorial_expansion_20261020';S=P/'experiments/factorial_training_20261021'
def read(p):return json.loads(p.read_text())
def save(name,x):(S/'results'/name).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
assert read(D/'results/final_data_audit.json')['passed']
allowed={'rgb_static','rgb_gripper','depth_static','depth_gripper','robot_obs','seg_static','seg_gripper'}
for cam in ['static','gripper']:
 allowed|={f'calib_{cam}_{k}' for k in ['width','height','fov','nearval','farval','viewMatrix','projectionMatrix']}
rows=read(D/'data/data_manifest.json')['records'];checked=0;h=hashlib.sha256();shapes=collections.Counter();start=time.time();video_samples=[]
for i,r in enumerate(rows):
 d=Path(r['labels_path']).parent;files=[d/'initial_public.npz',d/'final_public.npz']+[d/f'boundary_history_{j}.npz'for j in range(4)]
 if r['step']==0:files+=[d.parent/f'root_observation_{j}.npz'for j in range(4)]
 for f in files:
  raw=f.read_bytes();h.update(str(f.relative_to(D)).encode());h.update(hashlib.sha256(raw).digest())
  with np.load(f)as z:
   assert set(z.files)<=allowed,(str(f),z.files)
   assert {'rgb_static','rgb_gripper','depth_static','depth_gripper','robot_obs','seg_static','seg_gripper'}<=set(z.files),str(f)
   for k in z.files:
    a=z[k];assert np.isfinite(a).all(),(str(f),k)
   for cam in ['static','gripper']:
    assert z['rgb_'+cam].dtype==np.uint8 and z['rgb_'+cam].ndim==3 and z['rgb_'+cam].shape[2]==3
    assert z['depth_'+cam].shape==z['seg_'+cam].shape==z['rgb_'+cam].shape[:2]
    shapes[(cam,)+z['rgb_'+cam].shape]+=1
  checked+=1
 if i%100==0:
  save('full_qa_progress.json',dict(requests=i+1,public_files=checked,seconds=time.time()-start));print('QA',i+1,checked,round(time.time()-start,1),flush=True)
 # Fixed stratified video sample: first episode of every twentieth family, both cameras, all six steps.
 if r['episode_id'].endswith('_b1') and 'state0_'in r['episode_id'] and int(r['family_id'].rsplit('_',1)[1])%20==0:
  for cam in ['static','gripper']:
   f=d/f'rgb_{cam}.mp4'
   p=subprocess.run([FFMPEG,'-v','error','-i',str(f),'-f','null','-'],capture_output=True)
   assert p.returncode==0 and not p.stderr,(str(f),p.stderr.decode());video_samples.append(str(f.relative_to(D)))
result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,episodes=800,requests=4800,public_npz_fully_decoded=checked,public_file_digest=h.hexdigest(),allowlist_shapes_and_finite=True,video_full_decode_sample=len(video_samples),video_sampling='Branch1 state0, fixed every twentieth family index, all six actions, both cameras; includes empty/refused video',sampled_videos=video_samples,seconds=time.time()-start,no_resampling=True)
save('full_data_qa.json',result);print('FULL_QA_PASS',checked,len(video_samples),flush=True)
