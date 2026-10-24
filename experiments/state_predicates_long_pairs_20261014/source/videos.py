from pathlib import Path
import json,gzip,hashlib,time,argparse
import numpy as np
import imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont
P=Path('/home/robot/research/projects/wm_symbolic');E=P/'experiments/state_predicates_long_pairs_20261014';B=P/'data/predicate_long_pairs_20261014'
def read(p):return json.loads(Path(p).read_text())
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--follow',action='store_true');args=ap.parse_args();fontpath='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc';font=ImageFont.truetype(fontpath,15);small=ImageFont.truetype(fontpath,13);out=read(E/'videos/index.json')if(E/'videos/index.json').exists()else[];started=time.monotonic()
 while True:
  m=read(B/'data_manifest.json');done={r['root_id']for r in out};todo=[r for r in m['roots']if r['root_id']not in done]
  if not todo:
   if len(done)==30 or not args.follow:break
   if time.monotonic()-started>2400:raise TimeoutError('Bounded video follow timed out; original data retained')
   time.sleep(5);continue
  root=todo[0]
  assert root['collection_error']is None
  rid=root['root_id'];path=E/'videos'/(rid+'.mp4');assert not path.exists();folder=B/'episodes'/rid
  with gzip.open(folder/'_private/trace.json.gz','rt')as f:tr=json.load(f)
  inputs={i for w in m['windows']if w['root_id']==rid for i in w['history_frame_indices']};wr=imageio.get_writer(str(path),fps=10,macro_block_size=1,codec='libx264',quality=7,ffmpeg_params=['-threads','2']);indices=list(range(0,len(tr),3))
  try:
   for fi in indices:
    q=tr[fi];im=Image.new('RGB',(640,390),'#f5f8fa');d=ImageDraw.Draw(im);d.text((10,7),rid,font=font,fill='#18374a');d.text((10,30),f'真实仿真时间 {fi/30:.2f}s | 控制帧 {fi} | 宏 {q["macro_index"]}',font=small,fill='#18374a');d.text((10,53),q['segment']+' / '+q['action_private'].get('phase',''),font=small,fill='#18374a');d.text((10,77),'静态相机',font=small,fill='#18374a');d.text((355,77),'腕部相机',font=small,fill='#18374a')
    with np.load(folder/'public'/f'{fi:05d}.npz')as z:im.paste(Image.fromarray(z['rgb_static']).resize((260,260),Image.Resampling.NEAREST),(10,98));im.paste(Image.fromarray(z['rgb_gripper']).resize((210,210),Image.Resampling.NEAREST),(355,123))
    target=q['targets'];labels=' '.join('?'if not v else '1'if y else '0'for y,v in zip(target['y'],target['valid']));d.text((10,366),'物理标签 held红蓝粉 / in红蓝粉 / open：'+labels,font=small,fill='#18374a');wr.append_data(np.asarray(im))
  finally:wr.close()
  out.append({'root_id':rid,'path':str(path),'frames':len(indices),'source_frame_indices':indices,'fps':10,'original_control_Hz':30,'time_speed':1,'original_all30Hz_frames_retained':True,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()});(E/'videos/index.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print('VIDEO',rid,len(indices),flush=True)
 (E/'videos/index.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print('ALL_ROOT_VIDEOS_COMPLETE',len(out),flush=True)
if __name__=='__main__':main()
