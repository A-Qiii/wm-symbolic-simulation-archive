from pathlib import Path
import json,hashlib
import numpy as np
import imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont
P=Path('/home/robot/research/projects/wm_symbolic');E=P/'experiments/state_predicates_interface_acceptance_20261014';B=P/'data/predicate_interface_acceptance_20261014'
def read(p):return json.loads(Path(p).read_text())
def main():
 meta=read(E/'data/metadata.json');ev=read(E/'results/evaluation.json');out=[];fontpath='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc';font=ImageFont.truetype(fontpath,16);small=ImageFont.truetype(fontpath,14)
 for r,q in zip(meta,ev['details']):
  path=E/'videos'/(r['unique_window_id']+'.mp4');assert not path.exists();wr=imageio.get_writer(str(path),fps=30,macro_block_size=1,codec='libx264',quality=7);j=r['target_index'];value='UNKNOWN'if not q['known'][j]else str(q['probabilities'][j]>=.5);truth=str(q['truth'][j]);anchor=r['anchor_frame'];indices=r['history_frame_indices']
  try:
   for fi in range(anchor-63,anchor+1):
    with np.load(B/'episodes'/r['root_id']/'public'/f'{fi:04d}.npz')as z:
     canvas=Image.new('RGB',(700,550),'#f6f8fa');draw=ImageDraw.Draw(canvas);draw.text((12,8),r['root_id'],font=font,fill='#20394a');draw.text((12,32),' / '.join(r['conditions']),font=small,fill='#20394a');draw.text((12,57),f'相对末帧 {(fi-anchor)/30:+.2f} 秒  |  原控制帧 {fi}  |  橙框=模型输入帧',font=font,fill='#20394a');draw.text((12,84),'静态相机',font=font,fill='#20394a');draw.text((430,84),'腕部相机',font=font,fill='#20394a');canvas.paste(Image.fromarray(z['rgb_static']).resize((400,400),Image.Resampling.NEAREST),(12,110));canvas.paste(Image.fromarray(z['rgb_gripper']).resize((252,252),Image.Resampling.NEAREST),(430,184))
     if fi in indices:draw.rectangle((10,108,414,512),outline='#d77e29',width=3);draw.rectangle((428,182,684,438),outline='#d77e29',width=3)
     draw.text((12,522),f'末帧目标held：真值{truth}；冻结模型{value}，p={q["probabilities"][j]:.5f}（整窗输出）',font=small,fill='#20394a');wr.append_data(np.asarray(canvas))
  finally:wr.close()
  out.append({'window':r['unique_window_id'],'root_id':r['root_id'],'conditions':r['conditions'],'path':str(path),'frames':64,'fps':30,'first_frame':anchor-63,'anchor_frame':anchor,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'model_inputs':indices,'video_codec':'H264 derivative for viewing; original lossless NPZ retained; only four orange-marked frames enter model'})
 (E/'videos/index.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'videos':len(out),'frames_each':64,'original_npz_retained':True}))
if __name__=='__main__':main()
