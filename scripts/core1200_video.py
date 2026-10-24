"""Continuous real dual-camera frames; no freeze padding or synthetic actions."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import imageio.v2 as imageio
P=Path('/home/robot/research/projects/wm_symbolic');B=P/'data/core1200_v1';R=P/'runs/core1200_v1'
FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
NAMES={'open_drawer':'打开抽屉','close_drawer':'关闭抽屉','lift_red_block_table':'抓取红块','lift_blue_block_table':'抓取蓝块','lift_pink_block_table':'抓取粉块','place_in_drawer':'放入抽屉','reobserve':'短暂重新观察'}
STATUS={'success':'执行器达到终止条件','refused':'执行器拒绝，零动作','failed':'执行器未通过终止条件','reobserved':'重新观察'}
def main():
 a=argparse.ArgumentParser();a.add_argument('--episode');a.add_argument('--run',type=Path);a.add_argument('--name',required=True);a.add_argument('--title',required=True);args=a.parse_args();segments=[]
 if args.episode:
  rows=[x for x in json.loads((B/'data_manifest.json').read_text())['records'] if x['episode_id']==args.episode]
  for row in sorted(rows,key=lambda x:x['step']):
   d=Path(row['obs_before']).parent;res=json.loads((d/'result_private.json').read_text());segments.append((d,row['skill'],res))
 else:
  for f in sorted((args.run/'execution/evaluator_only').glob('*/summary.json')):
   x=json.loads(f.read_text());skill=f.parent.name.split('_',1)[1];segments.append((f.parent,skill,x['result']))
 assert segments
 dest=P/'reports/core1200_media';dest.mkdir(exist_ok=True);output=dest/args.name
 if output.exists():raise FileExistsError(output)
 writer=imageio.get_writer(str(output),fps=30,codec='libx264',quality=8,macro_block_size=1);font=ImageFont.truetype(FONT,22);small=ImageFont.truetype(FONT,17);frames=0;index=[]
 try:
  for si,(d,skill,res) in enumerate(segments):
   readers=[imageio.get_reader(str(d/(cam+'.mp4'))) for cam in ['rgb_static','rgb_gripper']];count=readers[0].count_frames();assert count==readers[1].count_frames()==res['controls']+1
   index.append({'skill':skill,'controls':res['controls'],'raw_frames':count,'start_output_frame':frames,'status':res['status']})
   for local,(left,right) in enumerate(zip(readers[0],readers[1])):
    if si and local==0:continue
    im=Image.new('RGB',(960,624),(244,247,251));draw=ImageDraw.Draw(im);draw.text((20,13),args.title,font=font,fill=(23,47,70));draw.text((20,48),f'步骤 {si+1}/{len(segments)}：{NAMES.get(skill,skill)}　|　该请求最终记录：{STATUS.get(res["status"],res["status"])}',font=small,fill=(63,84,108))
    for x,pix in [(16,left),(496,right)]:im.paste(Image.fromarray(pix).resize((448,448),Image.Resampling.NEAREST),(x,112))
    draw.text((20,80),'静态相机：整体场景与对象位置',font=small,fill=(30,80,120));draw.text((500,80),'夹爪相机：实际抓取 / 支撑细节',font=small,fill=(30,80,120))
    text=f'模拟时间 {frames/30:.2f}s　当前步骤控制帧 {local}/{res["controls"]}　真实双相机 · 30 Hz'
    draw.text((20,570),text,font=small,fill=(23,47,70))
    note='不补静止帧、不插入动作。零动作拒绝保持真实状态；步骤边界仅去除重复起始帧。'
    if si>0 and segments[si-1][2]['controls']==0:note='上一请求被拒绝，实际控制步数为0；本段展示随后请求产生的真实动作。'
    draw.text((20,596),note,font=small,fill=(80,94,111));writer.append_data(np.asarray(im));frames+=1
    if si==len(segments)-1 and local==count-1:im.save(dest/(output.stem+'_final.png'))
   for reader in readers:reader.close()
 finally:writer.close()
 assert frames==1+sum(s[2]['controls'] for s in segments)
 check=imageio.get_reader(str(output));assert check.count_frames()==frames;check.close()
 receipt={'path':str(output),'frames':frames,'fps':30,'duration_seconds':frames/30,'source_segments':index,'only_real_control_frames':True,'duplicate_boundary_start_frames_removed':True,'no_freeze_padding':True,'sha256':hashlib.sha256(output.read_bytes()).hexdigest()};(dest/(output.stem+'.json')).write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
