from common import *
import sys,json,subprocess,collections
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import imageio.v2 as imageio
R=P/'reports/state_semantics_v2_20261015';FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'

def main():
 pair=read(R/'tables/illustrated_pair.json');wins=read(E/'data/windows.json');font=ImageFont.truetype(FONT,18);small=ImageFont.truetype(FONT,15);manifest=[]
 choices=[(pair['branches'][c]['episode_id'],c,'同根堆叠：真实受阻，中间状态'if c=='stacked'else'同根并排：固定Close真实关严')for c in ['stacked','separated']]
 chain=next(e for e in read(B/'manifest.json')['episodes']if e['split']=='dev'and e['condition']=='chain');choices.append((chain['episode_id'],'four_skill_C','C真实四技能数据采集：固定脚本，不是新模型闭环'))
 for eid,name,title in choices:
  folder=B/'episodes'/eid;frames=sorted((folder/'public').glob('*.npz'));acts=read(folder/'_private/actions.json');summary=read(folder/'_private/summary.json');out=R/'videos'/(name+'.mp4');writer=iio=imageio.get_writer(out,fps=15,codec='libx264',quality=7,macro_block_size=16);n=0
  for j in sorted(set(range(0,len(frames),6))|{len(frames)-1}):
   z=np.load(frames[j]);im=Image.new('RGB',(880,592),'#f3f5f7');im.paste(Image.fromarray(z['rgb_static']).resize((480,480)),(12,82));im.paste(Image.fromarray(z['rgb_gripper']).resize((336,336)),(528,82));d=ImageDraw.Draw(im);d.text((12,10),title,font=font,fill='#193b48');d.text((12,42),f'真实双相机｜3倍播放｜控制时间 {j/30:.1f}s｜左静态、右腕部｜无补帧',font=small,fill='#384a56');active=next((a for a in acts if a['start']<=j<a['end']),None);past=[a for a in acts if a['end']<=j];d.text((520,432),('当前：'+active['skill'])if active else'复位后保持／已执行完',font=small,fill='#193b48')
   if past:d.text((520,460),'最近原生结果：'+past[-1]['native']['status'],font=small,fill='#193b48')
   if j==len(frames)-1:d.text((520,490),f"最终GT {summary['final_labels']['drawer_class']}\n实际开度 {summary['final_labels']['drawer_q_m']*100:.2f}cm",font=small,fill='#9e4a37')
   d.text((12,566),'GT仅在最后帧作评分说明；采集调用的是固定执行器，不给视觉模型补答案。',font=small,fill='#384a56');writer.append_data(np.asarray(im));n+=1
  writer.close();reader=imageio.get_reader(out);assert reader.count_frames()==n;reader.get_data(n-1);reader.close();manifest.append({'path':str(out),'episode_id':eid,'title':title,'frames':n,'fps':15,'stride':6,'source_control_fps':30,'speed':3,'actual_last_frame_included':True,'source_frames':len(frames),'sha256':sha(out),'all_frames_decoded':True})
 save(R/'videos/manifest.json',manifest)
 # Real CLI calls run later in world_model_env; preserve selected cases deterministically.
 queries=[]
 for state in ['Closed','Open','Intermediate']:
  w=next(w for w in wins if w['split']=='dev'and w['targets_v2_1']['drawer_class']==state and all(w['targets_v2_1']['v'][6:]));queries.append({'window_id':w['window_id'],'query':state+'(drawer)','label_class_for_report_only':state})
 w=next(w for w in wins if w['split']=='dev'and w['targets_v2_1']['y'][3]and w['targets_v2_1']['v'][3]);queries.append({'window_id':w['window_id'],'query':'In(obj_red,drawer)','truth_for_report_only':True});save(R/'tables/query_commands.json',queries);print('MEDIA_AND_QUERY_CASES_COMPLETE',flush=True)
if __name__=='__main__':main()
