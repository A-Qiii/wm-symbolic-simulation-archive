from experiment import *
from render import text,public,CN
from PIL import Image,ImageDraw
import numpy as np,imageio.v2 as iio
def main():
 q=frozen();first=read(E/'results/single_replacement_diagnostic.json')['first_error_in_actual_fixed_collection_order'];rid=first.rsplit('__b',1)[0];rows={r['drawer']:r for r in read(E/'results/call_details.json')if r['root_id']==rid};m=read(B/'data_manifest.json');wins={w['window_id']:w for w in m['windows']};end=96;frames=[];path=E/'videos'/f'{rid}_closed_open.mp4';fps=30
 with iio.get_writer(path,fps=fps,codec='libx264',quality=8,macro_block_size=16)as writer:
  for offset in range(-63,end+1):
   im=Image.new('RGB',(1280,800),'white');text(im,(20,12),f'{rid}：已持{CN[rows["closed"]["held_color"]]}，请求抓{CN[rows["closed"]["target_color"]]}',28);text(im,(20,55),f'相对调用时刻 {offset/30:+.2f}秒 | '+('真实历史观察'if offset<0 else'真实拒绝后的96步保持'),22)
   for side,x,col in [('closed',20,'#2474b5'),('open',650,'#d97706')]:
    r=rows[side];anchor=wins[r['before_window']]['anchor_frame'];fp=B/'episodes'/r['episode_id']/'public'/f'{anchor+offset:05d}.npz';static,wrist=public(fp);text(im,(x,98),'抽屉'+('关闭'if side=='closed'else'打开'),26,col);im.paste(static.resize((390,390)),(x,140));im.paste(wrist.resize((180,180)),(x+410,140));text(im,(x,540),'静态相机',20);text(im,(x+410,335),'腕部相机',20)
    ci=COLORS.index(r['target_color']);oi=COLORS.index(r['held_color'])
    if offset>=0:
     text(im,(x,580),f'调用前保存预测：持目标p={r["p_prediction"][ci]:.4f}',23,col);text(im,(x,620),f'持原物p={r["p_prediction"][oi]:.4f}；全7事实'+('正确'if r['candidate_all7_correct']else'错误/UNKNOWN'),23,col);text(im,(x,662),'实际：拒绝，继续持原物，目标仍在桌面',22)
    else:text(im,(x,585),'公开画面与机器人状态组成因果历史',22)
   text(im,(20,726),'蓝/橙标题=抽屉闭/开；物体颜色见实拍。帧来自30Hz真实控制；未插值、未瞬移。',22);text(im,(20,762),'预测在调用前生成；后续真实画面仅用于检验，最后画面停留3秒。',20)
   writer.append_data(np.array(im));last=im
  for _ in range(90):writer.append_data(np.array(last))
 reader=iio.get_reader(path);count=0
 for arr in reader:assert arr.shape==(800,1280,3);count+=1
 meta=reader.get_meta_data();reader.close();assert count==250;last.save(E/'figures'/f'{rid}_video_final.png');save(E/'results/video_validation.json',{'path':str(path),'sha256':sha(path),'width':1280,'height':800,'fps':fps,'decoded_frames':count,'real_source_control_frames':160,'final_freeze_frames':90,'duration_seconds':count/fps,'source_roots':[rid],'dual_conditions_and_dual_cameras':True,'audio':False,'all_frames_decoded':True});print('VIDEO_VALIDATED',count,flush=True)
if __name__=='__main__':main()
