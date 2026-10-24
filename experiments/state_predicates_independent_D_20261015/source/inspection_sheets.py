from experiment import *
import numpy as np
from PIL import Image,ImageDraw,ImageFont
F='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
def font(n):return ImageFont.truetype(F,n)
def main():
 m=read(B/'data_manifest.json');windows={r['window_id']:r for r in m['windows']};meta=read(E/'data/_private/evaluation_metadata.json');idx={r['episode_id']:r for r in meta};errors=[r for r in read(E/'results/pair_details.json')if r['model']=='candidate_seed0'and not r['all7_correct']];paths=[]
 for offset in range(0,len(errors),8):
  sheet=Image.new('RGB',(1600,1600),'white');d=ImageDraw.Draw(sheet)
  for k,r in enumerate(errors[offset:offset+8]):
   x=(k%2)*800;y=(k//2)*400;z=idx[r['episode_id']];d.text((x+10,y+5),r['episode_id'],font=font(20),fill='#17263d');d.text((x+10,y+32),'左：动作前  |  右：真实动作后  |  下排：各自手腕相机',font=font(18),fill='#42546a')
   for j,key in enumerate(['before_window_id','after_window_id']):
    with np.load(windows[z[key]]['public_frames'][-1])as arr:sheet.paste(Image.fromarray(arr['rgb_static']).resize((240,240)),(x+10+j*390,y+60));sheet.paste(Image.fromarray(arr['rgb_gripper']).resize((150,150)),(x+235+j*390,y+190))
   d.text((x+10,y+345),f'错误列{r["error_columns"]}；前读出{r["before_readout_correct"]}；后读出{r["actual_after_readout_correct"]}',font=font(18),fill='#a92d43')
  p=E/'figures'/f'error_inspection_{offset//8+1:02d}.png';sheet.save(p);paths.append(str(p))
 first=errors[0];r=idx[first['episode_id']];w=windows[r['before_window_id']];audit=read(r['preaction_prediction_record_path']);prob=np.load(audit['observed_path'])['probability'][0];im=Image.new('RGB',(1600,850),'white');d=ImageDraw.Draw(im);d.text((20,10),'首个影响结果的异常：动作前观测读出把桌面粉块也读成持有',font=font(30),fill='#17263d');d.text((20,60),first['episode_id']+'；以下四帧是真实模型输入，跨度2.1秒',font=font(23),fill='#42546a')
 for k,fp in enumerate(w['public_frames']):
  x=20+k*390;d.text((x,110),f't - {(3-k)*.7:.1f} 秒',font=font(23),fill='#315e9e')
  with np.load(fp)as a:im.paste(Image.fromarray(a['rgb_static']).resize((340,340)),(x,150));im.paste(Image.fromarray(a['rgb_gripper']).resize((168,168)),(x+170,410))
 d.text((20,610),f'当前持红概率 {prob[0]:.6f}；持粉概率 {prob[2]:.6f}；两者超过0.5，按冻结冲突规则均为UNKNOWN。',font=font(24),fill='#a92d43');d.text((20,655),'图像可见红块在夹爪中、粉块仍在桌面；私有接触审计给出同样真值，未进入模型。',font=font(24),fill='#334155');d.text((20,700),'这是固定顺序中最早可定位的识别异常；不能仅据此断言其解释了所有后继预测错误。',font=font(24),fill='#334155');d.text((20,745),'蓝字=输入相对时间，红字=异常概率；大图固定相机，小图手腕相机。未重采、未改阈值。',font=font(24),fill='#334155');im.save(E/'figures/05_first_input_error.png');save(E/'results/inspection_sheets.json',paths)
if __name__=='__main__':main()
