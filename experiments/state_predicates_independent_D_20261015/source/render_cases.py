"""Post-test evidence renderer; reads recorded frames, never runs physics or predicts."""
from experiment import *
import numpy as np,os
from PIL import Image,ImageDraw,ImageFont
import imageio.v2 as imageio,imageio_ffmpeg
FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
def font(n):return ImageFont.truetype(FONT,n)
LABELS=['持红块','持蓝块','持粉块','红块在抽屉','蓝块在抽屉','粉块在抽屉','抽屉打开']
NAME={'open':'打开抽屉','close':'关闭抽屉','lift':'抓起指定色块','place':'放入抽屉'}
def cameras(path,width=380):
 with np.load(path)as z:
  im=Image.new('RGB',(width,420),'#edf1f4');d=ImageDraw.Draw(im);a=Image.fromarray(z['rgb_static']).resize((300,300));b=Image.fromarray(z['rgb_gripper']).resize((168,168));im.paste(a,(0,25));im.paste(b,(width-168,245));d.text((0,0),'固定相机',font=font(19),fill='#25324a');d.text((width-168,220),'手腕相机',font=font(19),fill='#25324a');return im
def syms(vals):return ' / '.join('未知'if x=='UNKNOWN'else str(x)for x in vals)
def triptych(meta,detail,windows):
 im=Image.new('RGB',(1440,860),'white');d=ImageDraw.Draw(im);d.text((25,18),f"{NAME[meta['family']]} · {meta['episode_id']}",font=font(27),fill='#17263d');d.text((25,58),f"条件：{meta['condition']}  |  实际控制器状态：{meta['primitive_status']}  |  七事实预测：{'全部正确'if detail['all7_correct']else'存在错误'}",font=font(22),fill='#344966')
 for x,title,color in [(25,'动作前：真实双相机','#315e9e'),(505,'动作前已保存的预测后果','#bd7b13'),(990,'动作后：真实双相机','#24836c')]:d.text((x,105),title,font=font(26),fill=color)
 before=windows[meta['before_window_id']]['public_frames'][-1];after=windows[meta['after_window_id']]['public_frames'][-1];im.paste(cameras(before),(25,150));im.paste(cameras(after),(990,150));d.text((480,170),'事实              预测     概率     真值',font=font(20),fill='#17263d')
 for j,label in enumerate(LABELS):
  y=210+46*j;bad=j in detail['error_columns'];color='#b3273c'if bad else'#246a56';d.text((480,y),label,font=font(21),fill='#263b54');p=detail['prediction'][j];p='未知'if p=='UNKNOWN'else str(p);prob=detail['probabilities'][j];d.text((675,y),p,font=font(22),fill=color);d.text((750,y),'-'if prob is None else f'{prob:.3f}',font=font(20),fill=color);d.text((845,y),str(detail['truth_after'][j]),font=font(22),fill=color)
 d.text((25,590),'事实顺序：持红、持蓝、持粉、红在抽屉、蓝在抽屉、粉在抽屉、抽屉打开',font=font(22),fill='#17263d');d.text((25,630),'动作前真值：'+syms(detail['truth_before']),font=font(22),fill='#315e9e');d.text((25,668),'动作后真值：'+syms(detail['truth_after']),font=font(22),fill='#24836c');d.text((25,716),'图例：1=真，0=假；绿色=预测正确，红色=错误/未知。中栏是符号预测，没有生成未来图像。',font=font(21),fill='#334155');d.text((25,753),'左右为相应端点的最后一帧；模型实际各读取四帧，跨度2.1秒。动作后含96个真实保持步。',font=font(21),fill='#334155');d.text((25,790),'真值仅用于事后评分；预测输入只含动作前RGB、机器人状态和预先声明的动作。',font=font(21),fill='#334155');return im
def video_frame(path,meta,detail,t,end=False):
 im=Image.new('RGB',(1280,720),'#f7f9fc');d=ImageDraw.Draw(im);d.text((20,12),NAME[meta['family']]+' · '+meta['episode_id'],font=font(25),fill='#17263d');d.text((20,52),'实际记录回放 | '+('真实后果已到达'if end else'动作前历史'if t<0 else'执行及96步后保持')+f' | 相对计分动作开始 {t:+.2f} 秒',font=font(22),fill='#315e9e')
 with np.load(path)as z:im.paste(Image.fromarray(z['rgb_static']).resize((480,480)),(20,110));im.paste(Image.fromarray(z['rgb_gripper']).resize((168,168)),(320,420))
 d.text((20,88),'固定相机（大图） / 手腕相机（右下小图）',font=font(18),fill='#42546a');d.text((535,110),'动作开始前已冻结的预测',font=font(25),fill='#bd7b13');d.text((535,150),'事实                  预测          实际后果',font=font(22),fill='#17263d')
 for j,label in enumerate(LABELS):
  y=195+j*46;p=detail['prediction'][j];d.text((535,y),label,font=font(22),fill='#17263d');d.text((760,y),'未知'if p=='UNKNOWN'else str(p),font=font(23),fill='#bd7b13');d.text((925,y),str(detail['truth_after'][j])if end else'结束后揭示',font=font(22),fill=('#b3273c'if j in detail['error_columns']else'#24836c')if end else'#6b7280')
 d.text((535,550),'1=真，0=假；橙色=动作前预测。',font=font(21),fill='#334155');d.text((535,583),'后果绿色=预测正确，红色=预测错误。',font=font(21),fill='#334155');d.text((20,635),'真实仿真帧按30Hz回放，末帧定格3秒；模型计算时物理暂停，没有插入生成图像。',font=font(21),fill='#334155');d.text((20,673),'真值、拒绝原因与实际耗时只用于事后解释，未输入动作前预测器。',font=font(21),fill='#334155');return np.array(im)
def main():
 q=frozen();ev=read(E/'results/evaluation.json');meta=read(E/'data/_private/evaluation_metadata.json');m=read(B/'data_manifest.json');windows={w['window_id']:w for w in m['windows']};details={r['episode_id']:r for r in read(E/'results/pair_details.json')if r['model']=='candidate_seed0'};byep={r['episode_id']:r for r in meta};out=E/'figures/cases';out.mkdir(exist_ok=False);index=[]
 for r in meta:
  detail=details[r['episode_id']];p=out/(r['episode_id']+'.png');triptych(r,detail,windows).save(p);index.append({'episode_id':r['episode_id'],'path':str(p),'all7_correct':detail['all7_correct'],'error_columns':detail['error_columns'],'before_image':windows[r['before_window_id']]['public_frames'][-1],'after_image':windows[r['after_window_id']]['public_frames'][-1]})
 chosen=[]
 for fam,branches in q['paired_branches'].items():
  rows=[r for r in meta if r['family']==fam];first=next((r for r in rows if not details[r['episode_id']]['all7_correct']),rows[0]);eps=[first['root_id']+f'__b{b}'for b in branches];eps=[e for e in eps if e in byep];chosen.append({'family':fam,'root_id':first['root_id'],'episodes':eps,'selection_rule':'first candidate error in fixed order for family, else first covered root; paired branches shown'})
 firsterror=next((r for r in meta if not details[r['episode_id']]['all7_correct']),None)
 if firsterror and not any(firsterror['episode_id']in r['episodes']for r in chosen):chosen.append({'family':'first_error','root_id':firsterror['root_id'],'episodes':[firsterror['episode_id']],'selection_rule':'first candidate error in fixed order'})
 os.environ['IMAGEIO_FFMPEG_EXE']=imageio_ffmpeg.get_ffmpeg_exe();vidmeta=[];priv={r['episode_id']:r for r in m['private_pairs']}
 for group in chosen:
  vp=E/'videos'/(group['family']+'_paired_evidence.mp4');frames=0
  with imageio.get_writer(vp,fps=30,codec='libx264',quality=7,macro_block_size=16,ffmpeg_log_level='error')as writer:
   for eid in group['episodes']:
    r=byep[eid];dt=details[eid];tr=priv[eid];start,end=tr['start_frame'],tr['end_frame'];folder=B/'episodes'/eid/'public'
    for fi in range(start-63,end+1):writer.append_data(video_frame(folder/f'{fi:05d}.npz',r,dt,(fi-start)/30,fi==end));frames+=1
    last=video_frame(folder/f'{end:05d}.npz',r,dt,(end-start)/30,True)
    for _ in range(90):writer.append_data(last);frames+=1
  vidmeta.append(group|{'path':str(vp),'sha256':sha(vp),'frames':frames,'fps':30,'seconds':frames/30,'final_result_freeze_per_episode_seconds':3,'source_frames_real_only':True});print('VIDEO_SAVED '+str(vp),flush=True)
 save(E/'results/case_visuals.json',index);save(E/'results/videos.json',vidmeta)
 # Local HTML index keeps every attempted prediction available, including all errors.
 cards=[]
 for r in index:cards.append(f"<article><h3>{r['episode_id']} · {'正确'if r['all7_correct']else'错误'}</h3><a href='../figures/cases/{r['episode_id']}.png'><img loading='lazy' src='../figures/cases/{r['episode_id']}.png'></a></article>")
 (E/'results/case_gallery.html').write_text('<!doctype html><meta charset="utf-8"><title>D独立测试全部案例</title><style>body{font-family:sans-serif;margin:25px;background:#eef2f7}article{background:white;padding:15px;margin:20px 0}img{width:100%;max-width:1440px}h1{color:#17263d}</style><h1>全部动作前—预测后果—真实后果</h1><p>固定候选 seed0；每张图说明颜色、符号和输入。12个独立根内的七个分支相关；这些图片不是84个独立根。</p>'+''.join(cards))
 print(json.dumps({'case_images':len(index),'error_images':sum(not r['all7_correct']for r in index),'videos':len(vidmeta)},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
