"""Render stored endpoint comparisons. Never invoke environment/model or synthesize motion."""
from audit import *
from PIL import Image, ImageDraw, ImageFont
import imageio.v2 as imageio
import html
FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
BG='#f2f5f8';INK='#15283b';MUTED='#526579';BLUE='#287bb5';GREEN='#16816b';RED='#c74545'
def font(n): return ImageFont.truetype(FONT,n)
def txt(im,xy,s,size=22,color=INK):ImageDraw.Draw(im).text(xy,s,font=font(size),fill=color)
def truth(d):
 y=d['actual_labels_evaluator_only']['y'];held=','.join(c for c,b in zip(['红','蓝','粉'],y[:3])if b)or'无';inside=','.join(c for c,b in zip(['红','蓝','粉'],y[3:6])if b)or'无'
 return f"实际：持物={held}；抽屉={['关','开','中间'][int(np.argmax(y[6:]))]}；入屉={inside}"
def panel(im,path,x,y,w=600):
 with np.load(path)as a:
  im.paste(Image.fromarray(a['rgb_static']).resize((360,360),Image.Resampling.NEAREST),(x,y))
  im.paste(Image.fromarray(a['rgb_gripper']).resize((210,210),Image.Resampling.NEAREST),(x+376,y))
 txt(im,(x,y+366),'固定相机（原始200×200）',18)
 txt(im,(x+376,y+218),'腕部相机',18);txt(im,(x+376,y+243),'原始84×84',18)
def main():
 out=R/'videos';out.mkdir(parents=True,exist_ok=True);rows=read(OLD/'results/closed_loop.json');manifest=[]
 for suffix in ['red_1','blue_1','pink_1','pink_0']:
  rid='D_interaction_chain_'+suffix;row=next(r for r in rows if r['root_id']==rid and r['kind']=='interaction');base=OLD/'data/closed_loop'/rid/'interaction';actions=read(base/'_private/actions.json');decisions=read(base/'_private/decisions.json');frames=[];sources={};cards=[]
  for i,a in enumerate(actions):
   before=base/'public'/f'boundary{i:02d}.npz';after=base/'public'/f'boundary{i+1:02d}.npz'
   for p in [before,after]:sources[str(p)]=sha(p)
   im=Image.new('RGB',(1280,900),BG);txt(im,(28,15),'真实端点对比动画（非连续运动录像）',32)
   txt(im,(28,64),f'{rid}  |  交互模型 seed 0  |  第 {i+1}/{len(actions)} 次调用',22)
   txt(im,(28,102),f"{a['skill']}   执行结果：{a['native']['status']}   原因：{a['native'].get('reason')or'无'}",20,GREEN if a['native']['status']=='success'else RED)
   txt(im,(28,144),f"调用前：boundary{i:02d} / 控制帧 {a['start_frame']}",22,BLUE);txt(im,(660,144),f"调用后：boundary{i+1:02d} / 控制帧 {a['end_frame']}",22,BLUE)
   panel(im,before,28,184);panel(im,after,660,184)
   for x,d in [(28,decisions[i]),(660,decisions[i+1])]:
    txt(im,(x,590),truth(d),20)
    pr=d['current_probability'];target=['red','blue','pink'].index(row['target'])
    txt(im,(x,625),f'当前读出 P(目标入屉)={pr[3+target]:.3f}',20)
    txt(im,(x,657),f'当前读出 P(关闭)={pr[6]:.3f}',20)
   predicted=decisions[i]['search']['score'];txt(im,(28,704),f'选中计划的预测终点目标分数：{predicted:.3f}；接受阈值0.8。它不是当前实际状态。',21)
   txt(im,(28,747),f"整段实际结局：{'目标成功'if row['physical_strict_goal_success']else'目标失败'}；模型停止声明：{'完成'if row['reported_completion']else'未完成'}。",23,GREEN if row['physical_strict_goal_success']else RED)
   txt(im,(28,793),'左右为同一次调用的真实存档图；端点间没有RGB录像，未补帧、未重新执行。',20,MUTED)
   txt(im,(28,827),'图像按最近邻放大；每张对比卡展示4秒，播放时长不代表机器人运行时长。',20,MUTED)
   txt(im,(28,859),'“实际”来自独立物理评分，仅用于展示；未参与原计划搜索或停止判断。',18,MUTED)
   frames.append(im);card=out/f'{suffix}_step{i+1:02d}.png';im.save(card);cards.append(str(card))
  video=out/f'{suffix}_endpoint_comparison.mp4';writer=imageio.get_writer(video,fps=12,codec='libx264',quality=8,macro_block_size=None,ffmpeg_params=['-pix_fmt','yuv420p','-movflags','+faststart'])
  for im in frames:
   for _ in range(48):writer.append_data(np.asarray(im))
  writer.close();reader=imageio.get_reader(video);count=reader.count_frames();meta=reader.get_meta_data();assert count==48*len(frames);assert reader.get_data(0).shape==(900,1280,3);assert reader.get_data(count-1).shape==(900,1280,3);reader.close()
  item={'root_id':rid,'kind':'interaction','seed':0,'physical_success':row['physical_strict_goal_success'],'video':str(video),'sha256':sha(video),'source_endpoint_sha256':sources,'source_actions_sha256':sha(base/'_private/actions.json'),'source_decisions_sha256':sha(base/'_private/decisions.json'),'calls':len(actions),'frames':count,'fps':meta['fps'],'duration_seconds':meta['duration'],'format':'real endpoint comparison slideshow; no continuous motion footage','display_seconds_per_step':4,'interpolation':False,'new_physical_execution':False,'cards':cards};manifest.append(item);print('VIDEO',suffix,count,meta['duration'],flush=True)
 save(E/'results/media_manifest.json',manifest)
 # Development matched states: both cameras, no changed pixels apart from nearest-neighbour display sizing.
 cases=read(E/'results/extra_matched_place.json');dev=[r for r in cases if r['split']=='dev'and r['condition']in['held_open','held_closed']];im=Image.new('RGB',(1280,1140),BG)
 txt(im,(24,16),'已有C数据：持物时，抽屉开/关的真实输入对照',30)
 txt(im,(24,62),'左列=打开；右列=关闭。每格左图固定相机、右图腕部相机；没有后继图或GT输入。',20,MUTED)
 for j,root in enumerate(sorted({r['root_id']for r in dev})):
  y=120+j*500;txt(im,(24,y-16),root,23)
  for x,cond,label in [(24,'held_open','开'),(660,'held_closed','关')]:
   r=next(r for r in dev if r['root_id']==root and r['condition']==cond);txt(im,(x,y+26),f'实际抽屉={label}，已持物；该端点仅做冻结推理',20,BLUE);panel(im,r['path'],x,y+68)
 txt(im,(24,1100),'意义：图像与特征并未完全相同；两根样本不足以证明任何模型在所有持物视角都可靠。',20,MUTED)
 im.save(R/'matched_C_inputs.png')
 # Gallery keeps the previous report intact and links its gallery rather than modifying it.
 cards=[]
 for r in manifest:
  v=Path(r['video']);poster=Path(r['cards'][0]);label=('成功'if r['physical_success']else'失败')+'：'+r['root_id']
  cards.append(f'<section><h2>{html.escape(label)}</h2><p>{r["calls"]}次真实调用；{r["duration_seconds"]:g}秒对比动画。</p><video controls preload="metadata" poster="videos/{poster.name}" src="videos/{v.name}"></video><p><a href="videos/{v.name}">下载MP4</a></p></section>')
 oldgallery=next(iter((P/'reports/interaction_latent_query_20261016').glob('*.html')),None)
 oldlink=f'<a href="../interaction_latent_query_20261016/{oldgallery.name}">首版原图集（保留）</a>'if oldgallery else'<a href="../interaction_latent_query_20261016/">首版报告目录（保留）</a>'
 doc='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>首版端点动画与只读复核</title><style>body{font-family:system-ui,sans-serif;max-width:1240px;margin:40px auto;padding:0 24px;background:#f2f5f8;color:#15283b;line-height:1.7}section{background:white;padding:22px;margin:24px 0;border-radius:12px}video,img{width:100%;height:auto}h1{font-size:28px}h2{font-size:20px}a{color:#1766a2}.note{border-left:5px solid #c74545;padding:12px 20px;background:#fff}</style><h1>真实端点对比动画与只读复核</h1><p class="note">这些是当时保存的动作前后端点图，不是连续运动录像。没有新执行、补帧或隐藏失败调用。每张卡固定停留4秒，不能把播放时长当作机器人耗时。</p><p>颜色：蓝字表示端点标题；绿字表示执行或整体成功；红字表示拒绝或整体失败。物块本身的红、蓝、粉是任务对象身份。概率是模型输出；实际状态是独立评分。</p><p><a href="REVIEW.md">完整只读分析</a> · <a href="PROPOSED_PROTOCOL.md">未执行的修正协议</a> · '''+oldlink+'</p>'+''.join(cards)+'<section><h2>C输入可区分性对照</h2><img src="matched_C_inputs.png"><p>这是2个已有C独立根、每根2个条件的输入示例，不能当作4个独立测试根。</p></section></html>'
 (R/'index.html').write_text(doc)
if __name__=='__main__':main()
