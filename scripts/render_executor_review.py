"""Fixed-seed, outcome-blind review samples from completed paired real executions."""
from pathlib import Path
import json,random,collections,html,datetime,hashlib
import cv2,numpy as np,imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont
P=Path('/home/robot/research/projects/wm_symbolic');source=P/'runs/executor_visual_stop_v2_paired';data=json.loads((source/'results.json').read_text());assert data['status']=='complete' and len(data['pairs'])==30
out=P/'reports/executor_review_v2';out.mkdir(exist_ok=False);assets=out/'assets';assets.mkdir();font_path='/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc';font=ImageFont.truetype(font_path,22);small=ImageFont.truetype(font_path,17);big=ImageFont.truetype(font_path,28)
zh={'open_drawer':'打开抽屉','close_drawer':'关闭抽屉','lift_red_block_table':'从台面抓起红块','lift_blue_block_table':'从台面抓起蓝块','lift_pink_block_table':'从台面抓起粉块'};color_zh={'block_red':'红块','block_blue':'蓝块','block_pink':'粉块'};reason_zh={'visual_completed':'视觉检测完成后停止','wrong_visible_object_refusal':'检测到错误颜色后拒绝继续','ambiguous_visible_holding_refusal':'持物视觉判断冲突，拒绝继续','fixed_budget_exhausted':'已用完固定控制步预算'}
candidates=sorted([r for r in data['trials'] if r['executor_version']=='v2_visual_stop'],key=lambda r:(r['skill'],r['root_index']));rng=random.Random(20261013);selected=[]
# Stratify only by action name, never by success, confidence, color accuracy or length.
for skill in sorted(zh):selected.append(rng.choice([r for r in candidates if r['skill']==skill]))
selected.append(rng.choice([r for r in candidates if r not in selected]));rng.shuffle(selected)
selection={'seed':20261013,'method':'For each of five skill names, uniformly choose one of six roots, then uniformly choose one remaining execution; shuffle display order. No success/failure field is used.','selected':[{'root_index':r['root_index'],'skill':r['skill'],'run':r['run']} for r in selected],'population':30,'source_results_sha256':hashlib.sha256((source/'results.json').read_bytes()).hexdigest()};(out/'selection.json').write_text(json.dumps(selection,ensure_ascii=False,indent=2))
def pair_for(r):return next(x for x in data['trials'] if x['root_index']==r['root_index'] and x['skill']==r['skill'] and x['executor_version']=='v1_fixed_budget')
def raw_pair(folder,index):
 if index<0:
  with np.load(folder/'before.npz') as z:return z['rgb_static'].copy(),z['rgb_gripper'].copy()
 images=[]
 for camera in ['rgb_static','rgb_gripper']:
  capture=cv2.VideoCapture(str(folder/f'{camera}.mp4'));capture.set(cv2.CAP_PROP_POS_FRAMES,index);ok,img=capture.read();capture.release();assert ok,(folder,index,camera);images.append(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
 return images
def layout(left,right,top,bottom):
 frame=Image.new('RGB',(800,520),'#ffffff');draw=ImageDraw.Draw(frame);draw.text((18,8),top,font=font,fill='#17283d');draw.text((18,44),'静态相机',font=small,fill='#52647a');draw.text((418,44),'夹爪相机',font=small,fill='#52647a');frame.paste(Image.fromarray(left).resize((396,396)),(2,74));frame.paste(Image.fromarray(right).resize((396,396)),(402,74));draw.text((18,483),bottom,font=small,fill='#17283d');return frame
def make_video(r,name,number):
 folder=Path(r['run']);readers=[cv2.VideoCapture(str(folder/f'{k}.mp4')) for k in ['rgb_static','rgb_gripper']];dest=assets/name;writer=imageio.get_writer(str(dest),fps=30,codec='libx264',macro_block_size=1,quality=8);actual=0
 label='原版固定预算' if r['executor_version']=='v1_fixed_budget' else '候选视觉停止'
 try:
  for step in range(r['actual_steps']):
   frames=[]
   for capture in readers:
    ok,img=capture.read();assert ok,(name,step);frames.append(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
   combined=layout(*frames,f'样例{number} · C{r["root_index"]} · {zh[r["skill"]]}',f'{label} | 控制步 {step+1}/{r["actual_steps"]} | 仿真时间 {(step+1)/30:.2f} 秒')
   writer.append_data(np.asarray(combined));actual+=1
 finally:
  writer.close()
  for capture in readers:capture.release()
 assert actual==r['actual_steps'];return {'file':str(dest),'frames':actual,'fps':30,'input_dual_camera_frames_preserved':True,'no_time_compression':True}
videos=[];cards=[];thumbs=[]
for i,r in enumerate(selected,1):
 before=pair_for(r);v1=f'{i:02d}_baseline.mp4';v2=f'{i:02d}_candidate.mp4';videos.append(make_video(before,v1,i));videos.append(make_video(r,v2,i));folder=Path(r['run']);panels=[]
 for label,index in [('执行前',-1),('过程一半',max(0,r['actual_steps']//2-1)),('执行结束',r['actual_steps']-1)]:
  panel=layout(*raw_pair(folder,index),f'样例{i} C{r["root_index"]} · {zh[r["skill"]]}',f'{label} | '+('0秒' if index<0 else f'{(index+1)/30:.2f}秒'));panels.append(panel.resize((640,416)))
 sheet=Image.new('RGB',(1920,416),'white')
 for j,part in enumerate(panels):sheet.paste(part,(j*640,0))
 sheet.save(assets/f'{i:02d}_frames.png');thumbs.append(sheet)
 evidence='；'.join(f'{color_zh[name]}：正接触{value["positive_robot_contact_steps"]}步，最高升高{value["max_height_rise_m"]*1000:.1f}毫米' for name,value in r['actual_object_evidence_evaluator_only'].items())
 success='成功' if r['official_success_at_end'] else '失败';b_success='成功' if before['official_success_at_end'] else '失败'
 cards.append(f'''<section><h2>随机样例{i}｜C{r['root_index']}｜{zh[r['skill']]}</h2><p>输入指令：<b>{html.escape(r['command'])}</b></p><p>原版：{before['actual_steps']}步、{b_success}。候选：{r['actual_steps']}步、<b>{success}</b>；{reason_zh[r['stop_reason']]}。</p><div class="videos"><div><h3>原版：完整双相机过程</h3><video controls preload="metadata" src="assets/{v1}"></video></div><div><h3>候选：完整双相机过程</h3><video controls preload="metadata" src="assets/{v2}"></video></div></div><button onclick="playPair(this)">两段从头一起播放</button><p class="evidence">实际对象的事后评价证据：{evidence}。接触或升高不自动等于稳定抓住；目标技能成功由独立检查器评分，请结合完整视频判断。真值仅用于这段说明，未参与停止。</p><details><summary>查看执行前／中间／结束的双相机截图</summary><img src="assets/{i:02d}_frames.png" alt="真实双相机关键帧"></details></section>''')
summary=collections.defaultdict(lambda:[0,0,0,0,0])
for r in data['pairs']:
 v=summary[r['skill']];v[0]+=r['v1_final_success'];v[1]+=r['v2_final_success'];v[2]+=1;v[3]+=r['v1_controls'];v[4]+=r['v2_controls']
table=''.join(f'<tr><td>{zh[name]}</td><td>{v[0]}/{v[2]}</td><td>{v[1]}/{v[2]}</td><td>{v[3]/v[2]:.0f} → {v[4]/v[2]:.0f}</td></tr>' for name,v in summary.items());v2all=[r for r in data['trials'] if r['executor_version']=='v2_visual_stop'];visual=[r for r in v2all if r['visual_claimed_completion']];false=sum(not r['official_success_at_end'] for r in visual);regressions=[r for r in data['pairs'] if r['v1_final_success'] and not r['v2_final_success']]
for part in range(2):
 contact=Image.new('RGB',(1920,1248),'white')
 for j,im in enumerate(thumbs[part*3:part*3+3]):contact.paste(im,(0,j*416))
 contact.save(out/f'随机样例_{part+1}.png')
page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>执行器修复候选：随机完整过程复核</title><style>body{max-width:1240px;margin:30px auto;padding:0 22px;font:16px/1.6 system-ui,sans-serif;background:#f4f7fb;color:#17283d}h1{font-size:29px}h2{font-size:22px}h3{font-size:16px}section{background:#fff;padding:20px;margin:22px 0;border-radius:12px}table{border-collapse:collapse;width:100%;background:white}td,th{text-align:left;padding:12px;border-bottom:1px solid #dde5ee}.videos{display:grid;grid-template-columns:1fr 1fr;gap:20px}video,img{width:100%}.warning{border-left:5px solid #b5661e;background:#fff3e3;padding:16px}.evidence{font-size:14px;color:#4d6077}button{margin-top:14px;padding:9px 17px;font-size:15px}small{color:#58687a}@media(max-width:760px){.videos{grid-template-columns:1fr}}</style><h1>执行器修复候选：请查看随机完整过程</h1>'''
page+=f'''<p>生成：{datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')}。全部为本机真实执行；原始HULC权重未改变。</p><div class="warning"><b>部分改善，尚未全部修好。</b>抽屉完成后可提前停止；颜色选择问题仍存在，且粉块有一例原本最终成功、候选拒绝后未完成。候选未用于正式重采或训练，等待您的视觉复核。新旧轨迹分目录保存，旧4215条数据未删除或混入新版本。</div><h2>全体30组配对结果，不只看抽样</h2><table><tr><th>命令</th><th>原版成功</th><th>候选成功</th><th>平均控制步：原版 → 候选</th></tr>{table}</table><p>每项命令使用6个C初态，前3个复用已记录的颜色试验初态，后3个是新初态。新旧版本共享初态、指令、策略随机种子和最大预算；全部30组的初始RGB/机器人量及停止前动作前缀完全一致，因此差别来自停止层。候选共有{len(visual)}次视觉完成声明，其中{false}次被独立检查器判为失败；这只是小样本验证，不是普遍保证。</p><p>视觉层只读取公开RGB-D、匿名可见mask、相机标定与机器人自身量；抽屉视觉几何参照仅在C开发记录标定后冻结。检测到错误颜色时是拒绝继续，不是纠正抓取。未识别的情形保留原最大预算。</p><h2>以下6段如何抽取</h2><p>固定随机种子20261013：五种命令各从六个初态中均匀抽一条，再从余下记录抽一条，并随机排列。抽取不读取成功/失败字段。每段保留从首次控制到实际停止的全部双相机帧，30帧/秒对应30Hz仿真控制，没有加速或删去失败片段。左列与右列可同步从头播放，较短的一段会自然结束。</p>'''+''.join(cards)
page+='''<h2>您可以重点看什么</h2><p>抽屉是否到位后及时停下；抬取指令是否操作正确颜色；拒绝停止前是否已经抓起错误对象；结束时是否仍稳定持物。自动证据不足以确认的动作，请直接按视频指出问题。未得到复核结论前，不启动昂贵正式阶段。</p><small>可复现抽样清单：selection.json。完整试验：runs/executor_visual_stop_v2_paired/results.json。所有原始视频、控制和评价轨迹保留。</small><script>function playPair(button){let videos=button.closest('section').querySelectorAll('video');videos.forEach(v=>{v.pause();v.currentTime=0;v.play()})}</script></html>'''
(out/'index.html').write_text(page);(out/'video_integrity.json').write_text(json.dumps(videos,indent=2));(out/'summary.json').write_text(json.dumps({'all_pairs':30,'summary':dict(summary),'visual_completion_claims':len(visual),'false_visual_claims_in_these_trials':false,'regressions':regressions,'all_pairs_same_root_and_action_prefix':True,'not_promoted':True,'user_review_required':True},ensure_ascii=False,indent=2));print('EXECUTOR_REVIEW_READY',str(out/'index.html'),json.dumps(selection['selected'],ensure_ascii=False))
