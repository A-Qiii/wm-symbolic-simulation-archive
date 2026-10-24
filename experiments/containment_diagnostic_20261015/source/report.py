from common import *
import gzip,collections,html,shutil,copy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties,fontManager
from PIL import Image,ImageDraw,ImageFont
import imageio.v2 as imageio
FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc';fontManager.addfont(FONT)
plt.rcParams.update({'font.family':FontProperties(fname=FONT).get_name(),'font.size':11,'axes.unicode_minus':False})
def table(headers,rows):return '| '+' | '.join(headers)+' |\n|'+'|'.join(['---']*len(headers))+'|\n'+'\n'.join('| '+' | '.join(map(str,row))+' |'for row in rows)+'\n'
def figsave(f,n):f.savefig(R/'figures'/n,dpi=160,bbox_inches='tight');plt.close(f)
def camera(ax,path,title):
 z=np.load(path);ax.imshow(z['rgb_static']);ax.set_title(title,fontsize=10);ax.axis('off');sub=ax.inset_axes([.7,0,.3,.3]);sub.imshow(z['rgb_gripper']);sub.axis('off')
def main():
 geo=read(E/'results/geometry_visibility.json');st=read(E/'results/state_sufficiency.json');calls=read(E/'results/close_calls.json');pr=read(E/'results/probes.json');rows=read(E/'data/object_window_audit.json');counter=read(E/'results/C_41_error_details.json');wins={r['window_id']:r for r in read(BASE/'data/windows.json')};meta=read(E/'data/object_query_metadata.json');truth=dict(np.load(E/'data/_private/probe_targets_and_GT_features.npz'));scores=dict(np.load(E/'results/probe_scores.npz'));roots={r['root_id']:r for r in read(BASE_B/'roots_manifest.json')}
 # Correctly leave balanced accuracy undefined for one-class subsets. Original raw output retained.
 adjusted=copy.deepcopy(pr)
 for r in adjusted['results'].values():
  for k in ['AB','C','C_outside_supported','C_indirect_protruding']:
   if r[k]['positive']==0 or r[k]['negative']==0:r[k]['balanced_accuracy']=None
 adjusted['reporting_note']='One-class subset balanced accuracy is undefined; use raw FP/TN counts. Raw prior serialization is retained. This changes no score, fit or threshold.';save(R/'tables/probe_metrics_for_report.json',adjusted)
 fig,axs=plt.subplots(1,2,figsize=(12,4.6));xx=np.arange(3);groups=['edge_straddle','stacked','outside_front_supported'];labels=['边沿初始摆放后\n实际仍突出','堆叠相关窗口','外沿承重窗口'];colors=['#bd9c4b','#b56354','#5866a4']
 for i,(g,label,col)in enumerate(zip(groups,labels,colors)):
  a=geo['by_condition'][g]['protrusion_mm'];axs[0].bar(i,a['median'],color=col);axs[0].errorbar(i,a['median'],yerr=[[a['median']-a['min']],[a['max']-a['median']]],fmt='none',ecolor='#333',capsize=5);axs[0].text(i,a['max']+1.5,f"n={a['n']}\n{a['min']:.2f}–{a['max']:.2f}",ha='center',fontsize=10)
 axs[0].axhline(5,color='#444',ls='--',label='5mm仅作敏感性参照');axs[0].set_xticks(xx,labels);axs[0].set(ylabel='最大越界量（mm）',ylim=(0,52),title='41个误报均高出上沿，不是毫米级轻触');axs[0].legend(fontsize=9)
 axs[1].bar([0,1],[41,40],color=['#277f87','#a7c1c4']);axs[1].set_xticks([0,1],['静态相机','腕部相机']);axs[1].set(ylabel='能提取目标颜色区域的窗口数 / 41',ylim=(0,46),title='颜色可见，不等于全部几何边界可测');axs[1].text(0,42,'41/41；156–386像素',ha='center',fontsize=10);axs[1].text(1,41,'40/41；0–212像素',ha='center',fontsize=10)
 fig.text(.5,-.07,'左：柱为中位数、误差线为实际最小至最大；三色是不同构造条件。右：两相机分别统计当前帧，窗口同根相关，不能当41次独立试验。',ha='center',fontsize=10);figsave(fig,'protrusion_and_visibility.png')
 names=['frozen_edge_latent384','existing_pooled_visual3852','cached_spatial_patch_visual15372','RGB_bbox_geometry12','GT_relative_geometry_upper_bound'];short=['时序表示\n384维','池化缓存\n3852维','空间patch\n15372维','RGB框\n12维','GT参考\n8维'];cols=['#a26b55','#468b96','#276c73','#b79b5e','#8b8995'];fig,axs=plt.subplots(1,3,figsize=(15,5.5))
 for i,n in enumerate(names):
  rr=pr['results'][n];vals=[rr['C']['FP']+rr['C']['FN'],rr['C_outside_supported']['FP'],rr['paired_same_object']['dev']['both_correct']]
  for ax,val in zip(axs,vals):ax.bar(i,val,color=cols[i]);ax.text(i,val+.6,str(val),ha='center')
 for ax,title,ylabel in zip(axs,['C全部In错误（629事实）','C越界承重误报（41事实）','C同根同对象正负对都正确（12对）'],['错误事实数，越低越好','误报事实数，越低越好','配对数，越高越好']):ax.set_title(title,fontsize=11);ax.set_ylabel(ylabel);ax.set_xticks(range(5),short,fontsize=9);ax.grid(axis='y',alpha=.15)
 axs[2].set_ylim(0,14);fig.text(.5,-.06,'五组均只用AB拟合、固定同一线性岭模型和阈值，C不选参数。GT组是特权输入的线性探针，不是性能上界；不同维度和历史输入使其不构成公平架构竞赛。',ha='center',fontsize=10);figsave(fig,'feature_probe_comparison.png')
 # Directly examine the new real task failures, rather than labelling Close itself a failure.
 ext=[r for r in calls if r['origin']=='extension'];drop=[]
 for a in ext:
  summary=read(Path(a['data_base'])/'episodes'/a['episode_id']/'_private/summary.json');c=roots[a['root_id']]['color'];corners=summary['final_labels']['objects'][c]['world_corners'];drop.append({'episode_id':a['episode_id'],'environment':summary['environment'],'target':c,'closed':a['closed'],'target_In':bool(summary['final_labels']['y'][3+COLORS.index(c)]),'target_support':bool(summary['final_labels']['objects'][c]['load_bearing_paths']),'world_corner_z_mm':[min(v[2]*1000 for v in corners),max(v[2]*1000 for v in corners)]})
 save(R/'tables/extension_task_outcomes.json',drop);a=next(r for r in ext if r['split']=='dev'and roots[r['root_id']]['color']=='red');folder=Path(a['data_base'])/'episodes'/a['episode_id']
 with gzip.open(folder/'_private/full_physics_trace.json.gz','rt')as f:trace=json.load(f)
 falling=next(r['frame']for r in trace[a['start']:]if r['physical']['objects']['block_red']['position'][2]<.27);fig,axs=plt.subplots(1,3,figsize=(12,4.9))
 for ax,frame,title in zip(axs,[a['start'],falling,a['end']],['Close前：外沿承重','执行中：离开抽屉并下落','Close后：抽屉关上，物块落地']):camera(ax,folder/'public'/f'{frame:05d}.npz',title+f'\n控制帧{frame}')
 fig.text(.5,-.03,'C红色物块的真实相机序列，右下为腕部相机；末帧目标可能离开视野。落地由保存的GT角点高度约0–40mm、无抽屉支撑共同确认，未给视觉探针输入。',ha='center',fontsize=10);figsave(fig,'close_success_storage_failure.png')
 counts=st['relation_outcomes'];fig,ax=plt.subplots(figsize=(11,4.4));keys=list(counts);lab=['AB不突出','AB突出','C不突出','C突出']
 for i,k in enumerate(keys):
  rr=counts[k];vv=[rr['closed_and_contained'],rr['closed']-rr['closed_and_contained'],rr['not_closed']];bottom=0
  for j,(value,col,text)in enumerate(zip(vv,['#277f87','#b69546','#b56354'],['关严且原关联物块均在内','关严但有物块未收纳','未关严'])):
   ax.bar(i,value,bottom=bottom,color=col,label=text if i==0 else None)
   if value:ax.text(i,bottom+value/2,str(value),ha='center',color='white')
   bottom+=value
 ax.set_xticks(range(4),lab);ax.set(ylabel='真实Close调用数（合计90）',title='“高出抽屉上沿”解释风险，但不是确定的失败前提');ax.legend(fontsize=9);fig.text(.5,-.07,'突出=执行前任一由抽屉承重的物块高出上沿>2mm，全部由GT离线计算。绿色、金色、红色分别表示三类实际结局；空抽屉属于无物块掉落。',ha='center',fontsize=10);figsave(fig,'protrusion_and_actual_outcome.png')
 pair=[next(r for r in calls if r['episode_id']==f'C_semantics_v2_fixtures_red_{i}__stacked')for i in [0,1]];fig,axs=plt.subplots(2,3,figsize=(12,8));pairinfo=[]
 for ix,r in enumerate(pair):
  root=roots[r['root_id']];ff=Path(r['data_base'])/'episodes'/r['episode_id'];contact=r['first_other_fixed_link_contact']['frame'];g=next(x for x in r['object_geometry']if x['object']=='red');yaw=root['cavity_local_offsets']['yaw']*180/np.pi;pairinfo.append({'episode_id':r['episode_id'],'initial_local_centroid_mm':(np.array(g['before_local_centroid'])*1000).tolist(),'initial_requested_yaw_degrees':yaw,'height_above_rim_mm':r['max_above_rim_mm'],'first_fixed_contact':r['first_other_fixed_link_contact'],'final_q_cm':r['end_q_cm'],'final_red_above_rim_mm':g['after_above_rim_mm'],'native':r['native']})
  for ax,frame,title in zip(axs[ix],[r['start'],contact,r['end']],[f"起点：高出{r['max_above_rim_mm']:.2f}mm\n复位yaw={yaw:.1f}°",'首次碰到固定柜体',f"终点：开度{r['end_q_cm']:.2f}cm\n"+('红块滑落后在内'if r['closed']else'红块仍突出、关闭受阻')]):camera(ax,ff/'public'/f'{frame:05d}.npz',title)
 fig.text(.5,-.01,'上：失败；下：成功。九事实和“突出=True”相同，连续位置与朝向不同，所以不是同一物理初态的重复试验；不能把差异直接归为控制噪声。右下均为腕部相机。',ha='center',fontsize=10);figsave(fig,'same_symbols_different_geometry.png');save(R/'tables/representative_pose_pair.json',pairinfo)
 fig,axs=plt.subplots(1,2,figsize=(12,4.4))
 for r,col,label in zip(pair,['#b56354','#277f87'],['红：最终受阻','青：滑落重排后关严']):
  ts=r['trajectory'];x=[(z['frame']-r['start'])/30 for z in ts]
  for ax,key in zip(axs,['q_cm','max_above_rim_mm']):ax.plot(x,[z[key]for z in ts],color=col,label=label);ax.axvline((r['first_other_fixed_link_contact']['frame']-r['start'])/30,color=col,ls='--',alpha=.5)
 axs[0].set(ylabel='真实抽屉开度（cm）',title='同一固定控制器的实际轨迹');axs[1].set(ylabel='原关联物块最大突出量（mm）',title='成功例在接触后重排，失败例仍突出')
 for ax in axs:ax.set_xlabel('Close调用后的控制时间（秒）');ax.legend(fontsize=9);ax.grid(alpha=.15)
 fig.text(.5,-.06,'虚线为各次首次固定柜体接触；曲线来自30Hz保存的物理trace，止于各自技能返回。接触与重排是观察事实，具体姿态怎样导致分岔仍是机制推断。',ha='center',fontsize=10);figsave(fig,'contact_and_rearrangement_trace.png')
 errors=[];name='cached_spatial_patch_visual15372'
 for i,m in enumerate(meta):
  if m['split']=='dev'and truth['v'][i]and (scores[name][i]>=0)!=bool(truth['y'][i]):errors.append(m|{'score':float(scores[name][i]),'truth_In':bool(truth['y'][i])})
 save(R/'tables/remaining_patch_probe_errors.json',errors);unique={}
 for rr in errors:unique[(rr['root_id'],rr['condition'],rr['object'])]=rr
 fig,axs=plt.subplots(1,3,figsize=(12,5.2))
 for ax,m in zip(axs,unique.values()):
  w=wins[m['window_id']];g=w['targets_v2_1']['objects'][m['object']];camera(ax,w['public_frames'][-1],f"{m['object']} / {m['condition']}\nGT In={m['truth_In']}；探针分数={m['score']:.3f}");ax.text(.5,-.04,f"完整容纳余量 {g['minimum_side_rim_clearance_m']*1000:.2f}mm",ha='center',transform=ax.transAxes)
 fig.text(.5,-.06,'5个错误窗口归并为3个根／条件／对象案例展示，重复窗口不算独立证据。线性分数≥0判In，不是校准概率；没有依据C错误继续拟合或调阈值。',ha='center',fontsize=10);figsave(fig,'remaining_patch_probe_errors.png')
 # Three real videos, rendered only after all CPU probes completed.
 font=ImageFont.truetype(FONT,18);small=ImageFont.truetype(FONT,15);vids=[]
 for r,name,title in [(a,'front_drop','外沿接触：Close成功，但目标掉到地面'),(pair[0],'stack_blocked','相同符号前态之一：堆叠受阻'),(pair[1],'stack_rearranged','相同符号前态之二：滑落重排后关严')]:
  folder=Path(r['data_base'])/'episodes'/r['episode_id'];frames=sorted((folder/'public').glob('*.npz'));out=R/'videos'/(name+'.mp4');writer=imageio.get_writer(out,fps=15,codec='libx264',quality=7,macro_block_size=16);n=0
  for j in sorted(set(range(0,len(frames),6))|{len(frames)-1}):
   z=np.load(frames[j]);im=Image.new('RGB',(880,592),'#f3f5f7');im.paste(Image.fromarray(z['rgb_static']).resize((480,480)),(12,80));im.paste(Image.fromarray(z['rgb_gripper']).resize((336,336)),(528,80));d=ImageDraw.Draw(im);d.text((12,8),title,font=font,fill='#193b48');d.text((12,40),f'真实30Hz控制记录，3倍播放｜时间{j/30:.1f}s｜左静态、右腕部',font=small,fill='#384a56');d.text((520,445),'当前：'+('Close'if r['start']<=j<=r['primitive_end']else'保持'),font=small,fill='#193b48')
   if j==len(frames)-1:d.text((520,478),f"最终原生结果 {r['native']['status']}\n真实开度 {r['end_q_cm']:.2f}cm\n原关联物块均在内 {r['all_associated_contained_after']}",font=small,fill='#9e4a37')
   d.text((12,566),'GT仅作末帧评分；模型没有接收位置、支撑、接触等GT输入。固定脚本采集，非新规划闭环。',font=small,fill='#384a56');writer.append_data(np.asarray(im));n+=1
  writer.close();reader=imageio.get_reader(out);decoded=sum(1 for _ in reader);reader.close();assert decoded==n;vids.append({'path':str(out),'title':title,'episode_id':r['episode_id'],'frames':n,'every_frame_decoded':decoded,'fps':15,'speed':3,'sha256':sha(out)})
 save(R/'videos/manifest.json',vids)
 mr=[]
 for n,label in zip(names,['冻结时序查询表示','现有池化视觉缓存','保留空间patch缓存','RGB框几何','GT几何+支撑，线性诊断']):
  d=pr['results'][n];mr.append([label,d['original_dimension'],f"{d['C']['TP']+d['C']['TN']}/629",d['C']['FP'],d['C']['FN'],f"{d['C_outside_supported']['FP']}/41",f"{d['paired_same_object']['dev']['both_correct']}/12"])
 source_table=table(['输入','实际来源','能说明什么／不能说明什么'],[['冻结时序表示','原冻结编码器与GRU；4帧RGB特征和本体状态','检查现有查询入口的线性可读性；含历史，与单帧探针不完全同条件'],['现有池化缓存','当前帧对象每相机均值、场景每相机2×2格','未经过旧任务的时序瓶颈；不需要新DINO'],['空间patch缓存','当前帧对象2×2 ROI、场景4×4格、RGB框','保留布局；抽屉仍是场景区域，没有精确抽屉实例锚点'],['RGB框几何','两相机目标框坐标、可见性、占据比例，共12维','二维位置不等于对象—抽屉三维相对位置'],['GT相对量','5个内腔有符号余量、底面余量、支撑、Held','只用于特权诊断；禁止混进视觉输入或生产规划']])
 text=f'''# 容纳与关闭：有界诊断结论

本轮实际完成 **9次补充Close、5个预先固定的CPU线性探针、既有90次Close的几何／接触审计**。没有新D、没有神经模型再训练、没有下载模型、没有改变执行器。原版最小链路已跑通；新语义模型仍未通过开发门槛，本报告不声称新规划验证成功。

核心结论是：这批41个In误报不是可忽略的毫米级边缘误差；已有视觉缓存保留了大部分区分信息，当前冻结时序查询表示的读出较弱；一个“突出上沿”关系能提示风险，但不能决定Close必成或必败。

## 1. 越界定义是否服务任务

![越界幅度与可见性](figures/protrusion_and_visibility.png)

41个C越界承重误报来自6个根组，最大越界量10.35–42.19mm，中位22.34mm。全部最严重方向是高出抽屉上沿，而非侧壁极小数值碰撞；5个在10–20mm，36个在20–50mm。按2、5、10mm作**敏感性参照**都没有一个被归入轻微越界。这不为所有未来样本规定容错：小幅误碰是否可忽略，应与最终稳定收纳和真实关闭相连，不能只因一次接触或微小穿透便判任务失败。原标签、容差和旧成绩没有改写。

静态相机41/41提取到目标颜色区域（156–386像素），腕部40/41（0–212像素）。这排除了整物体完全不可见作为这41例的统一原因，却不能证明隐藏角点、上沿深度或完整容纳都能精确量测。原数据仅有RGB和本体状态，没有depth、相机标定或drawer分割。项目中旧`public_geometry.py`需要这些字段，不能把它当作此批RGB已经具有的三维空间输入。

![关上但没有收纳](figures/close_success_storage_failure.png)

外沿接触条件以前只观察静置，没有实际Close。此次按采集前冻结规则选A/B/C各三色、replicate0，共9个既有根的复位扩展；全部Close成功、全部目标物块不在抽屉内且失去抽屉承重，最终角点世界高度约0–40mm，落到地面。由此可见：**不能把所有越界直接等同Close失败，也不能把Close成功等同收纳完成**。这9例的大幅突出确实影响任务，不是可忽略的轻触。

原18组堆叠／并排对照有12组堆叠受阻而并排关严；另6组堆叠通过接触后滑移重排也能关严且收纳。起点未完整容纳不是不可恢复的任务失败，终点才检验物块稳定在内与Closed。边界的±2mm监督不确定带与真实技能结局分开报告。

## 2. 图像信息在哪里变得难以读出

{source_table}

每种输入只做一次相同的加权线性岭拟合：AB按根及正负类平衡，标准化仅用AB，核按特征维度归一化，岭系数0.1，决策分数阈值0；没有C选参。AB1253个有效对象In事实来自18根，C629个有效对象In事实来自9根（141正、488负）；这是对象事实分数，不是九事实完整窗口分数或规划成功率。C曾用于开发，不能称盲测。

![五个固定探针对照](figures/feature_probe_comparison.png)

{table(['固定探针','输入维数','C In正确','C误报','C漏报','41例误报','C同对象正负对均对'],mr)}

12对C关系对照来自6根，每对使用同对象的完整在内与突出／外沿状态；同一正例可能参与两对，不能当12个独立根。冻结时序表示只有0/12对两端均正确；现有池化缓存10/12，保留空间patch为11/12。五种模型没有数据或阈值搜索。单类反例子集只报告FP/TN，平衡准确率没有定义；报告表将该字段设为null，原运行输出保留。

该结果支持“原RGB/DINO缓存有可用空间信息，而经过现有冻结编码后线性读出明显变弱”。它不是编码器彻底丢失信息的数学证明，也不把时序编码、池化、非线性头和样本分布的影响全部混为一个已证实原因。较高维的空间探针不构成公平架构竞赛；但直接使用现有较低维池化缓存也大幅改善，说明无需先下载或更换骨干模型。

GT相对几何组的线性探针仍错36例，不能解释为GT不充分，更不能称该分数为性能上界：完整容纳标签含多边界、承重与非Held的逻辑合取，单一线性超平面不等于该逻辑。GT组是**特权信息参考**；直接按标签公式重算会重现标签，属于定义核查而非视觉能力。任何GT位置／支撑量均未加入四个视觉探针输入。

![空间探针剩余错误](figures/remaining_patch_probe_errors.png)

空间探针仍有3个误报、2个漏报，归并为3个根／条件／对象案例：一个蓝块落入抽屉后的漏报（两个时间窗）、一个蓝块堆叠突出误报（两个时间窗）、一个粉块关闭受阻后误报（腕部已不可见）。没有据这些C例继续调参或训练；仍不能通过原零反例误报门槛，未替换生产查询模型。

## 3. 多一个关系是否足以决定关闭

![突出关系与真实结局](figures/protrusion_and_actual_outcome.png)

诊断关系定义为：执行Close前，由抽屉直接或间接承重的任一物块高出上沿>2mm。这里只用完整物理记录离线计算，尚不是视觉可读的已训练谓词。90次真实Close中，不突出的63次均关严且原关联物块均在内；突出的27次分为6次重排后关严收纳、9次关严但掉落、12次未关严。这是当前样本的观察规律，不能手写为通用动作前提／效果。

加入这个关系后仍有完全相同九事实和相同关系、却不同Close结果的组：AB粉块堆叠4次中1成功3受阻；C红块堆叠2次中1成功1受阻。因而一个“突出”布尔量可以解释风险，却不够作确定后继。

![相同符号但几何不同](figures/same_symbols_different_geometry.png)

代表C红块两例起始突出都约22.34mm，但目标局部横向中心差5.36mm、深度差1.71mm，复位朝向相差约19.6°。两次都在抽屉约13cm开度处首次碰固定柜体；成功例随后滑落重排，最终关严，失败例仍突出并停在约6.13cm。物理初态不相同，不能直接归因随机控制噪声。

![真实接触与重排轨迹](figures/contact_and_rearrangement_trace.png)

保存的轨迹证实接触、滑移、下落、最终姿态及原生失败原因；“朝向与细微摆放改变接触机制进而造成分岔”是有证据支持的机制推断，尚未通过同一精确物理复位重复试验或单变量姿态替换确证。没有精确复位重复的证据，不能量化控制噪声；本轮没有为追逐单例增加新试验。

Open／Lift／Place的新版AB证据仍分别只有6成功、0失败，C各3成功、0失败。新增9例都属于Close原生成功而收纳失败，不能充作前三技能负例，也不能把Close成功强改成失败。动作覆盖缺口真实存在，规则仍不部署。

## 最小下一步与停止边界

本轮诊断已结束，保留原最小链路和新语义模型的失败结论。最小下一步是一个事先冻结的AB/C读出实验：把**已经存在的池化或空间patch缓存直接保留到共享查询入口**，先用AB选择一个方案，保持Holding／抽屉三态回归检查及原失败门槛。不要先更换视觉骨干或增加GT几何；不把本次C较优探针自动晋升为规划模型。

规划侧的下一步是补足新版Open／Lift／Place的真实失败覆盖，并用数据验证突出／接触相关状态能否支持所需后继。若同状态仍存在不同结局，应保留不确定／多后继及实际终点监测，而非手写“突出就不能Close”或“Close必然成功”。控制器、落点接口或整体模型结构若需改动，应单独明确范围。现有结果不支持直接启动新D或宣称新规划通过。

[图表与真实视频](gallery.html) · [运行说明与目录](README.md)。所有旧标签、模型、报告保持原位；本轮源码、固定协议、原始补充帧和完整trace分别位于`{E}`、`{B}`。
''';(R/'REPORT.md').write_text(text)
 for name in ['geometry_visibility','state_sufficiency','C_41_error_details'] :shutil.copy2(E/'results'/(name+'.json'),R/'tables'/(name+'.json'))
 save(R/'tables/diagnostic_summary.json',{'extra_Close_calls':9,'Closed_successes':9,'target_dropped_to_floor':9,'new_D':0,'neural_updates':0,'unique_linear_probe_fits':5,'probe_no_promotion':True,'protrusion_min_median_max_mm':[geo['protrusion_mm'][k]for k in ['min','median','max']],'source_Close_calls_reviewed':90,'state_relation_still_ambiguous':True,'visual_spatial_probe_errors':{'C_FP':3,'C_FN':2,'C_facts':629,'counterexample_FP':3,'counterexample_total':41},'figures':7,'videos':3,'prior_semantics_report':str(P/'reports/state_semantics_v2_20261015/REPORT.md')})
 (R/'README.md').write_text(f'''# 有界诊断运行说明

[结论](REPORT.md) · [图表与视频](gallery.html)

本轮已实际运行9次Close、5个一次性CPU线性探针；无新增D、无神经训练或控制器改动。不能把线性探针的In事实分数当作规划成功率。

|目录|内容|
|---|---|
|`{E}/configs`|采集前协议／哈希、9个原有根、探针拟合前冻结记录|
|`{E}/source`|采集、审计、探针和报告源码；首次日志错误版本保留|
|`{E}/data/public_probe_features.npz`|四种视觉输入，均来自旧公开缓存|
|`{E}/data/_private`|监督和GT参考，禁止加入视觉部署输入|
|`{E}/models`|5个独立线性探针系数，不是生产查询模型|
|`{E}/results`|全部90次Close审计、41例细节、分数、配对和指标|
|`{E}/logs`|物理采集、几何审计、探针首次日志错误与恢复记录|
|`{B}/episodes`|9次固定Close的逐控制RGB、完整物理trace和初末状态|
|`tables`|便于审阅的结果副本、剩余错误和完整性清单|

实际运行顺序：`prepare.py → collect_extension.py → audit.py → probes.py → report.py → post_audit.py → seal.py`。采集／视频用项目`env.sh`，CPU探针用`world_model_env.sh`。GPU采集完成后才拟合探针，所有DINO特征直接复用，无下载。第一次探针已经完成第一个拟合，在打印numpy整数时失败；恢复时校验并复用该模型，只拟合其余四个。没有改变超参数或重选C结果。

复核命令（只读，不重跑采集和拟合）：

```bash
source {P}/world_model_env.sh
python {E}/source/verify_delivery.py
```

先前完整语义报告在`{P}/reports/state_semantics_v2_20261015`，本轮从独立目录引用，不覆盖旧分数。`data/object_query_metadata.json`供审计分析，其中含反例类别；模型拟合只接收对应特征矩阵，未将类别／root／object名作为额外输入。
''')
 vidshtml=''.join(f'<article><h2>{html.escape(v["title"])}</h2><video controls preload="metadata" src="videos/{Path(v["path"]).name}"></video><p>真实固定技能执行，3倍播放；末帧GT仅用于评分，不是新模型闭环。</p></article>'for v in vids)
 figs=''.join(f'<article><a href="figures/{f.name}"><img src="figures/{f.name}"></a></article>'for f in sorted((R/'figures').glob('*.png')))
 (R/'gallery.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>容纳与关闭诊断</title><style>body{max-width:1100px;margin:32px auto;padding:0 20px;background:#f3f5f7;color:#193b48;font:17px/1.6 sans-serif}article{padding:20px;background:white;border-radius:10px;margin:24px 0}img,video{width:100%;height:auto}a{color:#176d83}</style><h1>容纳与关闭：有界诊断</h1><p>9次真实Close、5个固定CPU探针、90次Close记录审计。无新D、无生产模型替换。</p><p><a href="REPORT.md">完整结论</a> · <a href="README.md">证据目录</a></p>'+vidshtml+figs+'</html>')
 print('DIAGNOSTIC_REPORT_GENERATED',str(R),flush=True)
if __name__=='__main__':main()
